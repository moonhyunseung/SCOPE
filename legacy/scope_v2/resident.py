"""Actual resident-KV frontier. Controller waits on the task's exclusive GPU lock.

No network inference, synthetic prefix extension or foreign process termination.
Only same-representation full-model Gate-D passing points enter timing results.
"""
from __future__ import annotations
import argparse,collections,gc,hashlib,json,math,os,sqlite3,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from scope.runner import atomic,canonical,digest,gpu_gate,now
ROOT=Path(__file__).resolve().parents[1]
MODEL='qwen1_5b'
METHODS={'dynamic_eager':('dense','native'),'static_eager':('dense_static','native'),
         'static_sdpa':('dense_static','sdpa'),'int4_global':('global','native'),
         'int4_chunked':('chunked','chunked'),'int4_triton':('triton','triton')}

def source_sha():
    return digest(b''.join(p.name.encode()+b'\0'+p.read_bytes() for p in [Path(__file__),ROOT/'scope_v2/persistent.py',ROOT/'scope_v2/packed_attention.py']))

def plan(data,stage):
    docs=data['documents'][:2];jobs=[]
    grid=[(512,1)] if stage=='pilot' else [(512,b) for b in [1,16,256]]+[(1536,b) for b in [1,16,256,1024,2048,4096,6144]]
    for length,batch in grid:
        for method in METHODS:
            jobs.append(dict(document_id=docs[0]['document_id'],domain=docs[0]['domain'],length=length,batch=batch,method=method,steps=4))
    if stage!='pilot':
        for batch in [1,4]:
            for method in METHODS:jobs.append(dict(document_id=docs[1]['document_id'],domain=docs[1]['domain'],length=1536,batch=batch,method=method,steps=4))
    return jobs

def key(s):return digest(canonical(s).encode())[:16]

def seal(dest,result):
    from scope_v2.diagnose import csv_write
    atomic(dest/'summary.json',result)
    csv_write(dest/'metrics.csv',result.get('timing_repetitions',[]))
    (dest/'report.md').write_text('# Actual resident-KV full-model point\n\n'+json.dumps(result,indent=2)+'\n\nOnly COLLECTED points passed full-model Gate D. Repeated homogeneous inputs and repeated timing trials are not independent quality samples. No serving scheduler or maximum-capacity claim.\n')
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in dest.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',dict(files=files,resident_source_sha256=source_sha(),sealed=now()))
    (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))
    verify(dest)

def verify(dest):
    manifest=json.loads((dest/'manifest.json').read_text())
    for name,meta in manifest['files'].items():
        assert Path(name).name==name
        b=(dest/name).read_bytes();assert len(b)==meta['bytes'] and digest(b)==meta['sha256']
    r=json.loads((dest/'summary.json').read_text())
    if r['status']=='COLLECTED':
        assert len(r['gate_d_steps'])==r['spec']['steps'] and all(s['max_request_relative_l2']<=.025 for s in r['gate_d_steps'])
        assert len(r['timing_repetitions'])==10
        assert all(x['cuda_ms']>0 and x['wall_s']>0 for x in r['timing_repetitions'])
        if r['spec']['method'].startswith('int4'):assert r['resident_final']['dense_persistent_fields']==0
    return r

def available_host():
    mem={a:int(b.split()[0])*1024 for a,b in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    n=mem['MemAvailable']
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':n=min(n,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
    except (OSError,ValueError):pass
    return n

def compare_logits(ref,out):
    """Stream CPU tiles; no B*vocab FP32/FP64 full copy or mean-of-ratios shortcut."""
    import torch
    errors=[];maxabs=0.;same=0
    for start in range(0,len(ref),16):
        a=ref[start:start+16].double();b=out[start:start+16].double()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise RuntimeError('NONFINITE_LOGITS')
        diff=b-a;rel=diff.square().sum(-1).sqrt()/a.square().sum(-1).sqrt().clamp_min(1e-30)
        errors.extend(rel.tolist());maxabs=max(maxabs,float(diff.abs().max()));same+=int((a.argmax(-1)==b.argmax(-1)).sum())
    return dict(max_request_relative_l2=max(errors),mean_request_relative_l2=sum(errors)/len(errors),max_abs=maxabs,top1_agreement=same/len(ref))

def memory():
    import torch
    s=torch.cuda.memory_stats();snap=torch.cuda.memory_snapshot();active=0;requested=0
    for segment in snap:
        for block in segment['blocks']:
            if block['state'].startswith('active'):
                active+=block['size'];requested+=block.get('requested_size',block['size'])
    physical=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
    return dict(allocated_bytes=allocated,reserved_bytes=reserved,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        active_block_bytes=active,active_requested_bytes=requested,active_block_rounding_bytes=active-requested,reserved_minus_active_bytes=reserved-active,
        inactive_split_bytes=s.get('inactive_split_bytes.all.current',0),physical_used_bytes=int(physical[0])*2**20,physical_free_bytes=int(physical[1])*2**20,
        physical_minus_allocator_reserved_bytes=int(physical[0])*2**20-reserved,host_allocator_metadata_bytes=None)

def worker(manifest,folder,specfile):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ['TRITON_CACHE_DIR']=str(ROOT.parent/'triton_cache_continuation_v2')
    import torch,transformers,triton,platform
    from transformers import AutoModelForCausalLM
    from scope_v2.persistent import build_cache,payload,execution
    data=json.loads(manifest.read_text());spec=json.loads(specfile.read_text());dest=folder/'points'/key(spec);dest.mkdir(parents=True,exist_ok=True)
    model_spec=next(m for m in data['models'] if m['key']==MODEL)
    doc=next(d for d in data['documents'] if d['document_id']==spec['document_id'])
    assert doc['split']=='screening','confirmation inputs are forbidden for hardware tuning'
    result=dict(status='RUNNING',spec=spec,source_sha256=source_sha(),model=model_spec,
        input_manifest_sha256=digest(manifest.read_bytes()),gate_d=.025,representation='BF16 native dense' if not spec['method'].startswith('int4') else 'symmetric group32 INT4 codes plus FP32 scales; BF16 reconstructions',
        workload='Physically distinct copies of one real prefix per batch; homogeneous requests, no independent source-n claim. Teacher-forced four-token decode; excludes prefill and scheduling.',
        runtime=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,triton=triton.__version__,cuda=torch.version.cuda))
    model=None;cache=None
    try:
        gpu_gate(dict(reserve_hbm_gib=12,minimum_free_hbm_gib=32,reserve_disk_gib=24))
        torch.set_num_threads(4);torch.manual_seed(20261019);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(96*2**30/total)
        result['runtime'].update(gpu=torch.cuda.get_device_name(),deterministic_algorithms=True,tf32=False,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        model=AutoModelForCausalLM.from_pretrained(model_spec['id'],revision=model_spec['revision'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').eval().cuda()
        storages={p.untyped_storage().data_ptr():p.untyped_storage().nbytes() for p in list(model.parameters())+list(model.buffers())}
        result['model_storage_bytes']=sum(storages.values())
        length=spec['length'];batch=spec['batch'];steps=spec['steps'];tokens=doc['windows'][MODEL]['tokens']
        prefix=tokens[-(length+9):-9];teacher=tokens[-9:];assert len(prefix)==length
        result['token_window_sha256']=digest(canonical(dict(prefix=prefix,teacher=teacher)).encode())
        with torch.inference_mode():
            pre=model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=True,logits_to_keep=1).past_key_values
            states=[(l.keys.cpu(),l.values.cpu()) for l in pre.layers];del pre
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        result['model_runtime_baseline']=memory()
        kind,mode=METHODS[spec['method']];quant=kind not in ['dense','dense_static']
        capacity=math.ceil((length+8)/128)*128 if kind!='dense' else length+steps
        n=sum(k.shape[1]*k.shape[-1] for k,v in states)*batch*capacity*2
        predicted=n*(.5+4/32) if quant else n*2
        # Bound temporary reconstruction/expanded-GQA activations for eager methods.
        hidden=model.config.hidden_size
        workspace=512*2**20+batch*model.config.vocab_size*4
        if mode=='triton':workspace+=batch*model.config.num_attention_heads*math.ceil((length+steps)/128)*128*4
        elif mode=='chunked':workspace+=min(64,batch)*(length+steps)*hidden*6
        else:workspace+=batch*(length+steps)*hidden*6
        workspace+=batch*hidden*64
        free,total=torch.cuda.mem_get_info();limit=min(96*2**30-torch.cuda.memory_allocated(),free-44*2**30)
        host_reference=steps*batch*model.config.vocab_size*2
        result['budget_prediction']=dict(payload_bytes=int(predicted),workspace_upper_estimate_bytes=int(workspace),available_device_budget_bytes=int(limit),host_reference_bytes=host_reference)
        if predicted+workspace>limit or host_reference>available_host()*.35:
            result['status']='SKIPPED_BUDGET';return
        ids=[torch.full((batch,1),teacher[i],device='cuda',dtype=torch.long) for i in range(steps)]
        refkind='chunked' if quant else 'dense_static';refmode='chunked' if quant else 'native'
        cache=build_cache(states,batch,refkind)
        refs=[]
        with torch.inference_mode(),execution(model,refmode,chunk=64,validate_mask=True) as stats:
            for ids_s in ids:
                output=model(input_ids=ids_s,past_key_values=cache,use_cache=True,logits_to_keep=1)
                refs.append(output.logits[:,-1].cpu());del output
        assert stats['calls']==steps*len(states),'native reference hook coverage'
        del cache;cache=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        begin=time.perf_counter();cache=build_cache(states,batch,kind);torch.cuda.synchronize()
        result['cache_materialization_wall_s']=time.perf_counter()-begin
        result['resident_initial']=payload(cache);result['memory_initial']=memory()
        if quant:assert result['resident_initial']['dense_persistent_fields']==0,'dense data persisted in packed cache'
        gate=[]
        with torch.inference_mode(),execution(model,mode,chunk=64,validate_mask=True) as stats:
            for i,ids_s in enumerate(ids):
                output=model(input_ids=ids_s,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cpu=output.logits[:,-1].cpu();del output
                gate.append(compare_logits(refs[i],cpu));del cpu
                assert cache.get_seq_length()==length+i+1
        assert stats['calls']==steps*len(states),'candidate hook coverage'
        result['gate_d_steps']=gate
        if any(x['max_request_relative_l2']>.025 for x in gate):result['status']='GATE_D_FAIL';return
        del refs;gc.collect();torch.cuda.synchronize();result['memory_after_gate']=memory()
        def reset():cache.crop(length)
        with torch.inference_mode(),execution(model,mode,chunk=64,validate_mask=False):
            for _ in range(2):
                reset()
                for ids_s in ids:
                    output=model(input_ids=ids_s,past_key_values=cache,use_cache=True,logits_to_keep=1);del output
            torch.cuda.synchronize();reset();torch.cuda.reset_peak_memory_stats();initial=memory();result['resident_timed_start']=payload(cache)
            times=[]
            for _ in range(10):
                reset();torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                wall=time.perf_counter();a.record()
                for ids_s in ids:
                    output=model(input_ids=ids_s,past_key_values=cache,use_cache=True,logits_to_keep=1);del output
                b.record();b.synchronize();elapsed=time.perf_counter()-wall;ms=a.elapsed_time(b)
                times.append(dict(cuda_ms=ms,wall_s=elapsed,aggregate_tokens_per_s=batch*steps/(ms/1000),wall_tokens_per_s=batch*steps/elapsed))
        result.update(status='COLLECTED',timing_repetitions=times,memory_timed=memory(),resident_final=payload(cache),resident_tokens_start=batch*length,resident_tokens_end=batch*(length+steps),
            timing_contract='Whole model, four decode steps, KV append/encode and attention included. Two warmups then ten timed repetitions. Cache reset outside interval. Model prefill and cache materialization excluded and reported separately.',
            interpretation='A feasible measured grid point, not a maximum-OOM capacity claim or independently sampled serving workload.')
        result['peak_workspace_and_growth_upper_bytes']=result['memory_timed']['peak_allocated_bytes']-initial['allocated_bytes']
    except torch.cuda.OutOfMemoryError:
        result['status']='OOM';result['reason']='own worker allocator or physical capacity exhausted; no accepted timing point'
    except Exception as exc:
        result['status']='FAIL';result['reason']=type(exc).__name__+': '+str(exc)
        (dest/'traceback.private.log').write_text(traceback.format_exc())
    finally:
        # Private diagnostic logs are outside sealed/public blocks.
        private=dest/'traceback.private.log'
        if private.exists():private.rename(folder/(key(spec)+'.private.log'))
        seal(dest,result);verify(dest)
        print(json.dumps(dict(point=key(spec),status=result['status'],spec=spec)),flush=True)
    if result['status'] in ['FAIL','OOM']:raise RuntimeError('bounded resident worker failed')

def run(manifest,stage):
    import fcntl
    data=json.loads(manifest.read_text());audit=manifest.parent/'asset_audit.json'
    assert digest(manifest.read_bytes())==manifest.with_suffix('.sha256').read_text().split()[0]
    assets=json.loads(audit.read_text());assert assets['status']=='VERIFIED_CACHED_BYTES' and assets['manifest_sha256']==digest(manifest.read_bytes())
    identity=dict(source_sha256=source_sha(),manifest_sha256=digest(manifest.read_bytes()),asset_audit_sha256=digest(audit.read_bytes()),stage=stage)
    folder=ROOT/'results/resident_v2'/digest(canonical(identity).encode())[:16];folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'campaign.json',dict(identity,gate_d=.025,planned=plan(data,stage),created=now(),publication='PENDING'))
    db=sqlite3.connect(folder/'registry.sqlite');db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,spec TEXT,status TEXT,updated TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS publications(label TEXT PRIMARY KEY,status TEXT,manifest_sha256 TEXT,updated TEXT)')
    for s in plan(data,stage):db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,?)',(key(s),canonical(s),'PLANNED',now()))
    db.commit();lock=(ROOT.parent/'engineering_gpu.lock').open('a')
    atomic(folder/'status.json',dict(status='WAITING_FOR_GPU_LOCK',updated=now()))
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        for s in plan(data,stage):
            k=key(s);status=db.execute('SELECT status FROM jobs WHERE id=?',(k,)).fetchone()[0]
            if status!='PLANNED':continue
            db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',('RUNNING',now(),k));db.commit()
            atomic(folder/'status.json',dict(status='RUNNING',spec=s,updated=now()))
            sf=folder/(k+'.spec.json');atomic(sf,s)
            with (folder/(k+'.worker.private.log')).open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','scope_v2.resident','worker','--manifest',str(manifest.resolve()),'--folder',str(folder.resolve()),'--spec',str(sf.resolve())],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                try:rc=proc.wait(timeout=2700)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    rc=124
            point=folder/'points'/k
            if (point/'manifest.json').exists():r=verify(point)
            else:
                point.mkdir(parents=True,exist_ok=True);r=dict(status='TIMEOUT' if rc==124 else 'FAIL',spec=s,reason='worker did not seal; no accepted timing point');seal(point,r)
            db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(r['status'],now(),k))
            db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',(k,'PENDING',digest((point/'manifest.json').read_bytes()),now()));db.commit()
            with zipfile.ZipFile(ROOT.parent/('SCOPE_RESIDENT_'+k+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
                for p in point.iterdir():z.write(p,p.relative_to(ROOT).as_posix())
            atomic(folder/'status.json',dict(status='RUNNING',last_spec=s,last_status=r['status'],counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status')),updated=now()))
            if rc:raise RuntimeError('STOP: sealed resident failure; bounded repair required')
        atomic(folder/'status.json',dict(status='COMPLETE',counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status')),updated=now()))
    except Exception:
        atomic(folder/'status.json',dict(status='STOPPED_WITH_FAILURE',counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status')),updated=now()));raise
    finally:db.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--stage',choices=['pilot','frontier'],default='pilot');p.add_argument('--folder',type=Path);p.add_argument('--spec',type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a.manifest,a.folder,a.spec)
    else:run(a.manifest,a.stage)
