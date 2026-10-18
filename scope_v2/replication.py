"""Deterministic held-out K/V factorial campaign; native backends remain authoritative."""
from __future__ import annotations
import argparse,collections,contextlib,copy,csv,fcntl,hashlib,importlib,inspect,json,os
import math,sqlite3,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
from scope.runner import atomic,canonical,digest,gpu_gate,now,source_hash
from scope.engineering import metric,logits_metric,accepted,quantize,unpack
from scope.population_collect import cache_equal,cache_hash,interaction,target_nll
from scope.population_stats import bootstrap
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['native','K0V0','K1V0','K0V1','K1V1','packed11']

def csv_write(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def fingerprint():
    return digest(b''.join(p.name.encode()+b'\0'+p.read_bytes() for p in [ROOT/'scope_v2/heldout.py',ROOT/'scope_v2/asset_audit.py',Path(__file__)]))

def jobs_for(data):
    jobs=[]
    for m in data['models']:
        for index,d in enumerate(data['documents']):
            base=dict(model=m['key'],document_id=d['document_id'],domain=d['domain'],split=d['split'],
                dtype='bfloat16',length=512 if d['split']=='screening' else data['confirmation_length'],bits=4,group=32,layer='all',head=None,role='core')
            jobs.append(base)
            if d['split']=='screening':
                jobs.append(dict(base,dtype='float32'))
                extra=[dict(bits=8),dict(group=64),dict(layer='first'),dict(layer='middle'),dict(layer='last'),
                       dict(head=0),dict(length=data['confirmation_length']),dict(group=128)][(index//2)%8]
                jobs.append(dict(base,**extra,role='paired_parameter_screen'))
    return jobs

def ident(spec):return digest(canonical(spec).encode())[:16]

@contextlib.contextmanager
def intervention(model,spec,variant):
    """Quantize inputs to the ORIGINAL eager function, preserving native dtype/scaling/mask semantics."""
    import torch
    module=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type)
    original=module.eager_attention_forward
    layers=[m for n,m in model.named_modules() if type(m).__name__.endswith('Attention')]
    if not layers:raise RuntimeError('unsupported attention module discovery')
    pick={'first':0,'middle':len(layers)//2,'last':len(layers)-1}
    target={id(layers[pick[spec['layer']]])} if spec['layer']!='all' else {id(m) for m in layers}
    stats=dict(calls=0,target_layers=len(target),native_function_sha256=digest(inspect.getsource(original).encode()))
    def observed(att,q,k,v,mask,*args,**kwargs):
        if q.shape[-2]!=1 or id(att) not in target:return original(att,q,k,v,mask,*args,**kwargs)
        stats['calls']+=1
        if variant in ['native','K0V0']:return original(att,q,k,v,mask,*args,**kwargs)
        def convert(x):
            if x.shape[-1]%spec['group']:raise ValueError('UNSUPPORTED_GROUP_HEAD_DIM')
            packed,scales,decoded=quantize(x,spec['bits'],spec['group'])
            return unpack(packed,scales,spec['bits'],spec['group'],x.dtype) if variant=='packed11' else decoded.to(x.dtype)
        def selected(x):
            if spec['head'] is None:return convert(x)
            head=spec['head']
            if not 0<=head<x.shape[1]:raise ValueError('UNSUPPORTED_KV_HEAD')
            z=x.clone();z[:,head:head+1]=convert(x[:,head:head+1]);return z
        if variant in ['K1V0','K1V1','packed11']:k=selected(k)
        if variant in ['K0V1','K1V1','packed11']:v=selected(v)
        return original(att,q,k,v,mask,*args,**kwargs)
    module.eager_attention_forward=observed
    try:yield stats
    finally:module.eager_attention_forward=original

def seal(dest,result):
    atomic(dest/'summary.json',result)
    (dest/'report.md').write_text('# Held-out native factorial condition\n\n'+json.dumps(result,indent=2)+'\n\nNative prefill and dense resident cache. This is numerical K/V intervention, not persistent-cache capacity or a model-wide quality benchmark.\n')
    if not (dest/'metrics.csv').exists():csv_write(dest/'metrics.csv',[])
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in dest.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',dict(files=files,sealed=now(),code_sha256=fingerprint()))
    (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))

def verify(dest):
    m=json.loads((dest/'manifest.json').read_text())
    for n,v in m['files'].items():
        if digest((dest/n).read_bytes())!=v['sha256']:raise RuntimeError('artifact hash mismatch')
    return json.loads((dest/'summary.json').read_text())

def collect(model,spec,document,dest):
    import torch
    length=spec['length'];window=document['windows'][spec['model']]['tokens']
    prefix=window[-(length+9):-9];teacher=window[-9:]
    if len(prefix)!=length:raise RuntimeError('input length')
    maxpos=getattr(model.config,'max_position_embeddings',10**9)
    if length+8>maxpos:return dict(status='UNSUPPORTED',reason='context plus decode exceeds position limit',spec=spec)
    dim=getattr(model.config,'head_dim',None) or model.config.hidden_size//model.config.num_attention_heads
    if dim%spec['group']:return dict(status='UNSUPPORTED',reason='group does not divide native head dimension',head_dim=dim,spec=spec)
    # kwargs are used only when explicitly supported by the model forward signature.
    logits_keep={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
    def call(ids,cache=None):return model(input_ids=ids,past_key_values=cache,use_cache=True,**logits_keep)
    inp=torch.tensor([prefix],device='cuda');t=torch.tensor([teacher],device='cuda')
    token_sha=digest(canonical(dict(prefix=prefix,teacher=teacher)).encode())
    snapshots=[];hashes=[];native=[];full=[];dense=[];rows=[];outputs={};controls=[]
    with torch.inference_mode():
        past=call(inp).past_key_values
        for s in range(8):
            snap=copy.deepcopy(past);snapshots.append(snap);hashes.append(cache_hash(snap))
            out=call(t[:,s:s+1],past);past=out.past_key_values;native.append(out.logits[:,-1].clone());del out
            whole=torch.cat([inp,t[:,:s+1]],1)
            out=model(input_ids=whole,use_cache=False,**logits_keep);full.append(out.logits[:,-1].clone());del out
            dense.append(logits_metric(native[s],full[s]))
            if not dense[-1]['finite']:raise RuntimeError('CONTROL_FAIL nonfinite native baseline')
        for regime in ['isolated','recurrent']:
            order=VARIANTS if int(spec['document_id'][:8],16)%2==0 else list(reversed(VARIANTS))
            for variant in order:
                recurrent=copy.deepcopy(snapshots[0]);outputs[(regime,variant)]=[]
                with intervention(model,spec,variant) as hook:
                    for s in range(8):
                        cache=copy.deepcopy(snapshots[s]) if regime=='isolated' else recurrent
                        if (regime=='isolated' or s==0) and not cache_equal(cache,snapshots[s]):raise RuntimeError('CONTROL_FAIL cache snapshot clone')
                        before=int(cache.get_seq_length());out=call(t[:,s:s+1],cache);recurrent=out.past_key_values
                        if int(recurrent.get_seq_length())!=before+1:raise RuntimeError('CONTROL_FAIL cache append')
                        y=out.logits[:,-1].clone();outputs[(regime,variant)].append(y)
                        exact=bool(torch.equal(y,native[s]))
                        if variant in ['native','K0V0']:controls.append(exact)
                        m=logits_metric(y,native[s]);nll=target_nll(y,teacher[s+1]);refnll=target_nll(native[s],teacher[s+1])
                        if not m['finite']:raise RuntimeError('CONTROL_FAIL nonfinite intervention')
                        if not all(math.isfinite(z) for z in [nll,refnll,m['kl_reference_to_output']]):raise RuntimeError('CONTROL_FAIL nonfinite endpoint')
                        rows.append(dict(regime=regime,variant=variant,step=s,relative_l2=m['relative_l2'],max_abs=m['max_abs'],
                            kl=m['kl_reference_to_output'],top1_agreement=m['top1_agreement'],nll=nll,nll_delta=nll-refnll,
                            dense_relative_l2=dense[s]['relative_l2'],dense_pass=accepted(dense[s]),exact_vs_native=exact))
                        del out,cache
                    if hook['calls']!=8*hook['target_layers']:raise RuntimeError('CONTROL_FAIL native hook coverage')
                del recurrent
        packing=[];history=[];factorial=[]
        for regime in ['isolated','recurrent']:
            for s in range(8):
                y00,y10,y01,y11=[outputs[(regime,n)][s] for n in ['K0V0','K1V0','K0V1','K1V1']]
                _,inter=interaction(y00,y10,y01,y11)
                if not inter['finite']:raise RuntimeError('CONTROL_FAIL nonfinite interaction')
                factorial.append(dict(regime=regime,step=s,interaction_relative_l2=inter['relative_l2']))
                packing.append(bool(torch.equal(y11,outputs[(regime,'packed11')][s])))
        for variant in VARIANTS:history.append(bool(torch.equal(outputs[('isolated',variant)][0],outputs[('recurrent',variant)][0])))
        unchanged=all(cache_hash(x)==h for x,h in zip(snapshots,hashes))
    csv_write(dest/'metrics.csv',rows);csv_write(dest/'factorial.csv',factorial)
    if len(rows)!=96 or len(factorial)!=16:raise RuntimeError('row denominator')
    result=dict(status='COLLECTED' if all(controls+packing+history) and unchanged else 'CONTROL_FAIL',spec=spec,
        native_function_sha256=hook['native_function_sha256'],
        token_sha256=token_sha,window_sha256=document['windows'][spec['model']]['sha256'],noop_checked=len(controls),noop_exact=sum(controls),
        packing_checked=len(packing),packing_exact=sum(packing),first_step_history_checked=6,first_step_history_exact=sum(history),
        snapshots_unchanged=unchanged,rows=len(rows),factorial_rows=16,dense_fidelity='PASS' if all(accepted(x) for x in dense) else 'FAIL',
        dense_steps_pass=sum(accepted(x) for x in dense),dense_steps_checked=8,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        logits_policy='last-token projection' if logits_keep else 'native complete projection then last token',
        statistical_unit='source article/project family; steps, variants and models are repeated measures',
        scope='Natural teacher tokens, original native eager function, dense resident cache; numerical sensitivity only')
    return result

def connect(folder):
    db=sqlite3.connect(folder/'registry.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs(id PRIMARY KEY,spec,status,attempt,updated)')
    db.execute('CREATE TABLE IF NOT EXISTS publications(id PRIMARY KEY,status,manifest_sha,updated)');db.commit();return db

def worker(manifest,folder,batch):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from transformers import AutoModelForCausalLM
    data=json.loads(manifest.read_text());jobs=json.loads(batch.read_text());model_spec=next(m for m in data['models'] if m['key']==jobs[0]['model'])
    docs={d['document_id']:d for d in data['documents']};config=dict(reserve_hbm_gib=12,minimum_free_hbm_gib=32,reserve_disk_gib=24)
    gpu_gate(config);torch.set_num_threads(4);torch.manual_seed(20261019);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(96*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=AutoModelForCausalLM.from_pretrained(model_spec['id'],revision=model_spec['revision'],local_files_only=True,trust_remote_code=False,
        dtype=getattr(torch,jobs[0]['dtype']),attn_implementation='eager').eval().to('cuda')
    import transformers,triton,platform
    atomic(folder/('runtime-'+model_spec['key']+'-'+jobs[0]['dtype']+'.json'),dict(model=model_spec,dtype=jobs[0]['dtype'],
        python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,triton=triton.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),deterministic_algorithms=True,tf32=False,
        float32_matmul_precision=torch.get_float32_matmul_precision(),bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    db=connect(folder)
    for spec in jobs:
        key=ident(spec);attempt='attempt-'+str(time.time_ns());dest=folder/key/attempt;dest.mkdir(parents=True)
        db.execute('UPDATE jobs SET status=?,attempt=?,updated=? WHERE id=?',('RUNNING',attempt,now(),key));db.commit()
        try:
            torch.cuda.reset_peak_memory_stats();r=collect(model,spec,docs[spec['document_id']],dest)
        except Exception as e:
            status='OOM' if 'out of memory' in str(e).lower() else 'CONTROL_FAIL' if 'CONTROL_FAIL' in str(e) else 'FAIL'
            r=dict(status=status,spec=spec,error=str(e));atomic(dest/'failure.json',dict(r,traceback=traceback.format_exc()))
        seal(dest,r);verify(dest)
        db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(r['status'],now(),key));db.commit();print(json.dumps(r),flush=True)
        if r['status'] in ['CONTROL_FAIL','OOM','FAIL']:raise RuntimeError('STOP: sealed worker failure')
    db.close()

def aggregate(folder,db,label,selected,data):
    import numpy as np
    out=folder/'blocks'/label
    if out.exists():return
    metrics=[];interactions=[];conditions=[];missing=[]
    for spec in selected:
        key=ident(spec);row=db.execute('SELECT status,attempt FROM jobs WHERE id=?',(key,)).fetchone()
        if row[0] in ['PLANNED','RUNNING']:missing.append(dict(spec,status=row[0]));continue
        dest=folder/key/row[1];r=verify(dest);conditions.append(r)
        if r['status']!='COLLECTED':continue
        with (dest/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
        with (dest/'factorial.csv').open() as f:factor=list(csv.DictReader(f))
        for regime in ['isolated','recurrent']:
            for variant in VARIANTS:
                a=[x for x in rows if x['regime']==regime and x['variant']==variant]
                if len(a)!=8:raise RuntimeError('incomplete treatment entered statistics')
                metrics.append(dict(spec,regime=regime,variant=variant,dense_fidelity=r['dense_fidelity'],
                    **{m:float(np.mean([float(x[m]) for x in a])) for m in ['relative_l2','kl','top1_agreement','nll_delta']}))
            a=[float(x['interaction_relative_l2']) for x in factor if x['regime']==regime]
            interactions.append(dict(spec,regime=regime,interaction_relative_l2=float(np.mean(a))))
    if missing:raise RuntimeError('cannot seal incomplete block')
    contrasts=[]
    for model in sorted({s['model'] for s in selected}):
        for domain in ['prose','code']:
            for regime in ['isolated','recurrent']:
                pool=[x for x in metrics if x['model']==model and x['domain']==domain and x['regime']==regime and x['role']=='core' and x['dtype']=='bfloat16']
                for endpoint in ['relative_l2','kl','top1_agreement','nll_delta']:
                    a={r['document_id']:r[endpoint] for r in pool if r['variant']=='K1V0'};b={r['document_id']:r[endpoint] for r in pool if r['variant']=='K0V1'}
                    ids=sorted(set(a)&set(b));values=[a[i]-b[i] for i in ids]
                    primary=label.startswith('confirmatory') and model in ['pythia1b','opt1_3b'] and endpoint=='relative_l2'
                    contrasts.append(dict(model=model,domain=domain,regime=regime,endpoint=endpoint,contrast='K-only minus V-only',
                        prespecified_primary=primary,**bootstrap(values,confidence=.99375 if primary else .95,repeats=20000 if primary else 5000,seed=20261019)))
    out.mkdir(parents=True);csv_write(out/'metrics.csv',metrics);csv_write(out/'factorial.csv',interactions);atomic(out/'contrasts.json',contrasts)
    summary=dict(status='VERIFIED_BLOCK',label=label,planned_conditions=len(selected),conditions=conditions,
        counts=dict(collections.Counter(r['status'] for r in conditions)),source_units=len({r['spec']['document_id'] for r in conditions if r['status']=='COLLECTED'}),
        dense_counts=dict(collections.Counter(r['dense_fidelity'] for r in conditions if r['status']=='COLLECTED')),
        missing=missing,document_metric_rows=len(metrics),factorial_document_rows=len(interactions),publication='PENDING',
        primary=data['primary'],uncertainty='Eight prespecified primary contrasts use 99.375% percentile source-block bootstrap intervals (20,000 resamples), nominal Bonferroni 95% family coverage. All other intervals exploratory. Approximate intervals, no power guarantee.',
        input_manifest_sha256=json.loads((folder/'campaign.json').read_text())['manifest_sha256'],code_sha256=fingerprint(),
        limitations='Dense resident intervention; no performance/capacity or whole-corpus quality evidence. Dense/full failures are retained separately. Models, tokens, heads and steps do not multiply independent source n.')
    seal(out,summary);verify(out);db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',(label,'PENDING',digest((out/'manifest.json').read_bytes()),now()));db.commit()
    archive=ROOT.parent/('SCOPE_REPLICATION_'+label+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print('SEALED_BLOCK',label,archive.name,flush=True)

def run(manifest,stage='pilot'):
    data=json.loads(manifest.read_text());sha=digest(manifest.read_bytes())
    if manifest.with_suffix('.sha256').read_text().split()[0]!=sha:raise RuntimeError('input manifest drift')
    if source_hash()!='c072fa01688f90f2d516725d60b01c3d11ac7e380472823a21fe0164db066947':raise RuntimeError('historical source drift')
    audit=manifest.parent/'asset_audit.json'
    if not audit.exists():raise RuntimeError('BLOCKED pinned model byte audit required')
    assets=json.loads(audit.read_text())
    if assets['status']!='VERIFIED_CACHED_BYTES' or assets['manifest_sha256']!=sha:raise RuntimeError('asset audit drift')
    identity=dict(manifest_sha256=sha,source_sha256=fingerprint(),historical_source_sha256=source_hash(),asset_audit_sha256=digest(audit.read_bytes()))
    folder=ROOT/'results/replication_v2'/digest(canonical(identity).encode())[:16];folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'campaign.json',dict(identity,created=now(),planned_jobs=len(jobs_for(data)),planned_source_units=data['source_units']))
    db=connect(folder);jobs=jobs_for(data);atomic(folder/'plan.json',jobs)
    for s in jobs:db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?)',(ident(s),canonical(s),'PLANNED','',now()))
    db.commit();lock=(ROOT.parent/'engineering_gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    live=dict(status='RUNNING',stage=stage);stop=threading.Event()
    def beat():
        while not stop.is_set():atomic(folder/'heartbeat.json',dict(live,updated=now()));stop.wait(15)
    t=threading.Thread(target=beat,daemon=True);t.start()
    try:
        phases=['pilot','screening','confirmatory'] if stage=='all' else [stage]
        pilot_ids={d['document_id'] for d in data['documents'][:2]}
        for phase in phases:
            for model in [m['key'] for m in data['models']]:
                selected=[s for s in jobs if s['model']==model and (s['split']=='confirmatory' if phase=='confirmatory' else
                    s['split']=='screening' and (phase!='pilot' or s['document_id'] in pilot_ids and s['role']=='core'))]
                for dtype in ['bfloat16','float32']:
                    pending=[s for s in selected if s['dtype']==dtype and db.execute('SELECT status FROM jobs WHERE id=?',(ident(s),)).fetchone()[0]=='PLANNED']
                    for i in range(0,len(pending),8):
                        batch=pending[i:i+8];batchfile=folder/('batch-'+digest(canonical(batch).encode())[:16]+'.json');atomic(batchfile,batch)
                        gpu_gate(dict(reserve_hbm_gib=12,minimum_free_hbm_gib=32,reserve_disk_gib=24));live.update(phase=phase,model=model,dtype=dtype)
                        logfile=folder/(batchfile.stem+'.private.log')
                        with logfile.open('w') as log:
                            proc=subprocess.Popen([sys.executable,'-u','-m','scope_v2.replication','worker','--manifest',str(manifest.resolve()),'--folder',str(folder),'--batch',str(batchfile)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                            try:rc=proc.wait(timeout=2700)
                            except subprocess.TimeoutExpired:
                                proc.terminate()
                                try:proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                                rc=124
                        for spec in batch:
                            key=ident(spec);status,attempt=db.execute('SELECT status,attempt FROM jobs WHERE id=?',(key,)).fetchone()
                            if status in ['PLANNED','RUNNING']:
                                attempt=attempt or 'attempt-'+str(time.time_ns());dest=folder/key/attempt;dest.mkdir(parents=True,exist_ok=True)
                                terminal='SKIPPED_AFTER_WORKER_FAILURE' if status=='PLANNED' else 'TIMEOUT' if rc==124 else 'FAIL'
                                seal(dest,dict(status=terminal,spec=spec,reason='worker did not seal completion; partial artifacts retained'))
                                db.execute('UPDATE jobs SET status=?,attempt=?,updated=? WHERE id=?',(terminal,attempt,now(),key));db.commit()
                        counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status').fetchall());live['counts']=counts
                        atomic(folder/'status.json',dict(live,updated=now()));print(json.dumps(live),flush=True)
                        if rc:
                            aggregate(folder,db,'failure-'+batchfile.stem,batch,data)
                            raise RuntimeError('STOP: bounded worker failed; inspect retained attempts')
                aggregate(folder,db,phase+'-'+model,selected,data)
        live['status']='COMPLETE_'+stage
    finally:
        if live['status']=='RUNNING':live['status']='STOPPED_WITH_FAILURE'
        stop.set();t.join(timeout=2);atomic(folder/'status.json',dict(live,updated=now()));atomic(folder/'heartbeat.json',dict(live,worker='STOPPED',updated=now()));db.close()
    print('RESULT_DIRECTORY',folder,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--stage',choices=['pilot','screening','confirmatory','all'],default='pilot');p.add_argument('--folder',type=Path);p.add_argument('--batch',type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a.manifest,a.folder,a.batch)
    else:run(a.manifest,a.stage)
