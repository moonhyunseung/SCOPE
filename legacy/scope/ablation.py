"""Paired controlled ablation on frozen teacher tokens and native cache snapshots."""
from __future__ import annotations
import argparse,contextlib,copy,csv,fcntl,hashlib,importlib,inspect,json,os,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
from scope.runner import ROOT,atomic,canonical,digest,gpu_gate,now,source_hash
from scope.engineering import metric,logits_metric,accepted,quantize,unpack,sdpa,write_row,seal,verify,immutable_identity
from scope.experiments import native_eager_reference,versions

VARIANTS=[
 dict(name='passthrough',engine='original',K=False,V=False),
 dict(name='replay00',engine='eager',K=False,V=False),
 dict(name='K10',engine='eager',K=True,V=False),
 dict(name='V01',engine='eager',K=False,V=True),
 dict(name='KV11',engine='eager',K=True,V=True),
 dict(name='packedKV',engine='packed',K=True,V=True),
 dict(name='KV_first',engine='eager',K=True,V=True,layer='first'),
 dict(name='KV_middle',engine='eager',K=True,V=True,layer='middle'),
 dict(name='KV_last',engine='eager',K=True,V=True,layer='last'),
 dict(name='KV8bit',engine='eager',K=True,V=True,bits=8),
 dict(name='KV_group64',engine='eager',K=True,V=True,group=64),
 dict(name='KV_group128',engine='eager',K=True,V=True,group=128),
 dict(name='dense_sdpa',engine='sdpa',K=False,V=False),
 dict(name='dense_math_sdpa',engine='math',K=False,V=False)]

@contextlib.contextmanager
def override(model,model_type,variant):
    module=importlib.import_module('transformers.models.'+model_type+'.modeling_'+model_type)
    original=module.eager_attention_forward
    layers=[(n,m) for n,m in model.named_modules() if type(m).__name__.endswith('Attention')]
    indices={'first':0,'middle':len(layers)//2,'last':len(layers)-1}
    target={id(layers[indices[variant['layer']]][1])} if variant.get('layer') else {id(m) for n,m in layers}
    stats={'calls':0,'target_layers':[n for n,m in layers if id(m) in target]}
    def observed(attention,q,k,v,mask,*args,**kwargs):
        result=original(attention,q,k,v,mask,*args,**kwargs)
        if q.shape[-2]!=1 or id(attention) not in target:return result
        stats['calls']+=1
        if variant['engine']=='original':return result
        bound=inspect.signature(original).bind_partial(attention,q,k,v,mask,*args,**kwargs);bound.apply_defaults()
        scale=bound.arguments.get('scaling')
        if scale is None:scale=getattr(attention,'scaling',q.shape[-1]**-0.5)
        bits=variant.get('bits',4);group=variant.get('group',32)
        def convert(x):
            packed,scales,decoded=quantize(x,bits,group)
            return unpack(packed,scales,bits,group,q.dtype) if variant['engine']=='packed' else decoded.to(q.dtype)
        k=convert(k) if variant['K'] else k;v=convert(v) if variant['V'] else v
        if variant['engine'] in ['sdpa','math']:
            y=sdpa(q,k,v,mask,scale,variant['engine']=='math');prob=result[1]
        else:_,prob,y=native_eager_reference(q,k,v,mask,scale)
        return y.transpose(1,2).contiguous(),prob
    module.eager_attention_forward=observed
    try:yield stats
    finally:module.eager_attention_forward=original

def tensors(cache):
    for layer in cache.layers:
        for name in ['keys','values']:
            x=getattr(layer,name,None)
            if x is not None:yield x

def cache_equal(a,b):
    import torch
    aa=list(tensors(a));bb=list(tensors(b))
    return len(aa)==len(bb) and all(x.shape==y.shape and torch.equal(x,y) for x,y in zip(aa,bb))

def cache_hash(cache):
    import torch
    h=hashlib.sha256()
    for x in tensors(cache):
        h.update(str((list(x.shape),str(x.dtype))).encode());h.update(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def interaction(y00,y10,y01,y11):
    import torch
    i=y11.float()-y10.float()-y01.float()+y00.float()
    m=metric(i,torch.zeros_like(i));norm=y00.float().norm()
    m['relative_l2']=float(i.norm()/norm.clamp_min(1e-12)) if m['finite'] and bool(torch.isfinite(norm)) else None
    m['reference_norm']=float(norm) if bool(torch.isfinite(norm)) else None
    return m['max_abs'],m

def target_nll(logits,target):
    return float(-logits.float().log_softmax(-1)[0,int(target)])

def collect(spec,dest,c):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    inventory=gpu_gate(c)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(c['max_allocated_hbm_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(c['model']['id'],revision=c['model']['revision'],local_files_only=True,trust_remote_code=False)
    ids=tokenizer.encode(c['inputs'][spec['category']],add_special_tokens=False)
    continuation=tokenizer.encode(c['continuations'][spec['category']],add_special_tokens=False)
    L=spec['length'];steps=c['decode_steps']
    prefix=(ids*((L+len(ids)-1)//len(ids)))[:L]
    teacher=(continuation*((steps+1+len(continuation)-1)//len(continuation)))[:steps+1]
    frozen=dict(prefix_ids=prefix,continuation_ids=teacher,teacher_policy='external authored token sequence; never chosen from any model output',category=spec['category'])
    token_hash=digest(canonical(frozen).encode());atomic(dest/'tokens.json',dict(frozen,sha256=token_hash))
    dtype=getattr(torch,spec['dtype']);model=AutoModelForCausalLM.from_pretrained(c['model']['id'],revision=c['model']['revision'],local_files_only=True,trust_remote_code=False,dtype=dtype,attn_implementation='eager').eval().to('cuda')
    inp=torch.tensor([prefix],device='cuda');tokens=torch.tensor([teacher],device='cuda')
    snapshots=[];snap_hashes=[];native=[];full=[];dense=[];outputs={};controls={};errors=[]
    atomic(dest/'provenance.json',dict(identity=immutable_identity(),spec=spec,config=c,token_sha256=token_hash,gpu_inventory=inventory,software=versions(),tf32=False,deterministic_algorithms=True,scope='Native prefill; dense KV retained. Numerical causal screening, not latency or resident-cache capacity.'))
    with torch.inference_mode():
        pre=model(input_ids=inp,use_cache=True);past=pre.past_key_values;del pre
        for s in range(steps):
            snapshot=copy.deepcopy(past);snapshots.append(snapshot);snap_hashes.append(cache_hash(snapshot))
            out=model(input_ids=tokens[:,s:s+1],past_key_values=past,use_cache=True);past=out.past_key_values
            native.append(out.logits[:,-1].detach().clone())
            complete=model(input_ids=torch.cat([inp,tokens[:,:s+1]],1),use_cache=False)
            full.append(complete.logits[:,-1].detach().clone());d=logits_metric(native[-1],full[-1]);dense.append(accepted(d))
            write_row(dest/'trajectory.jsonl',dict(kind='native_cached_vs_full',step=s,metrics=d,gate_pass=dense[-1],token_sha256=token_hash,cache_snapshot_sha256=snap_hashes[-1],native_nll=target_nll(native[-1],teacher[s+1]),full_nll=target_nll(full[-1],teacher[s+1])))
            del out,complete
        # A separate cache clone is used for every treatment and every isolated step.
        ordered=VARIANTS if spec['order']==0 else list(reversed(VARIANTS))
        for regime in ['isolated','recurrent']:
            for variant in ordered:
                key=(regime,variant['name']);outputs[key]=[]
                try:
                    recurrent=copy.deepcopy(snapshots[0])
                    if not cache_equal(recurrent,snapshots[0]):raise RuntimeError('cache clone differs before treatment')
                    with override(model,model.config.model_type,variant) as hook:
                        for s in range(steps):
                            cache=copy.deepcopy(snapshots[s]) if regime=='isolated' else recurrent
                            clone_ok=cache_equal(cache,snapshots[s]) if regime=='isolated' or s==0 else None
                            if clone_ok is False:raise RuntimeError('isolated cache does not match native snapshot')
                            out=model(input_ids=tokens[:,s:s+1],past_key_values=cache,use_cache=True)
                            recurrent=out.past_key_values
                            y=out.logits[:,-1].detach().clone();outputs[key].append(y)
                            cmp=logits_metric(y,native[s]);exact=bool(torch.equal(y,native[s]))
                            if variant['name'] in ['passthrough','replay00']:controls.setdefault(variant['name'],[]).append(exact)
                            nll=target_nll(y,teacher[s+1]);ref_nll=target_nll(native[s],teacher[s+1])
                            write_row(dest/'trajectory.jsonl',dict(kind='controlled_treatment',regime=regime,variant=variant['name'],parameters=variant,step=s,vs_native_cached=cmp,vs_full_native=logits_metric(y,full[s]),exact_vs_native_cached=exact,nll=nll,nll_delta_vs_native=nll-ref_nll,clone_matches_native=clone_ok,initial_cache_sha256=snap_hashes[s] if regime=='isolated' else snap_hashes[0],token_sha256=token_hash,hook_calls=hook['calls'],target_layers=hook['target_layers'],dense_fidelity_at_step=dense[s]))
                            del out,cache
                    del recurrent
                except Exception as e:
                    errors.append(dict(regime=regime,variant=variant['name'],error=str(e),traceback=traceback.format_exc()))
                    write_row(dest/'variant_failures.jsonl',errors[-1])
        control_ok=all(len(controls.get(n,[]))==2*steps and all(controls[n]) for n in ['passthrough','replay00'])
        contrast_rows=[]
        def contrast(row):contrast_rows.append(row);write_row(dest/'contrasts.jsonl',row)
        for regime in ['isolated','recurrent']:
            names=['replay00','K10','V01','KV11']
            for s in range(steps):
                if not all(len(outputs.get((regime,n),[]))>s for n in names):continue
                y00,y10,y01,y11=[outputs[(regime,n)][s] for n in names]
                _,i=interaction(y00,y10,y01,y11)
                contrast(dict(kind='KV_factorial',regime=regime,step=s,valid_noop_controls=control_ok,K_main=metric(y10,y00),V_main=metric(y01,y00),combined=metric(y11,y00),interaction=i,definition='interaction vector=y11-y10-y01+y00, normalized by norm(y00)',token_sha256=token_hash))
                if len(outputs.get((regime,'packedKV'),[]))>s:
                    contrast(dict(kind='packing',regime=regime,step=s,valid_noop_controls=control_ok,metrics=metric(outputs[(regime,'packedKV')][s],y11)))
        for variant in VARIANTS:
            a=outputs.get(('isolated',variant['name']),[]);b=outputs.get(('recurrent',variant['name']),[])
            for s in range(min(len(a),len(b))):
                contrast(dict(kind='history_effect',variant=variant['name'],step=s,valid_noop_controls=control_ok,metrics=metric(b[s],a[s]),first_step_expected_exact=s==0,first_step_exact=bool(torch.equal(a[s],b[s])) if s==0 else None))
        history_firststep_ok=all(bool(__import__('torch').equal(outputs[('isolated',v['name'])][0],outputs[('recurrent',v['name'])][0])) for v in VARIANTS if outputs.get(('isolated',v['name'])) and outputs.get(('recurrent',v['name'])))
        torch.save(dict(token_sha256=token_hash,spec=spec,native=[x.cpu() for x in native],full=[x.cpu() for x in full],outputs={'/'.join(k):[x.cpu() for x in v] for k,v in outputs.items()}),dest/'paired_logits.pt')
        unchanged=[cache_hash(x)==h for x,h in zip(snapshots,snap_hashes)]
        atomic(dest/'controls.json',dict(noop_exact=control_ok,controls={k:dict(exact=sum(v),checked=len(v)) for k,v in controls.items()},native_snapshots_unchanged=all(unchanged),history_firststep_exact=history_firststep_ok,dense_cached_full_steps_passed=sum(dense),dense_cached_full_steps_checked=steps,variant_errors=errors,token_sha256=token_hash))
    with (dest/'contrasts.csv').open('w',newline='') as f:
        fields=sorted(set().union(*(set(x) for x in contrast_rows)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in contrast_rows])
    if not control_ok or not all(unchanged) or not history_firststep_ok: status='CONTROL_FAIL'
    elif errors:status='PARTIAL_VARIANTS'
    else:status='DATA_COLLECTED'
    return dict(status=status,noop_controls_pass=control_ok,history_firststep_exact=history_firststep_ok,snapshots_unchanged=all(unchanged),dense_fidelity='PASS' if all(dense) else 'FAIL',planned_variants=len(VARIANTS)*2,completed_variants=sum(len(x)==steps for x in outputs.values()),trajectory_rows=steps+sum(len(x) for x in outputs.values()),contrasts=len(contrast_rows),variant_failures=len(errors),peak_allocated_bytes=torch.cuda.max_memory_allocated(),token_sha256=token_hash,spec=spec,interpretation='Paired numerical ablation on one model and fixed authored tokens. No-op controls required for causal interpretation. Cross-dtype outputs share token IDs; dense cached/full fidelity is reported separately. No latency, population-quality or resident-capacity claim.')

def worker(spec,dest,c):
    dest.mkdir(parents=True,exist_ok=False)
    try:r=collect(spec,dest,c)
    except Exception as e:r=dict(status='OOM' if 'out of memory' in str(e).lower() else 'FAIL',error=str(e),traceback=traceback.format_exc());atomic(dest/'failure.json',r)
    seal(dest,r);print(json.dumps(r),flush=True)

def summarize(folder,registry,final=False):
    conditions=[];effects=[];token_pairs={};integrity=[]
    for record in registry:
        d=folder/record['id']/record['attempt'];r=verify(d) if final else record['result'];conditions.append(dict(r,id=record['id'],spec=record['spec']))
        s=record['spec'];key=(s['category'],s['length']);token_pairs.setdefault(str(key),{})[s['dtype']]=r.get('token_sha256')
        if (d/'contrasts.jsonl').exists():
            for line in (d/'contrasts.jsonl').read_text().splitlines():effects.append(dict(condition=record['id'],spec=s,**json.loads(line)))
        if final:integrity.append(record['id'])
    if final:
        import torch
        torch.set_num_threads(4)
        pairs={}
        for record in registry:
            s=record['spec'];pairs.setdefault((s['category'],s['length']),{})[s['dtype']]=record
        dtype_rows=[]
        for key,pair in pairs.items():
            if set(pair)!=set(['bfloat16','float32']):continue
            paths={dtype:folder/r['id']/r['attempt']/'paired_logits.pt' for dtype,r in pair.items()}
            if not all(p.exists() for p in paths.values()):continue
            a=torch.load(paths['bfloat16'],map_location='cpu',weights_only=True);b=torch.load(paths['float32'],map_location='cpu',weights_only=True)
            if a['token_sha256']!=b['token_sha256']:raise RuntimeError('cross-dtype token mismatch')
            views_a=dict(a['outputs'],native=a['native'],full=a['full']);views_b=dict(b['outputs'],native=b['native'],full=b['full'])
            for name in sorted(set(views_a)&set(views_b)):
                for step,(x,y) in enumerate(zip(views_a[name],views_b[name])):
                    dtype_rows.append(dict(category=key[0],length=key[1],variant=name,step=step,bf16_vs_fp32=logits_metric(x,y),valid_noop_controls=all(r['result'].get('noop_controls_pass',False) for r in pair.values()),token_sha256=a['token_sha256']))
            del a,b,views_a,views_b
        (folder/'dtype_contrasts.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in dtype_rows))
    paired_inputs={k:len(set(v.values()))==1 and None not in v.values() and len(v)==2 for k,v in token_pairs.items()}
    atomic(folder/'SUMMARY.json',dict(updated=now(),final=final,planned=8,processed=len(registry),conditions=conditions,cross_dtype_input_identity=paired_inputs,artifact_integrity='PASS' if final else 'PENDING_FINAL_AUDIT',verified_conditions=integrity))
    with (folder/'effects.csv').open('w',newline='') as f:
        fields=sorted(set().union(*(set(x) for x in effects))) if effects else ['status'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in effects])
    (folder/'REPORT.md').write_text('# Controlled ablation\n\n'+json.dumps(dict(final=final,processed=len(registry),results=[dict(spec=x['spec'],status=x.get('status'),controls=x.get('noop_controls_pass'),contrasts=x.get('contrasts')) for x in conditions],cross_dtype_input_identity=paired_inputs),indent=2)+'\n\nInspect controls.json and vector contrasts before interpreting treatment effects. Neither token agreement nor authored continuation NLL establishes general model quality.\n')

def run(config_file,limit=None,smoke=False):
    c=json.loads(Path(config_file).read_text());identity=immutable_identity()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise RuntimeError('freeze tracked source first')
    run_id=digest(canonical(dict(config=c,identity=identity)).encode())[:16]+('-smoke' if smoke else '')
    folder=ROOT/'results/ablation'/run_id;folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'campaign.json',dict(run_id=run_id,config=c,identity=identity,created=now()))
    lock=(ROOT.parent/'engineering_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    jobs=c['conditions'][:limit] if limit else c['conditions'];registry=[];live={'condition':None,'child_pid':None};stop=threading.Event();began=time.monotonic()
    def beat():
        while not stop.is_set():atomic(folder/'heartbeat.json',dict(pid=os.getpid(),updated=now(),**live));stop.wait(15)
    t=threading.Thread(target=beat,daemon=True);t.start()
    try:
        for s in jobs:
            if time.monotonic()-began>c['campaign_timeout_seconds']:break
            ident=digest(canonical(s).encode())[:12];job=folder/ident;job.mkdir(exist_ok=True);done=job/'completed.json'
            if done.exists():r=json.loads(done.read_text());verify(job/r['attempt']);registry.append(r);continue
            gpu_gate(c);attempt='attempt-'+str(time.time_ns());dest=job/attempt;live['condition']=ident
            cmd=[sys.executable,'-m','scope.ablation','worker','--config',str(Path(config_file).resolve()),'--spec',canonical(s),'--dest',str(dest)]
            with (job/(attempt+'.log')).open('w') as log:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);live['child_pid']=p.pid
                try:rc=p.wait(timeout=c['job_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    rc=124
            live['child_pid']=None
            if (dest/'manifest.json').exists():result=verify(dest)
            else:dest.mkdir(exist_ok=True);result=dict(status='TIMEOUT' if rc==124 else 'FAIL',exit_code=rc);seal(dest,result)
            r=dict(id=ident,spec=s,attempt=attempt,result=result,ended=now());atomic(done,r);registry.append(r);atomic(folder/'registry.json',registry)
            summarize(folder,registry);atomic(folder/'status.json',dict(status='RUNNING',processed=len(registry),planned=len(jobs),updated=now()))
            print(json.dumps(r),flush=True)
        summarize(folder,registry,True)
        atomic(folder/'status.json',dict(status='COMPLETE' if len(registry)==len(jobs) else 'BUDGET_EXHAUSTED',processed=len(registry),planned=len(jobs),updated=now()))
        archive=ROOT.parent/('SCOPE_ABLATION_'+run_id+'.zip')
        bundle=ROOT.parent/('scope_ablation_'+run_id+'.bundle');subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in folder.rglob('*'):
                if f.is_file() and f.suffix not in ['.tmp','.pt']:z.write(f,f.relative_to(ROOT).as_posix())
            z.write(bundle,bundle.name)
            z.write(Path(config_file),'config.json')
        (archive.with_suffix('.zip.sha256')).write_text(digest(archive.read_bytes())+'  '+archive.name+'\n')
        print('ARCHIVE',archive,flush=True)
    finally:
        stop.set();t.join(timeout=2);atomic(folder/'heartbeat.json',dict(pid=os.getpid(),updated=now(),state='STOPPED'))
    print('RESULT_DIRECTORY',folder,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--config',required=True);p.add_argument('--spec');p.add_argument('--dest');p.add_argument('--limit',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='worker':worker(json.loads(a.spec),Path(a.dest),json.loads(Path(a.config).read_text()))
    else:run(a.config,a.limit,a.smoke)
