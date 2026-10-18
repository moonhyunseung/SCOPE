"""Bounded engineering data collection; server storage only, no LLM calls."""
from __future__ import annotations
import argparse, collections, csv, fcntl, hashlib, importlib, inspect, json, math, os, subprocess, sys, threading, time, traceback
from pathlib import Path
from scope.runner import ROOT, atomic, canonical, digest, gpu_gate, now, source_hash
from scope.experiments import native_eager_reference, describe, versions

TOLERANCE = 0.025

def metric(x, ref):
    import torch
    x, ref = x.float(), ref.float()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(ref).all())
    result = {'finite':finite,'nonfinite_output':int((~torch.isfinite(x)).sum()),'nonfinite_reference':int((~torch.isfinite(ref)).sum())}
    if not finite:
        return dict(result, relative_l2=None, max_abs=None, reference_norm=None, cosine=None)
    d=x-ref; norm=ref.norm()
    den=x.norm()*norm
    return dict(result,relative_l2=float(d.norm()/norm.clamp_min(1e-12)),max_abs=float(d.abs().max()),reference_norm=float(norm),cosine=float((x*ref).sum()/den) if float(den)>0 else None)

def accepted(m):
    return m['finite'] and m['relative_l2'] is not None and m['relative_l2'] <= TOLERANCE

def logits_metric(x, ref):
    import torch
    m=metric(x,ref)
    if m['finite']:
        a,b=x.float(),ref.float(); la=a.log_softmax(-1); lb=b.log_softmax(-1)
        m.update(top1_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean()),kl_reference_to_output=float((lb.exp()*(lb-la)).sum(-1).mean()),reference_top1_nll_in_output=float(-la.gather(-1,b.argmax(-1,keepdim=True)).mean()))
    return m

def quantize(x,bits,group):
    import torch
    lim=2**(bits-1)-1
    z=x.float().reshape(*x.shape[:-1],-1,group)
    scales=z.abs().amax(-1).clamp_min(1e-12)/lim
    codes=(z/scales.unsqueeze(-1)).round().clamp(-lim,lim).to(torch.int16).reshape(x.shape)
    decoded=(codes.float().reshape_as(z)*scales.unsqueeze(-1)).reshape_as(x)
    if bits==4:
        u=(codes+8).to(torch.uint8); packed=u[...,::2]|(u[...,1::2]<<4)
    else: packed=(codes+128).to(torch.uint8)
    return packed,scales,decoded

def unpack(packed,scales,bits,group,dtype):
    import torch
    if bits==4: c=torch.stack([(packed&15).to(torch.int16)-8,(packed>>4).to(torch.int16)-8],-1).flatten(-2)
    else: c=packed.to(torch.int16)-128
    return (c.float().reshape(*c.shape[:-1],-1,group)*scales.unsqueeze(-1)).reshape(c.shape).to(dtype)

def sdpa(q,k,v,mask,scale,math_only=False):
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    rep=q.shape[1]//k.shape[1]; k=k.repeat_interleave(rep,1);v=v.repeat_interleave(rep,1)
    if math_only:
        with sdpa_kernel(SDPBackend.MATH): return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=scale,is_causal=False)
    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=scale,is_causal=False)

def write_row(path,row):
    with path.open('a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def immutable_identity():
    return {'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_hash':source_hash()}

def collect(spec,dest,config):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    inventory=gpu_gate(config)
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    torch.cuda.set_per_process_memory_fraction(config['max_allocated_hbm_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    m=spec['model']; mid=m['id'];rev=m['revision']; L=spec['length'];B=spec['batch'];steps=config['decode_steps']
    cfg=AutoConfig.from_pretrained(mid,revision=rev,trust_remote_code=False,local_files_only=True)
    if L+steps>getattr(cfg,'max_position_embeddings',10**9): return {'status':'UNSUPPORTED','reason':'context plus decode exceeds position limit'}
    tokenizer=AutoTokenizer.from_pretrained(mid,revision=rev,trust_remote_code=False,local_files_only=True)
    ids=tokenizer.encode(config['inputs'][spec['category']],add_special_tokens=False)
    tokens=[(ids*((L+31+len(ids)-1)//len(ids)))[b*7:b*7+L] for b in range(B)]
    inputs=torch.tensor(tokens,device='cuda'); dtype=getattr(torch,spec['dtype'])
    model=AutoModelForCausalLM.from_pretrained(mid,revision=rev,trust_remote_code=False,local_files_only=True,dtype=dtype,attn_implementation='eager').eval().to('cuda')
    module=importlib.import_module('transformers.models.'+cfg.model_type+'.modeling_'+cfg.model_type)
    original=module.eager_attention_forward
    layers=[(n,x) for n,x in model.named_modules() if type(x).__name__.endswith('Attention')]
    if not layers: raise RuntimeError('no attention layers')
    selected={id(layers[i][1]):layers[i][0] for i in sorted({0,len(layers)//2,len(layers)-1})}
    state={'mode':'native','step':-1,'sample_saved':False};counts=collections.Counter(); dense_gates=[]; quant_gates=[]
    atomic(dest/'provenance.json',dict(immutable_identity(),spec=spec,config=config,gpu_inventory=inventory,worker_pid=os.getpid(),software=versions(),gpu=torch.cuda.get_device_name(),dtype=str(dtype),token_sha256=digest(canonical(tokens).encode()),native_function_sha256=digest(inspect.getsource(original).encode()),tf32_allowed=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),teacher_forcing='next input chosen by full native recomputation; all paths consume same tokens',quality_scope='8-step decode-only quantization intervention after native prefill; dense caches retained; not perplexity or packed-resident capacity'))
    def observed(attention,q,k,v,mask,*args,**kwargs):
        result=original(attention,q,k,v,mask,*args,**kwargs)
        if q.shape[-2]!=1 or state['mode']=='native': return result
        bound=inspect.signature(original).bind_partial(attention,q,k,v,mask,*args,**kwargs);bound.apply_defaults()
        scale=bound.arguments.get('scaling')
        if scale is None:scale=getattr(attention,'scaling',q.shape[-1]**-0.5)
        if state['mode']=='quant':
            _,_,dk=quantize(k,4,32);_,_,dv=quantize(v,4,32)
            _,prob,y=native_eager_reference(q,dk.to(q.dtype),dv.to(q.dtype),mask,scale)
            return y.transpose(1,2).contiguous(),prob
        if id(attention) not in selected or state['step'] not in config['capture_steps']: return result
        native=result[0].transpose(1,2);_,prob,replayed=native_eager_reference(q,k,v,mask,scale)
        row={'kind':'state','step':state['step'],'layer':selected[id(attention)],'shape_q':list(q.shape),'shape_k':list(k.shape),'native_replay':metric(replayed,native),'native_exact':bool(torch.equal(replayed,native)),'Q':describe(q),'K':describe(k),'V':describe(v),'P':describe(prob),'Y':describe(native)}
        if not row['native_exact']:counts['native_replay_failures']+=1
        row['dense_sdpa']=metric(sdpa(q,k,v,mask,scale),native)
        row['dense_math_sdpa']=metric(sdpa(q,k,v,mask,scale,True),native)
        _,_,fp32=native_eager_reference(q.float(),k.float(),v.float(),mask.float() if mask is not None and mask.dtype!=torch.bool else mask,scale)
        row['fp32_diagnostic']=metric(fp32,native)
        write_row(dest/'states.jsonl',row);counts['states']+=1
        if not state['sample_saved']:
            torch.save({'q':q.detach().cpu(),'k':k.detach().cpu(),'v':v.detach().cpu(),'mask':mask.detach().cpu() if mask is not None else None,'scale':float(scale),'spec':spec,'layer':selected[id(attention)]},dest/'sample.pt');state['sample_saved']=True
        for bits in config['bits']:
            for group in config['groups']:
                if k.shape[-1]%group:continue
                pk,sk,dk=quantize(k,bits,group);pv,sv,dv=quantize(v,bits,group)
                rk=unpack(pk,sk,bits,group,q.dtype);rv=unpack(pv,sv,bits,group,q.dtype)
                _,_,quant=native_eager_reference(q,dk.to(q.dtype),dv.to(q.dtype),mask,scale)
                _,_,packed=native_eager_reference(q,rk,rv,mask,scale)
                _,_,only_k=native_eager_reference(q,dk.to(q.dtype),v,mask,scale)
                _,_,only_v=native_eager_reference(q,k,dv.to(q.dtype),mask,scale)
                fast=sdpa(q,rk,rv,mask,scale)
                z={'kind':'quantization','step':state['step'],'layer':selected[id(attention)],'bits':bits,'group':group,'pure_quantization':metric(quant,native),'packed_executor':metric(packed,quant),'packed_total':metric(packed,native),'sdpa_executor':metric(fast,quant),'sdpa_total':metric(fast,native),'K_only':metric(only_k,native),'V_only':metric(only_v,native),'K_reconstruction':metric(dk,k),'V_reconstruction':metric(dv,v),'pack_exact':bool(torch.equal(unpack(pk,sk,bits,group,torch.float32),dk) and torch.equal(unpack(pv,sv,bits,group,torch.float32),dv)),'packed_bytes':sum(x.untyped_storage().nbytes() for x in [pk,pv,sk,sv]),'dense_bytes':k.untyped_storage().nbytes()+v.untyped_storage().nbytes()}
                z['executor_pass']=accepted(z['packed_executor']);z['sdpa_executor_pass']=accepted(z['sdpa_executor'])
                counts['quant_rows']+=1;counts['packed_failures']+=not(z['pack_exact'] and z['executor_pass']);counts['sdpa_failures']+=not z['sdpa_executor_pass']
                write_row(dest/'quantization.jsonl',z)
        return result
    module.eager_attention_forward=observed
    try:
        with torch.inference_mode():
            initial=model(input_ids=inputs,use_cache=True);past=initial.past_key_values;token=initial.logits[:,-1].argmax(-1,keepdim=True)
            qprefill=model(input_ids=inputs,use_cache=True);qpast=qprefill.past_key_values
            del initial,qprefill
            for step in range(steps):
                state.update(mode='capture',step=step);before=int(past.get_seq_length())
                inputs=torch.cat([inputs,token],1)
                cached=model(input_ids=token,past_key_values=past,use_cache=True);past=cached.past_key_values
                state['mode']='native';full=model(input_ids=inputs,use_cache=False)
                d=logits_metric(cached.logits[:,-1],full.logits[:,-1]);d['append_pass']=int(past.get_seq_length())==before+1
                ok=accepted(d) and d['append_pass'];dense_gates.append(ok)
                write_row(dest/'trajectory.jsonl',dict(kind='dense_cached_vs_full',step=step,metrics=d,fidelity_pass=ok,threshold=TOLERANCE))
                state['mode']='quant';qo=model(input_ids=token,past_key_values=qpast,use_cache=True);qpast=qo.past_key_values
                a=logits_metric(qo.logits[:,-1],full.logits[:,-1]);b=logits_metric(qo.logits[:,-1],cached.logits[:,-1])
                quant_gates.append(accepted(a))
                write_row(dest/'trajectory.jsonl',dict(kind='decode_INT4_group32_intervention',step=step,vs_full_native=a,vs_dense_cached=b,dense_fidelity_at_step=ok,scope='all-layer decode intervention; native prefill; dense KV retained; no generation quality certification'))
                if not d['finite'] or not a['finite']:raise RuntimeError('nonfinite logits: exact metrics preserved; stop condition')
                token=full.logits[:,-1].argmax(-1,keepdim=True)
                del cached,full,qo
            state['mode']='native'
    finally:module.eager_attention_forward=original
    return {'status':'DATA_COLLECTED','dense_fidelity':'PASS' if all(dense_gates) and not counts['native_replay_failures'] else 'FAIL','dense_steps_passed':sum(dense_gates),'dense_steps_checked':len(dense_gates),'quant_intervention_steps_within_exploratory_threshold':sum(quant_gates),'counts':dict(counts),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'interpretation':'Data collected even on fidelity failure. No latency ranking, model quality certification, or resident-KV capacity claim.'}

def seal(dest,result):
    atomic(dest/'result.json',result)
    atomic(dest/'summary.json',dict(result=result,identity=immutable_identity(),sealed=now()))
    rows=[]
    def flatten(x,prefix=''):
        out={}
        for k,v in x.items():
            key=prefix+k
            if isinstance(v,dict):out.update(flatten(v,key+'.'))
            else:out[key]=json.dumps(v) if isinstance(v,list) else v
        return out
    for name in ['trajectory.jsonl','quantization.jsonl']:
        if (dest/name).exists():
            for line in (dest/name).read_text().splitlines():rows.append(dict(source=name,**flatten(json.loads(line))))
    with (dest/'metrics.csv').open('w',newline='') as f:
        fields=sorted(set().union(*(set(x) for x in rows))) if rows else ['status']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows or [{'status':result['status']}])
    (dest/'report.md').write_text('# Engineering data condition\n\n'+json.dumps(result,indent=2)+'\n\nExploratory data only. Dense fidelity failures remain failures. Quantization intervention retains dense KV and native prefill; no packed resident-capacity, perplexity or latency claim.\n')
    files={p.name:{'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size} for p in dest.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',{'files':files,'sealed':now(),'identity':immutable_identity()})
    (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))

def verify(dest):
    m=json.loads((dest/'manifest.json').read_text())
    if not all((dest/n).exists() and digest((dest/n).read_bytes())==v['sha256'] for n,v in m['files'].items()):raise RuntimeError('artifact hash mismatch')
    return json.loads((dest/'result.json').read_text())

def worker(spec,dest,config):
    dest.mkdir(parents=True,exist_ok=False)
    try:result=collect(spec,dest,config)
    except Exception as e:
        result={'status':'OOM' if 'out of memory' in str(e).lower() else ('BLOCKED' if 'BLOCKED' in str(e) else 'FAIL'),'error':str(e),'traceback':traceback.format_exc()};atomic(dest/'failure.json',result)
    seal(dest,result);print(json.dumps(result,allow_nan=False),flush=True)

def run(config_file,limit=None,smoke=False):
    config=json.loads(Path(config_file).read_text());identity=immutable_identity()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    if dirty:raise RuntimeError('tracked source/config must be committed before execution')
    run_id=digest(canonical(dict(config=config,identity=identity)).encode())[:16]+('-smoke' if smoke else '')
    folder=ROOT/'results'/'engineering'/run_id;folder.mkdir(parents=True,exist_ok=True)
    lock=(ROOT.parent/'engineering_gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(folder/'campaign.json',dict(run_id=run_id,identity=identity,config=config,created=now(),storage='server only'))
    jobs=config['conditions'][:limit] if limit else config['conditions'];registry=[];began=time.monotonic();stop=threading.Event();live={'condition':None,'child_pid':None}
    def heartbeat():
        while not stop.is_set():
            atomic(folder/'heartbeat.json',dict(pid=os.getpid(),updated=now(),**live));stop.wait(15)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        for index,spec in enumerate(jobs):
            if time.monotonic()-began>config['campaign_timeout_seconds']:break
            ident=digest(canonical(spec).encode())[:12];job=folder/ident;job.mkdir(exist_ok=True)
            complete=job/'completed.json'
            if complete.exists():
                record=json.loads(complete.read_text());verify(job/record['attempt']);registry.append(record);continue
            gpu_gate(config)
            attempt='attempt-'+str(time.time_ns());dest=job/attempt;live.update(condition=ident)
            cmd=[sys.executable,'-m','scope.engineering','worker','--config',str(Path(config_file).resolve()),'--spec',canonical(spec),'--dest',str(dest)]
            with (job/(attempt+'.log')).open('w') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);live['child_pid']=proc.pid
                try:rc=proc.wait(timeout=config['job_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    rc=124
            live['child_pid']=None
            if (dest/'manifest.json').exists(): result=verify(dest)
            else:
                dest.mkdir(exist_ok=True);result={'status':'TIMEOUT' if rc==124 else 'FAIL','exit_code':rc,'reason':'worker did not seal output; partial data preserved'};seal(dest,result)
            record=dict(id=ident,spec=spec,attempt=attempt,ended=now(),result=result);atomic(complete,record);registry.append(record)
            atomic(folder/'registry.json',registry)
            counts=collections.Counter(x['result']['status'] for x in registry)
            atomic(folder/'status.json',dict(status='RUNNING',completed=len(registry),planned=len(jobs),counts=dict(counts),updated=now()))
            print(json.dumps(dict(condition=ident,index=index+1,planned=len(jobs),result=result)),flush=True)
        counts=collections.Counter(x['result']['status'] for x in registry)
        atomic(folder/'status.json',dict(status='COMPLETE' if len(registry)==len(jobs) else 'BUDGET_EXHAUSTED',completed=len(registry),planned=len(jobs),counts=dict(counts),updated=now()))
    finally:
        stop.set();thread.join(timeout=2);live.update(condition=None,child_pid=None);atomic(folder/'heartbeat.json',dict(pid=os.getpid(),updated=now(),state='STOPPED',**live))
    print('RESULT_DIRECTORY',folder,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--config',required=True);p.add_argument('--spec');p.add_argument('--dest');p.add_argument('--limit',type=int);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.mode=='worker':worker(json.loads(args.spec),Path(args.dest),json.loads(Path(args.config).read_text()))
    else:run(args.config,args.limit,args.smoke)
