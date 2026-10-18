"""Qwen cached/full causal diagnostics. Native eager operations are observed, not replaced.

Run prepare once against the frozen population, then run in a durable terminal.
No network/model API calls occur in the experiment loop. Raw tensors stay private.
"""
from __future__ import annotations
import argparse, collections, contextlib, copy, csv, hashlib, inspect, json, os
import shutil, sqlite3, subprocess, threading, time, traceback
from pathlib import Path
from scope.runner import atomic, canonical, digest, now, gpu_gate
from scope.engineering import metric, logits_metric, accepted, verify as old_verify

ROOT = Path(__file__).resolve().parents[1]
BASE = '3f06d6f5e8f86959f6e1a1bc9fc67ccdd05c67a8'
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REV = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
INPUT_HASH = '6f1963d16f90931ec56fc6d710b5fef20584f9f40b39d45a38b94a110a3f0319'

def csv_write(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def code_hash():
    return digest(b''.join(p.name.encode()+b'\0'+p.read_bytes()
                          for p in sorted([ROOT/'scope_v2/__init__.py',ROOT/'scope_v2/diagnose.py'])))

def seal(dest, result):
    atomic(dest/'summary.json', result)
    rows = result.get('steps', [])
    if rows or not (dest/'metrics.csv').exists(): csv_write(dest/'metrics.csv', rows)
    (dest/'report.md').write_text('# Native cached/full diagnostic\n\n'+json.dumps(result, indent=2)+'\n\n'
        'Outcome-stratified mechanism sample, not a prevalence estimate. Dense fidelity uses the unchanged 0.025 gate. '
        'FP32 uses the same checkpoint revision, independently loaded at native FP32 precision. '
        'Same-weight projection controls cast the current dtype weights to FP32 separately. '
        'No latency, quality or capacity inference from tracing runs.\n')
    files={p.name:dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size)
           for p in sorted(dest.iterdir()) if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',dict(files=files,baseline=BASE,code_sha256=code_hash(),sealed=now()))
    (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in files.items()))
    verify(dest)

def verify(dest):
    m=json.loads((dest/'manifest.json').read_text())
    for n,v in m['files'].items():
        if digest((dest/n).read_bytes())!=v['sha256']: raise RuntimeError('HASH_MISMATCH '+n)
    r=json.loads((dest/'summary.json').read_text())
    if r['status']=='COLLECTED':
        if len(r['steps'])!=8: raise RuntimeError('step denominator')
        with (dest/'stages.csv').open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=r['stage_rows']: raise RuntimeError('stage denominator')
    return r

def prepare(population, input_manifest, out, per_stratum=2):
    if out.exists(): raise RuntimeError('immutable plan already exists')
    raw=input_manifest.read_bytes()
    if digest(raw)!=INPUT_HASH: raise RuntimeError('historical input hash mismatch')
    docs=json.loads(raw)['documents']
    if isinstance(docs,list): docs={d['document_id']:d for d in docs}
    records=json.loads((population/'registry.json').read_text())
    strata=collections.defaultdict(list)
    for r in records:
        s=r['spec']; d=docs[s['document_id']]
        if s['dtype']=='bfloat16' and d['dtype_audit']:
            strata[(s['category'],s['length'],r['result']['dense_fidelity'])].append(r)
    cases=[]
    # FAIL/PASS balanced within domain and length; hash order never selected by effect size.
    for label in ['FAIL','PASS']:
        for domain in ['prose','code']:
            for length in [512,2048]:
                pool=sorted(strata[(domain,length,label)],key=lambda r:digest(canonical(r['spec']).encode()))
                if len(pool)<per_stratum: raise RuntimeError('insufficient stratum '+str((domain,length,label,len(pool))))
                for r in pool[:per_stratum]:
                    dest=population/r['id']/r['attempt']; old_verify(dest)
                    tokens=json.loads((dest/'tokens.json').read_text())
                    trajectory=[json.loads(l) for l in (dest/'trajectory.jsonl').read_text().splitlines()]
                    native=[x for x in trajectory if x['kind']=='native_cached_vs_full']
                    if len(native)!=8:raise RuntimeError('historical trajectory denominator')
                    cases.append(dict(case_id=r['id'],document_id=r['spec']['document_id'],domain=domain,
                        length=length,historical_fidelity=label,tokens=tokens,
                        historical_relative_l2=[x['metrics']['relative_l2'] for x in native],
                        historical_manifest_sha256=digest((dest/'manifest.json').read_bytes())))
    # Smallest meaningful pilot first: one historical fail and one pass, otherwise frozen order.
    a=next(x for x in cases if x['historical_fidelity']=='FAIL' and x['length']==512)
    b=next(x for x in cases if x['historical_fidelity']=='PASS' and x['length']==512)
    cases=[a,b]+[x for x in cases if x not in [a,b]]
    c=dict(protocol='native-diagnosis-v2.1',baseline=BASE,model=MODEL,revision=REV,steps=8,
        trace_steps=[0,3,7],per_stratum=per_stratum,input_manifest_sha256=INPUT_HASH,
        stage_contract='last query row; native scores and softmax captured inside original eager; full K/V compared',
        tolerances=dict(dense_relative_l2=.025,noop='exact',structure='exact',first_nonzero='max_abs > 0'),
        classification='structural mismatch => unresolved; exact upstream plus projection-shape intervention => demonstrated shape sensitivity; history intervention is descriptive, not additive decomposition',
        independent_unit='source article/project family; outcome-stratified cases are repeated across lengths and dtype',
        max_allocated_hbm_gib=96,reserve_hbm_gib=12,minimum_free_hbm_gib=32,reserve_disk_gib=24,
        seed=20261018,cases=cases)
    atomic(out,c)
    print(json.dumps(dict(plan=str(out),conditions=len(cases)*2,source_units=len({x['document_id'] for x in cases}),
        strata={str(k):len(v) for k,v in strata.items()},plan_sha256=digest(out.read_bytes()))),flush=True)

def tensor_hash(x):
    import torch
    return digest(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

def projection_probe(module, x, rows):
    """Same exact input vector/weights/bias; change GEMM M only. Not a backend label."""
    import torch
    import torch.nn.functional as F
    v=x[..., -1:, :].reshape(1,-1).contiguous()
    large=v.expand(rows,-1).contiguous()
    one=F.linear(v,module.weight,module.bias)
    many=F.linear(large,module.weight,module.bias)[-1:]
    fp=F.linear(v.float(),module.weight.float(),None if module.bias is None else module.bias.float())
    one_round=fp.to(one.dtype)
    return dict(rows=rows,one_vs_many=metric(one,many),one_vs_same_weight_fp32=metric(one,fp),
                many_vs_same_weight_fp32=metric(many,fp),one_vs_fp32_rounded=metric(one,one_round),
                input_sha256=tensor_hash(v),weight_sha256=tensor_hash(module.weight),
                bias_sha256=tensor_hash(module.bias) if module.bias is not None else None,
                input=v.cpu(),one=one.cpu(),many=many.cpu(),fp32=fp.cpu())

class Trace:
    def __init__(self,model,probe_rows=None):
        self.model=model; self.values={}; self.handles=[]; self.probe_rows=probe_rows; self.probes={}
    def put(self,key,x,axis=-2):
        if x is None:return
        self.values[key]=x.detach().select(axis,x.shape[axis]-1).contiguous().cpu() if axis is not None else x.detach().contiguous().cpu()
    def __enter__(self):
        import torch
        import transformers.models.qwen2.modeling_qwen2 as m
        def hook(key,axis=-2):
            return lambda mod,args,out:self.put(key,out[0] if isinstance(out,tuple) else out,axis)
        self.handles.append(self.model.model.embed_tokens.register_forward_hook(hook('embedding')))
        def rotary(mod,args,out):
            self.put('position_ids',args[1],axis=-1)
            self.put('rope_cos',out[0]);self.put('rope_sin',out[1])
        self.handles.append(self.model.model.rotary_emb.register_forward_hook(rotary))
        for i,layer in enumerate(self.model.model.layers):
            prefix=f'L{i:02d}.'
            for name,mod in [('norm1',layer.input_layernorm),('q_proj',layer.self_attn.q_proj),
                ('k_proj',layer.self_attn.k_proj),('v_proj',layer.self_attn.v_proj),
                ('o_proj',layer.self_attn.o_proj),('norm2_residual',layer.post_attention_layernorm),
                ('mlp_gate',layer.mlp.gate_proj),('mlp_up',layer.mlp.up_proj),('mlp_down',layer.mlp.down_proj),('layer_out',layer)]:
                self.handles.append(mod.register_forward_hook(hook(prefix+name)))
            if i==0 and self.probe_rows:
                for name in ['q_proj','k_proj','v_proj']:
                    def pre(mod,args,name=name): self.probes[name]=projection_probe(mod,args[0],self.probe_rows)
                    self.handles.append(getattr(layer.self_attn,name).register_forward_pre_hook(pre))
        self.handles.append(self.model.model.norm.register_forward_hook(hook('final_norm')))
        self.handles.append(self.model.lm_head.register_forward_hook(hook('logits')))
        self.orig=m.eager_attention_forward; self.module=m
        def observed(att,q,k,v,mask,*args,**kwargs):
            p=f'L{att.layer_idx:02d}.'
            self.put(p+'rope_q',q);self.put(p+'rope_k_last',k);self.put(p+'cache_k',k,axis=None);self.put(p+'cache_v',v,axis=None)
            self.put(p+'mask',mask)
            self.values[p+'scaling']=torch.tensor(float(kwargs.get('scaling',att.scaling)))
            original_softmax=torch.nn.functional.softmax
            def softmax(z,*a,**kw):
                self.put(p+'scores',z)
                result=original_softmax(z,*a,**kw);self.put(p+'softmax_fp32',result)
                return result
            torch.nn.functional.softmax=softmax
            try: result=self.orig(att,q,k,v,mask,*args,**kwargs)
            finally:torch.nn.functional.softmax=original_softmax
            self.put(p+'probabilities',result[1]);self.put(p+'attention_out',result[0],axis=1)
            return result
        m.eager_attention_forward=observed
        return self
    def __exit__(self,*exc):
        self.module.eager_attention_forward=self.orig
        for h in self.handles:h.remove()

def compare_traces(a,b,case_id,dtype,step,contrast):
    import torch
    if set(a.values)!=set(b.values): raise RuntimeError('trace schema mismatch')
    rows=[]
    # Explicit semantic order. Module hook execution order interleaves branches.
    order=['embedding','position_ids','rope_cos','rope_sin']
    stages=['norm1','q_proj','k_proj','v_proj','rope_q','rope_k_last','cache_k','cache_v','mask','scaling',
            'scores','softmax_fp32','probabilities','attention_out','o_proj','norm2_residual','mlp_gate','mlp_up','mlp_down','layer_out']
    for i in range(len(a.model.model.layers)):order += [f'L{i:02d}.'+s for s in stages]
    order += ['final_norm','logits']
    for key in order:
        if key not in a.values:continue
        x,y=a.values[key],b.values[key]
        if x.shape!=y.shape:raise RuntimeError('shape mismatch '+key)
        rows.append(dict(case_id=case_id,dtype=dtype,step=step,contrast=contrast,stage=key,
            exact=bool(torch.equal(x,y)),elements=x.numel(),**metric(x,y)))
    return rows

def distribution(trace,case_id,dtype,step):
    """Native marginals and paired head statistics. W is explicitly reconstructed."""
    import torch
    from scope.engineering import quantize
    records=[]
    for layer in [0,len(trace.model.model.layers)//2,len(trace.model.model.layers)-1]:
        p=f'L{layer:02d}.'
        q,k,v,z,prob,y=[trace.values[p+s].float() for s in ['rope_q','cache_k','cache_v','scores','probabilities','attention_out']]
        for label,x in [('Q',q),('K',k),('V',v),('Z',z),('P',prob),('Y',y)]:
            flat=x.flatten();sample=flat[::max(1,flat.numel()//8192)]
            records.append(dict(case_id=case_id,dtype=dtype,step=step,layer=layer,kind='native_marginal',state=label,
                elements=flat.numel(),sampled=sample.numel(),mean=float(sample.mean()),sd=float(sample.std()),
                quantiles=[float(t) for t in torch.quantile(sample,torch.tensor([0.,.01,.5,.99,1.]))]))
        for head in range(prob.shape[1]):
            kh=head//(prob.shape[1]//k.shape[1]); pv=prob[0,head];vv=v[0,kh]
            _,sk,dk=quantize(k[:,kh:kh+1],4,32);_,sv,dv=quantize(v[:,kh:kh+1],4,32)
            vs=sv[0,0]; w=pv[:,None]*vs
            def corr(a,b):
                a=a.flatten();b=b.flatten();a=a-a.mean();b=b-b.mean();den=a.norm()*b.norm()
                return float((a*b).sum()/den) if float(den)>0 else None
            signed=pv[:,None]*vv;total=signed.sum(0).abs().sum();mass=signed.abs().sum()
            zz=z[0,head];top=torch.topk(zz,2).values
            # FP32 same inputs, same quantizer: diagnostic sensitivity, not native intermediate.
            kk=k[0,kh]; qh=q[0,head]
            base=torch.mv(kk,qh); quant=torch.mv(dk[0,0],qh)
            _,exp=torch.frexp(w)
            ck=(k[0,kh].reshape(k.shape[-2],-1,32)/sk[0,0].unsqueeze(-1)).round()
            cv=(vv.reshape(vv.shape[0],-1,32)/vs.unsqueeze(-1)).round()
            records.append(dict(case_id=case_id,dtype=dtype,step=step,layer=layer,head=head,
                kind='head_joint_native_plus_reconstructed',score_gap=float(top[0]-top[1]),p_max=float(pv.max()),
                entropy=float(-(pv*pv.clamp_min(1e-30).log()).sum()),p_vscale_corr=corr(pv,vs.mean(-1)),
                w_exponent_min=int(exp.min()),w_exponent_max=int(exp.max()),w_zero_fraction=float((w==0).float().mean()),
                k_code_saturation=float((ck.abs()==7).float().mean()),v_code_saturation=float((cv.abs()==7).float().mean()),
                cancellation_ratio=float(total/mass) if float(mass)>0 else None,
                reconstructed_score_quantization=metric(quant,base),w_definition='P * FP32 per-token group V scale; reconstructed'))
    return records

def collect(model,case,dtype,dest,c):
    import torch
    tokens=case['tokens'];prefix=torch.tensor([tokens['prefix_ids']],device='cuda');teacher=torch.tensor([tokens['continuation_ids']],device='cuda')
    assert len(tokens['prefix_ids'])==case['length'] and teacher.shape[1]==9
    atomic(dest/'case.json',case)
    stage_rows=[];steps=[];dist=[];controls=[];probes=[];first=[]
    with torch.inference_mode():
        pre=model(input_ids=prefix,use_cache=True,logits_to_keep=1);past=pre.past_key_values;del pre
        for s in range(8):
            before=int(past.get_seq_length()); snapshot=copy.deepcopy(past)
            one=teacher[:,s:s+1];whole=torch.cat([prefix,teacher[:,:s+1]],1)
            cached=model(input_ids=one,past_key_values=past,use_cache=True,logits_to_keep=1);past=cached.past_key_values
            cy=cached.logits[:,-1].clone();del cached
            full=model(input_ids=whole,use_cache=False,logits_to_keep=1);fy=full.logits[:,-1].clone();del full
            d=logits_metric(cy,fy)
            step=dict(case_id=case['case_id'],dtype=dtype,step=s,relative_l2=d['relative_l2'],max_abs=d['max_abs'],
                kl=d['kl_reference_to_output'],top1=d['top1_agreement'],dense_pass=accepted(d),
                native_nll=float(-cy.float().log_softmax(-1)[0,teacher[0,s+1]]),
                full_nll=float(-fy.float().log_softmax(-1)[0,teacher[0,s+1]]),append_ok=int(past.get_seq_length())==before+1)
            if s in c['trace_steps']:
                with Trace(model,probe_rows=whole.shape[1] if s==0 else None) as tc:
                    out=model(input_ids=one,past_key_values=copy.deepcopy(snapshot),use_cache=True,logits_to_keep=1)
                controls.append(bool(torch.equal(out.logits[:,-1],cy)));del out
                with Trace(model) as tf:out=model(input_ids=whole,use_cache=False,logits_to_keep=1)
                controls.append(bool(torch.equal(out.logits[:,-1],fy)));del out
                # Prefix recomputed at each step: history length/chunk shape intervention.
                fresh=model(input_ids=whole[:,:-1],use_cache=True,logits_to_keep=1).past_key_values
                with Trace(model) as th:out=model(input_ids=one,past_key_values=fresh,use_cache=True,logits_to_keep=1)
                fresh_logits=out.logits[:,-1].clone();del out,fresh
                with Trace(model) as te:
                    out=model(input_ids=one,past_key_values=copy.deepcopy(snapshot),position_ids=torch.tensor([[before]],device='cuda'),
                              attention_mask=torch.ones((1,before+1),device='cuda',dtype=torch.long),use_cache=True,logits_to_keep=1)
                explicit_exact=bool(torch.equal(out.logits[:,-1],cy));controls.append(explicit_exact);del out
                for contrast,a,b in [('cached_full',tc,tf),('fresh_full',th,tf),('history',tc,th),('explicit_default',te,tc)]:
                    rows=compare_traces(a,b,case['case_id'],dtype,s,contrast);stage_rows+=rows
                    nonzero=next((r['stage'] for r in rows if not r['exact']),None)
                    structural=all(r['exact'] for r in rows if r['stage'] in ['position_ids','rope_cos','rope_sin'] or r['stage'].endswith(('.mask','.scaling')))
                    first.append(dict(case_id=case['case_id'],dtype=dtype,step=s,contrast=contrast,first_nonzero=nonzero,
                        structural_exact=structural,first_over_025=next((r['stage'] for r in rows if not accepted(r)),None)))
                step.update(fresh_full_relative_l2=metric(fresh_logits,fy)['relative_l2'],
                    recurrent_fresh_relative_l2=metric(cy,fresh_logits)['relative_l2'],explicit_positions_mask_exact=explicit_exact)
                if s==0:
                    for name,p in tc.probes.items():
                        raw={k:v for k,v in p.items() if isinstance(v,torch.Tensor)}
                        torch.save(raw,dest/('projection_'+name+'.pt'))
                        probes.append(dict(name=name,**{k:v for k,v in p.items() if k not in raw}))
                dist+=distribution(tc,case['case_id'],dtype,s)
                del tc,tf,th,te
            steps.append(step);del snapshot,cy,fy
    csv_write(dest/'stages.csv',stage_rows);atomic(dest/'first_divergence.json',first)
    atomic(dest/'projection_controls.json',probes);atomic(dest/'attention_distributions.json',dist)
    structural=all(x['structural_exact'] for x in first if x['contrast']=='cached_full')
    earliest=sorted({x['first_nonzero'] for x in first if x['contrast']=='cached_full' and x['first_nonzero']})
    demonstrated=any(x['one_vs_many']['max_abs']>0 for x in probes)
    result=dict(status='COLLECTED',case_id=case['case_id'],document_id=case['document_id'],domain=case['domain'],length=case['length'],
        dtype=dtype,historical_fidelity=case['historical_fidelity'],dense_fidelity='PASS' if all(x['dense_pass'] for x in steps) else 'FAIL',
        controls_exact=all(controls),control_checks=len(controls),append_ok=all(x['append_ok'] for x in steps),
        structural_exact=structural,earliest_stages=earliest,projection_shape_sensitivity_demonstrated=demonstrated,
        classification='SHAPE_SENSITIVITY_DEMONSTRATED_NOT_EXHAUSTIVE_CAUSE' if structural and all(controls) and demonstrated else 'UNRESOLVED',
        unresolved_exhaustive_cause=True,stage_rows=len(stage_rows),steps=steps,model=MODEL,revision=REV,
        token_sha256=tokens['sha256'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        caution='Exact shape-control differences demonstrate a mechanism, not complete attribution of all 453 historical failures.')
    seal(dest,result);return result

def aggregate(folder,db,c,complete):
    results=[];missing=[];allfirst=[];allprobes=[]
    for ident,status,attempt in db.execute('SELECT id,status,attempt FROM jobs ORDER BY id'):
        if status in ['COLLECTED','FAIL']:
            dest=folder/ident/attempt;r=verify(dest);results.append(r)
            if (dest/'first_divergence.json').exists():allfirst+=json.loads((dest/'first_divergence.json').read_text())
            if (dest/'projection_controls.json').exists():
                allprobes += [dict(case_id=r['case_id'],dtype=r['dtype'],**x) for x in json.loads((dest/'projection_controls.json').read_text())]
        else:missing.append(dict(id=ident,status=status))
    out=folder/'analysis';out.mkdir(exist_ok=True)
    rows=[dict(x) for r in results for x in r.get('steps',[])]
    csv_write(out/'metrics.csv',rows);csv_write(out/'first_divergence.csv',allfirst);atomic(out/'projection_controls.json',allprobes)
    good=[r for r in results if r['status']=='COLLECTED']
    s=dict(status='COMPLETE' if complete and not missing else 'PARTIAL',planned=len(c['cases'])*2,collected=len(good),
        failed=len(results)-len(good),missing=missing,source_units=len({r['document_id'] for r in good}),
        counts=dict(collections.Counter((r.get('dtype','unknown')+'_'+r.get('dense_fidelity',r['status'])) for r in results)),
        exact_controls=sum(r['control_checks'] for r in good if r['controls_exact']),
        control_failures=sum(not r['controls_exact'] or not r['append_ok'] or not r['structural_exact'] for r in good),
        shape_sensitivity_conditions=sum(r['projection_shape_sensitivity_demonstrated'] for r in good),
        exhaustive_cause_unresolved=sum(r['unresolved_exhaustive_cause'] for r in good),
        baseline=BASE,plan_sha256=digest(canonical(c).encode()),code_sha256=code_hash(),results=results,
        gate_A='BOUNDED_DIAGNOSTIC_ONLY' if good and all(r['controls_exact'] and r['structural_exact'] and r['append_ok'] for r in good) else 'NOT_PASSED',
        gate_C='NOT_RUN',gate_D='NOT_RUN',gate_E='NOT_RUN',publication='PENDING',
        blocker='Noninteractive server push unavailable: GitHub authentication absent.',
        limitations='Outcome-stratified mechanism sample. No prevalence, model-quality, latency or resident-KV capacity estimate. Exhaustive cause attribution remains unresolved.')
    seal(out,s)
    # Seal compact analysis transaction before publication is eligible.
    manifest_hash=digest((out/'manifest.json').read_bytes())
    db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',('diagnosis-analysis','PENDING',manifest_hash,now()));db.commit()
    return s

def run(plan,limit=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import fcntl
    c=json.loads(plan.read_text());identity=digest(canonical(dict(config=c,code=code_hash())).encode())[:16]
    folder=ROOT/'results'/'continuation_v2'/identity;folder.mkdir(parents=True,exist_ok=True)
    lock=(ROOT.parent/'engineering_gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_gate(c)
    import torch,triton,transformers
    if not torch.cuda.is_available() or not torch.version.cuda:raise RuntimeError('BLOCKED CUDA runtime')
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(96*2**30/torch.cuda.get_device_properties(0).total_memory)
    import transformers.models.qwen2.modeling_qwen2 as native
    from scope.runner import source_hash
    if source_hash()!='c072fa01688f90f2d516725d60b01c3d11ac7e380472823a21fe0164db066947':raise RuntimeError('historical source drift')
    atomic(folder/'campaign.json',dict(config=c,code_sha256=code_hash(),historical_code_sha256=source_hash(),
        native_attention_sha256=digest(inspect.getsource(native.eager_attention_forward).encode()),
        native_rope_sha256=digest(inspect.getsource(native.apply_rotary_pos_emb).encode()),
        torch=torch.__version__,transformers=transformers.__version__,triton=triton.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),tf32=False,float32_matmul_precision=torch.get_float32_matmul_precision(),
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,created=now()))
    db=sqlite3.connect(folder/'registry.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,status TEXT,attempt TEXT,updated TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS publications(id TEXT PRIMARY KEY,status TEXT,manifest_sha TEXT,updated TEXT)')
    for dtype in ['bfloat16','float32']:
        for case in c['cases']:db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,?)',(case['case_id']+'-'+dtype,'PLANNED','',now()))
    db.commit();stop=threading.Event();live={'status':'RUNNING','completed':0}
    def heartbeat():
        while not stop.is_set():atomic(folder/'heartbeat.json',dict(live,updated=now()));stop.wait(15)
    t=threading.Thread(target=heartbeat,daemon=True);t.start()
    try:
        from transformers import AutoModelForCausalLM
        for dtype in ['bfloat16','float32']:
            jobs=c['cases'][:limit] if limit else c['cases']
            pending=[x for x in jobs if db.execute('SELECT status FROM jobs WHERE id=?',(x['case_id']+'-'+dtype,)).fetchone()[0] not in ['COLLECTED','FAIL']]
            if not pending:continue
            model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,local_files_only=True,trust_remote_code=False,
                dtype=getattr(torch,dtype),attn_implementation='eager').eval().to('cuda')
            for case in pending:
                ident=case['case_id']+'-'+dtype;attempt='attempt-'+str(time.time_ns());dest=folder/ident/attempt;dest.mkdir(parents=True)
                live.update(condition=ident);db.execute('UPDATE jobs SET status=?,attempt=?,updated=? WHERE id=?',('RUNNING',attempt,now(),ident));db.commit()
                try:
                    torch.cuda.reset_peak_memory_stats();r=collect(model,case,dtype,dest,c)
                except Exception as e:
                    r=dict(status='FAIL',case_id=case['case_id'],dtype=dtype,error=str(e));atomic(dest/'failure.json',dict(r,traceback=traceback.format_exc()));seal(dest,r)
                db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(r['status'],now(),ident));db.commit()
                live['completed']+=1;print(json.dumps(dict(condition=ident,result=r)),flush=True)
                aggregate(folder,db,c,False)
                if r['status']!='COLLECTED' or not r['controls_exact'] or not r['append_ok'] or not r['structural_exact']:
                    raise RuntimeError('STOP: diagnostic controls failed; sealed evidence retained')
            del model;torch.cuda.empty_cache()
        summary=aggregate(folder,db,c,limit is None);live.update(status=summary['status'],condition=None)
        atomic(folder/'status.json',summary);print('RESULT_DIRECTORY',folder,flush=True)
    finally:
        stop.set();t.join(timeout=2);atomic(folder/'heartbeat.json',dict(live,worker='STOPPED',updated=now()));db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--population',type=Path);p.add_argument('--input',type=Path)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--per-stratum',type=int,default=2);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.population,a.input,a.plan,a.per_stratum)
    else:run(a.plan,a.limit)
