"""Bounded real native Q,K,V,Z,P,Y and reconstructed C,S,W sketches.

Observation calls installed native eager unchanged. Exact output/cache controls
precede sealing. All saved values are aggregates, never activation tensors.
"""
from __future__ import annotations
import argparse,contextlib,copy,importlib,inspect,json,os,sqlite3,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from scope.runner import atomic,canonical,digest,gpu_gate,now
ROOT=Path(__file__).resolve().parents[1]

def fingerprint():return digest(Path(__file__).read_bytes()+(ROOT/'docs/ATTENTION_DISTRIBUTION_PROTOCOL_V2.md').read_bytes())

def marginal(x,limit=8192):
    import torch
    v=x.detach().float().flatten().cpu();n=v.numel()
    indices=torch.linspace(0,n-1,min(limit,n)).long();v=v[indices]
    finite=torch.isfinite(v);clean=v[finite]
    if not len(clean):return dict(population_elements=n,sampled_elements=len(v),nonfinite=int((~finite).sum()))
    a=clean.abs();nz=a[a>0];mant,exp=torch.frexp(nz)
    hist={str(int(k)):int(v) for k,v in zip(*torch.unique(exp,return_counts=True))}
    edges=torch.linspace(.5,1.,9);mh=torch.histogram(mant,bins=edges)[0].long().tolist()
    return dict(population_elements=n,sampled_elements=len(v),nonfinite=int((~finite).sum()),zero_fraction=float((clean==0).float().mean()),
        negative_fraction=float((clean<0).float().mean()),mean=float(clean.mean()),rms=float(clean.square().mean().sqrt()),
        quantile_levels=[0,.01,.05,.5,.95,.99,1],quantiles=torch.quantile(clean,torch.tensor([0.,.01,.05,.5,.95,.99,1.])).tolist(),
        frexp_exponent_counts=hist,frexp_abs_mantissa_edges=edges.tolist(),frexp_abs_mantissa_counts=mh)

def joints(p,sv,cv):
    import torch
    # Token/group coordinates remain paired; do not independently sort marginals.
    pp=p.float()[:,None].expand_as(sv);w=pp*sv.float()
    exps=torch.frexp(sv.float())[1];expw=torch.frexp(w)[1];mask=w>0
    def span(e,valid):
        z=e[valid].float()
        if not len(z):return None
        q=torch.quantile(z,torch.tensor([.05,.95]));return float(q[1]-q[0])
    p_edges=torch.tensor([-float('inf'),-24.,-16.,-12.,-8.,-4.,-2.,0.,float('inf')])
    s_edges=torch.tensor([-float('inf'),-24.,-16.,-8.,-4.,0.,4.,8.,16.,24.,float('inf')])
    pi=torch.bucketize(pp.clamp_min(2**-126).log2().flatten(),p_edges[1:-1]);si=torch.bucketize(sv.float().clamp_min(2**-126).log2().flatten(),s_edges[1:-1])
    hist=torch.bincount(pi*(len(s_edges)-1)+si,minlength=(len(p_edges)-1)*(len(s_edges)-1)).reshape(len(p_edges)-1,-1)
    c=cv.float().reshape(*sv.shape,32);weighted=(w[...,None]*c).flatten(1)
    numerator=weighted.sum(0).abs().sum();denominator=weighted.abs().sum()
    return dict(w=marginal(w),s=marginal(sv),p_vs_s_log2_histogram=hist.tolist(),
        p_log2_bin_edges=['-inf',-24,-16,-12,-8,-4,-2,0,'inf'],s_log2_bin_edges=['-inf',-24,-16,-8,-4,0,4,8,16,24,'inf'],
        w_exponent_90_span=span(expw,mask),s_exponent_90_span=span(exps,sv>0),
        zero_code_fraction=float((cv==0).float().mean()),code_saturation_fraction=float((cv.abs()==7).float().mean()),
        reconstructed_p_sv_c_cancellation_ratio=float(numerator/denominator) if float(denominator)>0 else None,
        top32_probability_mass=float(p.topk(min(32,len(p))).values.sum()),
        joint_elements=int(sv.numel()),interpretation='P is native BF16 attention probability. C,S are fresh INT4 group32 V quantizer outputs. W=P*S and cancellation are reconstructed, not native intermediate tensors.')

@contextlib.contextmanager
def observe(model,document,step,records):
    import torch
    from scope.engineering import quantize,unpack
    module=importlib.import_module('transformers.models.'+model.config.model_type+'.modeling_'+model.config.model_type)
    original=module.eager_attention_forward;layers=[m for _,m in model.named_modules() if type(m).__name__.endswith('Attention')]
    lookup={id(m):i for i,m in enumerate(layers)};selected={0,len(layers)//2,len(layers)-1};stats=dict(calls=0,selected_calls=0)
    def wrapped(att,q,k,v,mask,*args,**kwargs):
        stats['calls']+=1;i=lookup[id(att)]
        if i not in selected:return original(att,q,k,v,mask,*args,**kwargs)
        stats['selected_calls']+=1;z_saved=[];sm=torch.nn.functional.softmax
        def softmax(z,*a,**kw):
            z_saved.append(z.detach().cpu());return sm(z,*a,**kw)
        torch.nn.functional.softmax=softmax
        try:result=original(att,q,k,v,mask,*args,**kwargs)
        finally:torch.nn.functional.softmax=sm
        assert len(z_saved)==1 and result[1] is not None,'native score/probability capture'
        pk,sk,dk=quantize(k,4,32);pv,sv,dv=quantize(v,4,32)
        assert torch.equal(unpack(pk,sk,4,32,k.dtype),dk.to(k.dtype))
        assert torch.equal(unpack(pv,sv,4,32,v.dtype),dv.to(v.dtype))
        codes=lambda x:torch.stack([(x&15).to(torch.int16)-8,(x>>4).to(torch.int16)-8],-1).flatten(-2)
        ck,cv=codes(pk).cpu(),codes(pv).cpu();sk=sk.cpu();sv=sv.cpu()
        qq=q.detach().cpu();kk=k.detach().cpu();vv=v.detach().cpu();p=result[1].detach().cpu();y=result[0].detach().cpu();z=z_saved[0]
        rep=q.shape[1]//k.shape[1];assert q.shape[-2]==1 and q.shape[0]==1
        for head in range(q.shape[1]):
            kv=head//rep
            native=dict(Q=qq[0,head,0],K=kk[0,kv],V=vv[0,kv],Z=z[0,head,0],P=p[0,head,0],Y=y[0,0,head])
            records.append(dict(document_id=document['document_id'],domain=document['domain'],step=step,layer=i,query_head=head,kv_head=kv,
                native={name:marginal(value) for name,value in native.items()},reconstructed=dict(Ck=marginal(ck[0,kv]),Cv=marginal(cv[0,kv]),Sk=marginal(sk[0,kv]),Sv=marginal(sv[0,kv])),
                joint=joints(p[0,head,0].float(),sv[0,kv].float(),cv[0,kv]),native_function_sha256=digest(inspect.getsource(original).encode())))
        return result
    module.eager_attention_forward=wrapped
    try:yield stats
    finally:module.eager_attention_forward=original

def seal(dest,result,records):
    from scope_v2.diagnose import csv_write
    dest.mkdir(parents=True,exist_ok=True)
    for part in range(0,len(records),64):
        (dest/f'sketch-{part//64:03d}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records[part:part+64]))
    csv_write(dest/'metrics.csv',[dict(document_id=r['document_id'],domain=r['domain'],layer=r['layer'],head=r['query_head'],step=r['step'],
        **{n:r['joint'][n] for n in ['w_exponent_90_span','s_exponent_90_span','top32_probability_mass','zero_code_fraction','code_saturation_fraction','reconstructed_p_sv_c_cancellation_ratio']}) for r in records])
    result.update(head_observations=len(records),source_units=len({r['document_id'] for r in records}),publication='PENDING')
    if records:
        spans=[r['joint']['w_exponent_90_span'] for r in records]
        result['prespecified_screens']=dict(narrow_w_fraction=sum(x is not None and x<=8 for x in spans)/len(records),
            concentration_fraction=sum(r['joint']['top32_probability_mass']>=.9 for r in records)/len(records),
            severe_cancellation_fraction=sum(r['joint']['reconstructed_p_sv_c_cancellation_ratio'] is not None and r['joint']['reconstructed_p_sv_c_cancellation_ratio']<.1 for r in records)/len(records),
            interpretation='Within one source/model/length cell; descriptive repeated head/layer/step observations, no source-level CI.')
        result['prespecified_screens']['narrow_w_pass']=result['prespecified_screens']['narrow_w_fraction']>=.9
        result['prespecified_screens']['concentration_pass']=result['prespecified_screens']['concentration_fraction']>=.8
    atomic(dest/'summary.json',result);(dest/'report.md').write_text('# Native attention distribution block\n\n'+json.dumps(result,indent=2)+'\n\nHead/layer/step observations are repeated measures, not independent sources. Native and reconstructed variables are explicitly labelled. Fixed bounded coordinate samples and selected screening sources limit inference. No speed or quality benefit follows from a distribution alone.\n')
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in dest.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',dict(files=files,code_sha256=fingerprint(),sealed=now()));(dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))

def worker(manifest,folder,specfile):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch,transformers,platform
    from transformers import AutoModelForCausalLM
    from scope.population_collect import cache_equal
    data=json.loads(manifest.read_text());spec=json.loads(specfile.read_text());m=next(m for m in data['models'] if m['key']==spec['model'])
    doc=next(d for d in data['documents'] if d['document_id']==spec['document_id']);assert doc['split']=='screening'
    records=[];controls=[];result=dict(status='RUNNING',spec=spec,model=m,code_sha256=fingerprint(),input_manifest_sha256=digest(manifest.read_bytes()),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,attention_backend='native eager',dtype='bfloat16',tf32=False,deterministic_algorithms=True),
        sampling='First/middle/last layers, all query heads, decode steps0/7; marginal deterministic evenly spaced flatten coordinates up to8192; P-S joints use all valid token/group coordinates; no random outcome selection.')
    dest=folder/'blocks'/digest(canonical(spec).encode())[:16]
    try:
        gpu_gate(dict(reserve_hbm_gib=12,minimum_free_hbm_gib=32,reserve_disk_gib=24))
        torch.set_num_threads(4);torch.manual_seed(20261019);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(96*2**30/torch.cuda.get_device_properties(0).total_memory)
        model=AutoModelForCausalLM.from_pretrained(m['id'],revision=m['revision'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').eval().cuda()
        keep={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
        L=spec['length'];tokens=doc['windows'][spec['model']]['tokens'];prefix=tokens[-(L+9):-9];teacher=tokens[-9:]
        result['token_window_sha256']=digest(canonical(dict(prefix=prefix,teacher=teacher)).encode());assert len(prefix)==L
        with torch.inference_mode():
            cache=model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=True,**keep).past_key_values
            for s in range(8):
                ids=torch.tensor([[teacher[s]]],device='cuda');snapshot=copy.deepcopy(cache) if s in [0,7] else None
                out=model(input_ids=ids,past_key_values=cache,use_cache=True,**keep);baseline=out.logits[:,-1].clone();cache=out.past_key_values;del out
                if snapshot is not None:
                    original_snapshot=copy.deepcopy(snapshot)
                    with observe(model,doc,s,records) as stats:observed=model(input_ids=ids,past_key_values=snapshot,use_cache=True,**keep)
                    controls.append(bool(torch.equal(observed.logits[:,-1],baseline) and cache_equal(observed.past_key_values,cache)))
                    assert stats['selected_calls']==3 and stats['calls']>0
                    assert original_snapshot.get_seq_length()==L+s
                    del observed,snapshot,original_snapshot
                del baseline
        assert len(controls)==2 and all(controls),'observer altered native logits/cache'
        result.update(status='COLLECTED',native_controls_exact=controls,original_source_count=1)
    except Exception as exc:
        result.update(status='FAIL',reason=type(exc).__name__+': '+str(exc))
        (folder/(dest.name+'.private.log')).write_text(traceback.format_exc())
    seal(dest,result,records)
    if result['status']!='COLLECTED':raise RuntimeError('STOP: sealed distribution worker failure')

def run(manifest):
    import fcntl
    data=json.loads(manifest.read_text());assert digest(manifest.read_bytes())==manifest.with_suffix('.sha256').read_text().split()[0]
    audit=manifest.parent/'asset_audit.json';assets=json.loads(audit.read_text())
    assert assets['status']=='VERIFIED_CACHED_BYTES' and assets['manifest_sha256']==digest(manifest.read_bytes())
    identity=dict(code_sha256=fingerprint(),manifest_sha256=digest(manifest.read_bytes()),asset_audit_sha256=digest(audit.read_bytes()))
    folder=ROOT/'results/distributions_v2'/digest(canonical(identity).encode())[:16];folder.mkdir(parents=True,exist_ok=True)
    jobs=[dict(model=m['key'],document_id=d['document_id'],domain=d['domain'],length=L) for m in data['models'] for d in data['documents'][:2] for L in [512,1536]]
    atomic(folder/'campaign.json',dict(identity,planned=jobs,preregistered_hypotheses='docs/ATTENTION_DISTRIBUTION_PROTOCOL_V2.md',source_units=2))
    db=sqlite3.connect(folder/'registry.sqlite');db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,spec TEXT,status TEXT,updated TEXT)');db.execute('CREATE TABLE IF NOT EXISTS publications(label TEXT PRIMARY KEY,status TEXT,manifest_sha256 TEXT,updated TEXT)')
    for s in jobs:db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,?)',(digest(canonical(s).encode())[:16],canonical(s),'PLANNED',now()))
    db.commit();lock=(ROOT.parent/'engineering_gpu.lock').open('a');atomic(folder/'status.json',dict(status='WAITING_FOR_GPU_LOCK',updated=now()));fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        for s in jobs:
            k=digest(canonical(s).encode())[:16]
            if db.execute('SELECT status FROM jobs WHERE id=?',(k,)).fetchone()[0]!='PLANNED':continue
            db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',('RUNNING',now(),k));db.commit();sf=folder/(k+'.spec.json');atomic(sf,s)
            atomic(folder/'status.json',dict(status='RUNNING',spec=s,updated=now()))
            with (folder/(k+'.worker.private.log')).open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','scope_v2.distributions','worker','--manifest',str(manifest.resolve()),'--folder',str(folder.resolve()),'--spec',str(sf.resolve())],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                try:rc=proc.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    rc=124
            dest=folder/'blocks'/k
            if not (dest/'manifest.json').exists():seal(dest,dict(status='TIMEOUT' if rc==124 else 'FAIL',spec=s,reason='worker stopped before sealing'),[])
            r=json.loads((dest/'summary.json').read_text());meta=json.loads((dest/'manifest.json').read_text())
            for n,v in meta['files'].items():assert digest((dest/n).read_bytes())==v['sha256']
            db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(r['status'],now(),k));db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',(k,'PENDING',digest((dest/'manifest.json').read_bytes()),now()));db.commit()
            with zipfile.ZipFile(ROOT.parent/('SCOPE_DISTRIBUTIONS_'+k+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
                for p in dest.iterdir():z.write(p,p.relative_to(ROOT).as_posix())
            if rc:raise RuntimeError('STOP: bounded distribution failure')
        atomic(folder/'status.json',dict(status='COMPLETE',counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status')),updated=now()))
    except Exception:
        atomic(folder/'status.json',dict(status='STOPPED_WITH_FAILURE',counts=dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status')),updated=now()));raise
    finally:db.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--folder',type=Path);p.add_argument('--spec',type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a.manifest,a.folder,a.spec)
    else:run(a.manifest)
