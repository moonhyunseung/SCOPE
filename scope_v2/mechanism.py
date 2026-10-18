"""Mechanism follow-up: matched native inputs, GEMM shape and reduction controls.

This is exploratory causal diagnosis selected after the published pilot. It does not
change the original dense fidelity threshold or claim complete population attribution.
"""
import argparse,copy,fcntl,json,os,sqlite3,threading,time,traceback,zipfile
from pathlib import Path
from scope.runner import atomic,canonical,digest,gpu_gate,now
from scope.engineering import metric
from scope_v2.diagnose import ROOT,MODEL,REV,Trace,projection_probe,tensor_hash,csv_write

NAMES=['q_proj','k_proj','v_proj','o_proj','mlp_gate','mlp_up','mlp_down']

def modules(model):
    l=model.model.layers[0]
    return dict(q_proj=l.self_attn.q_proj,k_proj=l.self_attn.k_proj,v_proj=l.self_attn.v_proj,
        o_proj=l.self_attn.o_proj,mlp_gate=l.mlp.gate_proj,mlp_up=l.mlp.up_proj,mlp_down=l.mlp.down_proj)

class InputTrace(Trace):
    def __enter__(self):
        super().__enter__();self.inputs={}
        for name,module in modules(self.model).items():
            def pre(mod,args,name=name):self.inputs[name]=args[0][:,-1:,:].detach().clone()
            self.handles.append(module.register_forward_pre_hook(pre))
        return self

def matmul_probe(a,b,rows):
    """M=1 vs M=N for the exact same last query and fixed right operand."""
    import torch
    one=torch.matmul(a,b);many=torch.matmul(a.expand(*a.shape[:-2],rows,a.shape[-1]).contiguous(),b)[...,-1:,:]
    fp=torch.matmul(a.float(),b.float())
    return dict(rows=rows,one_vs_many=metric(one,many),one_vs_same_weight_fp32=metric(one,fp),
        many_vs_same_weight_fp32=metric(many,fp),left_sha256=tensor_hash(a),right_sha256=tensor_hash(b)),one

def collect(model,case,dtype,dest):
    import torch
    inp=torch.tensor([case['tokens']['prefix_ids']],device='cuda');t=torch.tensor([case['tokens']['continuation_ids']],device='cuda')
    rows=[];details=[];old_flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with torch.inference_mode():
        past=model(inp,use_cache=True,logits_to_keep=1).past_key_values
        for s in range(8):
            snapshot=copy.deepcopy(past);one=t[:,s:s+1];whole=torch.cat([inp,t[:,:s+1]],1)
            plain=model(one,past_key_values=past,use_cache=True,logits_to_keep=1);past=plain.past_key_values
            if s not in [0,3,7]:continue
            with InputTrace(model) as tc:
                cached=model(one,past_key_values=copy.deepcopy(snapshot),use_cache=True,logits_to_keep=1)
            if not torch.equal(cached.logits,plain.logits):raise RuntimeError('instrumentation changes logits')
            with InputTrace(model) as tf:full=model(whole,use_cache=False,logits_to_keep=1)
            for name,module in modules(model).items():
                x=tc.inputs[name];xf=tf.inputs[name]
                input_metric=metric(x,xf)
                for flag in [True,False]:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
                    p=projection_probe(module,x,whole.shape[1]);native=tc.values['L00.'+name]
                    r=dict(case_id=case['case_id'],document_id=case['document_id'],domain=case['domain'],length=case['length'],
                        dtype=dtype,step=s,operation=name,reduced_precision_reduction=flag,
                        upstream_input_exact=bool(torch.equal(x,xf)),upstream_relative_l2=input_metric['relative_l2'],
                        native_cached_full=metric(native,tf.values['L00.'+name]),
                        one_vs_native=metric(p['one'].reshape_as(native),native),
                        **{k:v for k,v in p.items() if not isinstance(v,torch.Tensor)})
                    details.append(r)
                    if flag==old_flag and r['one_vs_native']['max_abs']!=0:raise RuntimeError('linear proxy does not replay native '+name)
                    rows.append(dict(case_id=r['case_id'],dtype=dtype,step=s,operation=name,reduced_precision_reduction=flag,
                        upstream_exact=r['upstream_input_exact'],upstream_relative_l2=r['upstream_relative_l2'],
                        shape_relative_l2=p['one_vs_many']['relative_l2'],shape_max_abs=p['one_vs_many']['max_abs'],
                        one_vs_fp32=p['one_vs_same_weight_fp32']['relative_l2'],
                        many_vs_fp32=p['many_vs_same_weight_fp32']['relative_l2'],
                        one_vs_native=r['one_vs_native']['relative_l2'],native_cached_full=r['native_cached_full']['relative_l2']))
                    if s==0 and flag==old_flag:
                        torch.save({k:v for k,v in p.items() if isinstance(v,torch.Tensor)},dest/(name+'_input.pt'))
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_flag
            # Native Q/K/V and P are captured, so both attention matmuls get same-input controls.
            q=tc.values['L00.rope_q'].unsqueeze(-2).cuda().to(model.dtype)
            k=tc.values['L00.cache_k'].cuda().to(model.dtype);v=tc.values['L00.cache_v'].cuda().to(model.dtype)
            rep=q.shape[1]//k.shape[1];kh=k.repeat_interleave(rep,1);vh=v.repeat_interleave(rep,1)
            prob=tc.values['L00.probabilities'].unsqueeze(-2).cuda().to(model.dtype)
            for name,a,b in [('QK',q,kh.transpose(-2,-1)),('PV',prob,vh)]:
                for flag in [True,False]:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
                    p,proxy=matmul_probe(a,b,whole.shape[1])
                    if name=='QK':
                        actual=tc.values['L00.scores'];proxy=(proxy*model.model.layers[0].self_attn.scaling).squeeze(-2).cpu()
                    else:actual=tc.values['L00.attention_out'];proxy=proxy.squeeze(-2).cpu()
                    native_replay=metric(proxy,actual)
                    details.append(dict(case_id=case['case_id'],document_id=case['document_id'],
                        dtype=dtype,step=s,operation=name,reduced_precision_reduction=flag,native_replay=native_replay,**p))
                    if flag==old_flag and native_replay['max_abs']!=0:raise RuntimeError('matmul proxy does not replay native '+name)
                    rows.append(dict(case_id=case['case_id'],dtype=dtype,step=s,operation=name,reduced_precision_reduction=flag,
                        shape_relative_l2=p['one_vs_many']['relative_l2'],shape_max_abs=p['one_vs_many']['max_abs'],
                        one_vs_fp32=p['one_vs_same_weight_fp32']['relative_l2'],many_vs_fp32=p['many_vs_same_weight_fp32']['relative_l2']))
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_flag
            del tc,tf,cached,full,snapshot,plain
    if len(rows)!=54:raise RuntimeError('mechanism row denominator')
    atomic(dest/'details.json',details);csv_write(dest/'metrics.csv',rows)
    return dict(status='COLLECTED',case_id=case['case_id'],document_id=case['document_id'],dtype=dtype,rows=54,
        token_sha256=case['tokens']['sha256'],exact_trace=True)

def seal(dest,summary):
    atomic(dest/'summary.json',summary)
    (dest/'report.md').write_text('# Native operation mechanism controls\n\n'+json.dumps(summary,indent=2)+'\n\nExploratory mechanism data; no latency, capacity, quality or complete population-cause claim. Native replay must be exact in the original reduction mode.\n')
    files={p.name:dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size) for p in dest.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',dict(files=files,sealed=now()))
    (dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in sorted(files.items())))

def verify(dest):
    m=json.loads((dest/'manifest.json').read_text())
    for n,v in m['files'].items():
        if digest((dest/n).read_bytes())!=v['sha256']:raise RuntimeError('HASH_MISMATCH '+n)
    return json.loads((dest/'summary.json').read_text())

def export(folder,db,label,config):
    import csv
    rows=[];details=[];results=[]
    for ident,status,attempt in db.execute('SELECT id,status,attempt FROM jobs ORDER BY id'):
        if status not in ['COLLECTED','FAIL']:continue
        dest=folder/ident/attempt;r=verify(dest);results.append(r)
        if status=='COLLECTED':
            with (dest/'metrics.csv').open() as f:rows.extend(csv.DictReader(f))
            details+=json.loads((dest/'details.json').read_text())
    out=folder/'blocks'/label
    if out.exists():raise RuntimeError('immutable mechanism block exists')
    out.mkdir(parents=True);atomic(out/'details.json',details);csv_write(out/'metrics.csv',rows);atomic(out/'config.json',config)
    good=[r for r in results if r['status']=='COLLECTED']
    summary=dict(status='VERIFIED_BLOCK',block=label,conditions=len(results),collected=len(good),failed=len(results)-len(good),
        source_units=len({r['document_id'] for r in good}),rows=len(rows),expected_rows=len(good)*54,
        results=results,publication='PENDING',inference='Exploratory repeated-measure causal controls, not independent model replication.')
    if len(rows)!=summary['expected_rows']:raise RuntimeError('block row denominator')
    seal(out,summary);verify(out)
    db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',(label,'PENDING',digest((out/'manifest.json').read_bytes()),now()));db.commit()
    archive=ROOT.parent/('SCOPE_MECHANISM_'+label+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    print('BLOCK_ARCHIVE',archive.name,digest(archive.read_bytes()),flush=True)

def run(plan,limit=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    c=json.loads(plan.read_text());config=dict(plan_sha256=digest(plan.read_bytes()),steps=[0,3,7],operations=NAMES+['QK','PV'],
        flags=[True,False],source=digest(Path(__file__).read_bytes()),classification='exploratory mechanism intervention, not prevalence or complete attribution')
    key=digest(canonical(config).encode())[:16];folder=ROOT/'results/mechanism_v2'/key;folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'config.json',config);stop=threading.Event();live=dict(status='WAITING_FOR_OWN_GPU_LOCK',completed=0)
    def beat():
        while not stop.is_set():atomic(folder/'heartbeat.json',dict(live,updated=now()));stop.wait(15)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    db=sqlite3.connect(folder/'registry.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs(id PRIMARY KEY,status,attempt,updated)')
    db.execute('CREATE TABLE IF NOT EXISTS publications(id PRIMARY KEY,status,manifest_sha,updated)')
    for dtype in ['bfloat16','float32']:
        for case in c['cases']:db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,?)',(case['case_id']+'-'+dtype,'PLANNED','',now()))
    db.commit()
    try:
        lock=(ROOT.parent/'engineering_gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
        gpu_gate(c);live['status']='RUNNING'
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.set_per_process_memory_fraction(96*2**30/torch.cuda.get_device_properties(0).total_memory)
        for dtype in ['bfloat16','float32']:
            jobs=(c['cases'][:limit] if limit else c['cases'])
            pending=[x for x in jobs if db.execute('SELECT status FROM jobs WHERE id=?',(x['case_id']+'-'+dtype,)).fetchone()[0] not in ['COLLECTED','FAIL']]
            if not pending:continue
            model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,local_files_only=True,trust_remote_code=False,
                dtype=getattr(torch,dtype),attn_implementation='eager').eval().to('cuda')
            for case in pending:
                ident=case['case_id']+'-'+dtype;attempt='attempt-'+str(time.time_ns());dest=folder/ident/attempt;dest.mkdir(parents=True)
                live['condition']=ident;db.execute('UPDATE jobs SET status=?,attempt=?,updated=? WHERE id=?',('RUNNING',attempt,now(),ident));db.commit()
                try:r=collect(model,case,dtype,dest)
                except Exception as e:r=dict(status='FAIL',case_id=case['case_id'],dtype=dtype,error=str(e));atomic(dest/'failure.json',dict(r,traceback=traceback.format_exc()))
                seal(dest,r);verify(dest)
                db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(r['status'],now(),ident));db.commit();live['completed']+=1
                print(json.dumps(r),flush=True)
                if r['status']!='COLLECTED':raise RuntimeError('STOP mechanism failure retained')
            del model;torch.cuda.empty_cache()
        export(folder,db,'pilot' if limit else 'stratified32',config)
        live['status']='COMPLETE' if not limit else 'PILOT_COMPLETE';print('RESULT_DIRECTORY',folder,flush=True)
    finally:
        stop.set();thread.join(timeout=2);atomic(folder/'heartbeat.json',dict(live,worker='STOPPED',updated=now()));db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();run(a.plan,a.limit)
