"""Durable document-level collection. Local server only, no LLM/API calls."""
import argparse,collections,gc,json,os,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
from scope.runner import ROOT,atomic,canonical,digest,gpu_gate,now
from scope.engineering import seal,verify,immutable_identity
MANIFEST=ROOT/'data/population_v1/manifest.json'
def inputs():
    payload=json.loads(MANIFEST.read_text());expected=(MANIFEST.parent/'manifest.sha256').read_text().split()[0]
    if digest(MANIFEST.read_bytes())!=expected:raise RuntimeError('INPUT_DRIFT')
    return payload,expected
def config():
    c=json.loads((ROOT/'configs/controlled_ablation_v1.json').read_text())
    for k in ['inputs','continuations','conditions','models']:c.pop(k,None)
    c.update(protocol_version='scope-independent-population-v1',campaign_timeout_seconds=28800,job_timeout_seconds=2400,sampling='frozen independent source units; 256 per domain; one window each',publication='server only',teacher_forcing='natural frozen source continuation; matched across all conditions',statistical_unit='source article or conservative project-name family',primary_contrasts='confirmatory half: K10 minus V01 mean relative L2 over 8 steps at L2048; separately domain x isolated/recurrent; four contrasts; bootstrap document blocks; familywise 98.75% per-contrast intervals')
    return c
def planned(payload,smoke=False):
    docs=payload['documents'][:2] if smoke else payload['documents'];jobs=[]
    for dtype in ['bfloat16','float32']:
        for d in docs:
            if dtype=='float32' and not d['dtype_audit']:continue
            for length in [512,2048]:jobs.append(dict(document_id=d['document_id'],category=d['domain'],split=d['split'],length=length,dtype=dtype,order=int(digest(d['document_id'].encode())[:8],16)%2))
    return jobs
def batches_for(jobs):
    return [part[j:j+8] for dtype in ['bfloat16','float32'] for part in [[s for s in jobs if s['dtype']==dtype]] for j in range(0,len(part),8)]
def ident(s):return digest(canonical(s).encode())[:16]
def batch_worker(folder,plan):
    import torch
    from transformers import AutoModelForCausalLM
    from scope.population_collect import collect
    payload,sha=inputs();docs={d['document_id']:d for d in payload['documents']};c=config();inventory=gpu_gate(c)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(c['max_allocated_hbm_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
    specs=json.loads(Path(plan).read_text());assert len({s['dtype'] for s in specs})==1;dtype=getattr(torch,specs[0]['dtype'])
    model=AutoModelForCausalLM.from_pretrained(c['model']['id'],revision=c['model']['revision'],local_files_only=True,trust_remote_code=False,dtype=dtype,attn_implementation='eager').eval().to('cuda')
    for spec in specs:
        job=folder/ident(spec);job.mkdir(exist_ok=True);done=job/'completed.json'
        if done.exists():
            verify(job/json.loads(done.read_text())['attempt']);continue
        dest=job/('attempt-'+str(time.time_ns()));dest.mkdir()
        atomic(folder/'worker_progress.json',dict(pid=os.getpid(),condition=ident(spec),spec=spec,started=now()))
        began=time.monotonic()
        try:r=collect(spec,dest,dict(c,documents={spec['document_id']:docs[spec['document_id']]},input_manifest_sha256=sha),shared={'model':model,'inventory':inventory})
        except Exception as e:r=dict(status='OOM' if 'out of memory' in str(e).lower() else 'FAIL',error=str(e),traceback=traceback.format_exc());atomic(dest/'failure.json',r)
        r.update(elapsed_seconds=time.monotonic()-began);seal(dest,r)
        atomic(done,dict(id=ident(spec),spec=spec,attempt=dest.name,result=r,ended=now()))
        print('CONDITION',ident(spec),r['status'],round(r['elapsed_seconds'],2),flush=True)
        gc.collect();torch.cuda.empty_cache()
        if r['status']=='CONTROL_FAIL':raise RuntimeError('CONTROL_FAIL: stop batch')
def registry(folder,jobs,final=False):
    records=[]
    for s in jobs:
        p=folder/ident(s)/'completed.json'
        if p.exists():
            r=json.loads(p.read_text())
            if final:verify(p.parent/r['attempt'])
            records.append(r)
    atomic(folder/'registry.json',records);return records
def export(folder,records,complete=False):
    from scope.population_stats import analyze
    analyze(folder,records,complete=complete)
    archive=ROOT.parent/('SCOPE_POPULATION_'+folder.name+'.zip');tmp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob('*'):
            if f.is_file() and f.suffix not in ['.pt','.tmp']:z.write(f,f.relative_to(ROOT).as_posix())
        for root in ['scope','tests','configs','docs']:
            for f in (ROOT/root).rglob('*'):
                if f.is_file() and f.suffix not in ['.pyc']:z.write(f,f.relative_to(ROOT).as_posix())
        z.write(MANIFEST,'data/population_v1/manifest.json');z.write(MANIFEST.parent/'manifest.sha256','data/population_v1/manifest.sha256')
    os.replace(tmp,archive);archive.with_suffix('.zip.sha256').write_text(digest(archive.read_bytes())+'  '+archive.name+'\n')
def run(smoke=False):
    import fcntl
    payload,sha=inputs();c=config();identity=immutable_identity()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise RuntimeError('freeze tracked source first')
    key=digest(canonical(dict(config=c,input_sha=sha,identity=identity)).encode())[:16]+('-smoke' if smoke else '')
    folder=ROOT/'results/population'/key;folder.mkdir(parents=True,exist_ok=True);jobs=planned(payload,smoke)
    atomic(folder/'campaign.json',dict(config=c,identity=identity,input_manifest_sha256=sha,planned_independent_units=2 if smoke else 512,planned_conditions=len(jobs),created=now(),smoke=smoke))
    atomic(folder/'plan.json',jobs);print('RUN_DIRECTORY',folder,flush=True)
    lock=(ROOT.parent/'engineering_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    stop=threading.Event();live={};start=time.monotonic();state='RUNNING'
    def beat():
        while not stop.is_set():atomic(folder/'heartbeat.json',dict(pid=os.getpid(),updated=now(),**live));stop.wait(15)
    t=threading.Thread(target=beat,daemon=True);t.start()
    try:
        for i,chunk in enumerate(batches_for(jobs)):
            if time.monotonic()-start>c['campaign_timeout_seconds']:state='BUDGET_EXHAUSTED';break
            pending=[s for s in chunk if not (folder/ident(s)/'completed.json').exists()]
            if pending:
                gpu_gate(c);plan=folder/('batch-'+str(i)+'.json');atomic(plan,pending)
                with (folder/('batch-'+str(i)+'.log')).open('w') as log:
                    p=subprocess.Popen([sys.executable,'-m','scope.population','worker','--folder',str(folder),'--plan',str(plan)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);live.update(child_pid=p.pid,batch=i)
                    try:rc=p.wait(timeout=c['job_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        p.terminate()
                        try:p.wait(timeout=10)
                        except subprocess.TimeoutExpired:p.kill();p.wait()
                        rc=124
                live['child_pid']=None
                for s in pending:
                    job=folder/ident(s)
                    if not (job/'completed.json').exists():
                        job.mkdir(exist_ok=True);dest=job/('attempt-parent-'+str(time.time_ns()));dest.mkdir();r=dict(status='TIMEOUT' if rc==124 else 'FAIL',exit_code=rc,reason='batch worker did not seal this planned condition; inspect batch log');seal(dest,r);atomic(job/'completed.json',dict(id=ident(s),spec=s,attempt=dest.name,result=r))
            records=registry(folder,jobs);statuses=collections.Counter(r['result']['status'] for r in records)
            primary_units={r['spec']['document_id'] for r in records if r['spec']['dtype']=='bfloat16' and r['result']['status']=='DATA_COLLECTED'}
            atomic(folder/'status.json',dict(status='RUNNING',processed_conditions=len(records),planned_conditions=len(jobs),observed_primary_units=len(primary_units),planned_independent_units=2 if smoke else 512,counts=dict(statuses),updated=now()))
            print('PROGRESS',len(records),'/',len(jobs),dict(statuses),flush=True)
            if any(r['result']['status']=='CONTROL_FAIL' for r in records):state='CONTROL_FAILURE_STOP';break
            if len(records)==8 or len(records)%128==0:export(folder,records)
        records=registry(folder,jobs,True)
        if state=='RUNNING':state='COMPLETE'
        atomic(folder/'status.json',dict(status=state,processed_conditions=len(records),planned_conditions=len(jobs),counts=dict(collections.Counter(r['result']['status'] for r in records)),updated=now()))
        export(folder,records,complete=state=='COMPLETE');print('FINAL',state,len(records),flush=True)
    except Exception as e:
        atomic(folder/'status.json',dict(status='BLOCKED_OR_FAILED',error=str(e),traceback=traceback.format_exc(),updated=now()))
        records=registry(folder,jobs);export(folder,records);raise
    finally:stop.set();t.join(2);atomic(folder/'heartbeat.json',dict(pid=os.getpid(),state='STOPPED',updated=now()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--smoke',action='store_true');p.add_argument('--folder');p.add_argument('--plan');a=p.parse_args()
    if a.mode=='worker':batch_worker(Path(a.folder),a.plan)
    else:run(a.smoke)
