"""ACL-R1 bounded follow-up on the three already sealed BF16 reproducers.

Profiler kernel names are observations, never inferred cuBLAS algorithm IDs.
No new cases, relaxed gates or full-corpus tracing.
"""
import argparse,copy,json,os,sqlite3,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from scope.runner import atomic,canonical,digest,gpu_gate,now,source_hash
from scope_v2.diagnose import ROOT,MODEL,REV,tensor_hash
from scope_v2.mechanism import InputTrace,modules
from scope.engineering import metric

def fingerprint():
    return digest(b''.join(p.name.encode()+b'\0'+p.read_bytes() for p in [Path(__file__),ROOT/'scope_v2/minimal.py',ROOT/'scope_v2/mechanism.py',ROOT/'scope_v2/diagnose.py']))

def row_counts(native_rows):return sorted({1,2,8,32,native_rows})

def seal(dest,result):
    from scope_v2.diagnose import csv_write
    dest.mkdir(parents=True,exist_ok=True);atomic(dest/'summary.json',result)
    csv_write(dest/'metrics.csv',result.get('records',[]))
    (dest/'report.md').write_text('# BF16 shape-cause bounded follow-up, ACL-R1\n\n'+json.dumps(result,indent=2)+'\n\nThree pre-existing frozen reproducers only. Same-weight FP32 is a diagnostic arithmetic control, not original-model full-reference fidelity. Profiling overhead is not benchmark latency. Kernel names do not establish cuBLAS/cuBLASLt algorithm IDs; those remain UNKNOWN unless independently observed. No exhaustive explanation of historical453 failures.\n')
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in dest.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',dict(files=files,code_sha256=fingerprint(),sealed=now()));(dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))

def profile_kernels(call):
    import torch
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
            y=call();torch.cuda.synchronize()
        names=sorted({e.name for e in p.events() if str(e.device_type).endswith('CUDA')})
        return dict(status='OBSERVED_KERNEL_NAMES' if names else 'CUDA_KERNEL_NAMES_UNAVAILABLE',kernel_names=names,algorithm_id='UNKNOWN'),y
    except Exception as exc:
        # Failure of optional profiling is not a fabricated algorithm observation.
        return dict(status='PROFILER_UNAVAILABLE',error_type=type(exc).__name__,algorithm_id='UNKNOWN'),call()

def probe(model,case,prior,flag_default):
    import torch
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag_default
    operation=prior['operation'];step=prior['step'];inp=torch.tensor([case['tokens']['prefix_ids']],device='cuda');teacher=torch.tensor([case['tokens']['continuation_ids']],device='cuda')
    with torch.inference_mode():
        past=model(inp,use_cache=True,logits_to_keep=1).past_key_values
        for s in range(step):past=model(teacher[:,s:s+1],past_key_values=past,use_cache=True,logits_to_keep=1).past_key_values
        before=copy.deepcopy(past);plain=model(teacher[:,step:step+1],past_key_values=past,use_cache=True,logits_to_keep=1)
        with InputTrace(model) as cached:observed=model(teacher[:,step:step+1],past_key_values=before,use_cache=True,logits_to_keep=1)
        assert torch.equal(observed.logits,plain.logits),'observer changes native logits'
        whole=torch.cat([inp,teacher[:,:step+1]],1)
        with InputTrace(model) as full:full_out=model(whole,use_cache=False,logits_to_keep=1)
        del plain,observed,full_out,past,before
        if operation=='mlp_down':
            a=cached.inputs[operation];layer=modules(model)[operation];right=layer.weight;bias=layer.bias;name='L00.mlp_down'
            assert torch.equal(a,full.inputs[operation]),'upstream mismatch'
            def calculate(rows,fp32=False):
                x=a.expand(*a.shape[:-2],rows,a.shape[-1]).contiguous()
                if fp32:return F.linear(x.float(),right.float(),bias.float() if bias is not None else None)[...,-1:,:]
                return F.linear(x,right,bias)[...,-1:,:]
        else:
            k=cached.values['L00.cache_k'].cuda().to(model.dtype);v=cached.values['L00.cache_v'].cuda().to(model.dtype)
            q=cached.values['L00.rope_q'].unsqueeze(-2).cuda().to(model.dtype);rep=q.shape[1]//k.shape[1]
            if operation=='QK':
                a=q;right=k.repeat_interleave(rep,1).transpose(-2,-1);name='L00.scores';scale=model.model.layers[0].self_attn.scaling
                assert all(torch.equal(cached.values[n],full.values[n]) for n in ['L00.rope_q','L00.cache_k']),'upstream mismatch'
            else:
                a=cached.values['L00.probabilities'].unsqueeze(-2).cuda().to(model.dtype);right=v.repeat_interleave(rep,1);name='L00.attention_out';scale=1.
                assert all(torch.equal(cached.values[n],full.values[n]) for n in ['L00.probabilities','L00.cache_v']),'upstream mismatch'
            def calculate(rows,fp32=False):
                x=a.expand(*a.shape[:-2],rows,a.shape[-1]).contiguous()
                y=torch.matmul(x.float(),right.float()) if fp32 else torch.matmul(x,right)
                return y[...,-1:,:]*scale
        native=cached.values[name];native_full=full.values[name];N=whole.shape[1]
        sha_a=tensor_hash(a);sha_b=tensor_hash(right);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag_default
        one=calculate(1).cpu();many=calculate(N).cpu()
        assert torch.equal(one.reshape_as(native),native),'one-row native replay mismatch'
        assert torch.equal(many.reshape_as(native_full),native_full),'N-row full native replay mismatch'
        records=[];kernel=[];flags={};reference_fp32=calculate(1,True).cpu()
        for reduced in [True,False]:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
            flag_one=calculate(1).cpu()
            for rows in row_counts(N):
                metadata,y=profile_kernels(lambda:calculate(rows));z=y.cpu();del y
                flags[reduced,rows]=z
                # Profiling cannot alter the numeric output.
                unprofiled=calculate(rows).cpu();assert torch.equal(z,unprofiled),'profiling changes output'
                fp32=calculate(rows,True).cpu()
                m=metric(z,flag_one);f=metric(z,fp32);fp_shape=metric(fp32,reference_fp32);default=metric(z,one)
                records.append(dict(operation=operation,case_id=case['case_id'],step=step,rows=rows,native_rows=N,reduced_precision_reduction=reduced,
                    shape_relative_l2=m['relative_l2'],shape_max_abs=m['max_abs'],default_native_relative_l2=default['relative_l2'],same_weight_fp32_relative_l2=f['relative_l2'],same_weight_fp32_shape_relative_l2=fp_shape['relative_l2'],
                    output_sha256=tensor_hash(z),fp32_output_sha256=tensor_hash(fp32),left_sha256=sha_a,right_sha256=sha_b,cuBLAS_algorithm_id='UNKNOWN'))
                kernel.append(dict(operation=operation,rows=rows,reduced_precision_reduction=reduced,**metadata))
        pairs=[dict(operation=operation,rows=rows,raw_output_bitwise_equal=bool(torch.equal(flags[True,rows],flags[False,rows])),true_sha256=tensor_hash(flags[True,rows]),false_sha256=tensor_hash(flags[False,rows])) for rows in row_counts(N)]
        assert tensor_hash(a)==sha_a and tensor_hash(right)==sha_b,'operand mutated'
        result=dict(operation=operation,case_id=case['case_id'],step=step,native_rows=N,source_unit=case['document_id'],token_sha256=case['tokens']['sha256'],
            upstream_exact=True,native_cached_proxy_exact=True,native_full_proxy_exact=True,operands_unchanged=True,left_shape=list(a.shape),right_shape=list(right.shape),
            left_stride=list(a.stride()),right_stride=list(right.stride()),records=records,raw_flag_pairs=pairs,kernel_observations=kernel,
            native_gap=metric(native,native_full),original_minimal_result_canonical_sha256=digest(canonical(prior).encode()))
        del cached,full,a,right,one,many,flags;return result

def worker(plan,minimal,dest):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch,transformers,platform
    from transformers import AutoModelForCausalLM
    config=json.loads(plan.read_text());priors=[json.loads((minimal/(name+'.json')).read_text()) for name in ['mlp_down','QK','PV']]
    result=dict(status='RUNNING',model=MODEL,revision=REV,plan_sha256=digest(plan.read_bytes()),baseline_minimal_manifest_sha256=digest((minimal/'manifest.json').read_bytes()),records=[],cases=[],publication='PENDING',
        runtime=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda),algorithm_id='UNKNOWN',additional_gpu_budget_seconds=1800)
    flag=None
    try:
        gpu_gate(config);assert source_hash()=='c072fa01688f90f2d516725d60b01c3d11ac7e380472823a21fe0164db066947'
        torch.set_num_threads(4);torch.manual_seed(20261018);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.set_per_process_memory_fraction(96*2**30/torch.cuda.get_device_properties(0).total_memory);flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').eval().cuda()
        result['runtime'].update(gpu=torch.cuda.get_device_name(),default_bf16_reduction_flag=flag)
        for prior in priors:
            case=next(c for c in config['cases'] if c['case_id']==prior['case_id']);assert prior['dtype']=='bfloat16' and prior['upstream_exact']
            assert prior['token_sha256']==case['tokens']['sha256'],'frozen token identity mismatch'
            r=probe(model,case,prior,flag);result['cases'].append(r);result['records'].extend(r['records'])
            atomic(dest/(prior['operation']+'.json'),r)
        result.update(status='COLLECTED',case_operations=3,all_native_replays_exact=True,raw_flag_pairs=sum(len(r['raw_flag_pairs']) for r in result['cases']),raw_flag_pairs_different=sum(not p['raw_output_bitwise_equal'] for r in result['cases'] for p in r['raw_flag_pairs']))
    except Exception as exc:
        status='OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'CONTROL_FAIL' if isinstance(exc,AssertionError) else 'FAIL'
        result.update(status=status,reason=type(exc).__name__+': '+str(exc));(dest.parent/'worker.private.log').write_text(traceback.format_exc())
    finally:
        if flag is not None:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
        seal(dest,result)
    if result['status']!='COLLECTED':raise RuntimeError('STOP: sealed bounded shape-cause failure')

def run(plan,minimal):
    import fcntl
    from scope_v2.replication import verify
    verify(minimal)
    identity=dict(source_sha256=fingerprint(),plan_sha256=digest(plan.read_bytes()),minimal_manifest_sha256=digest((minimal/'manifest.json').read_bytes()))
    folder=ROOT/'results/shape_cause_v2'/digest(canonical(identity).encode())[:16];folder.mkdir(parents=True,exist_ok=True);atomic(folder/'campaign.json',identity)
    dest=folder/'blocks/probe'
    if (dest/'manifest.json').exists():return
    deadline=time.monotonic()+86400
    while True:
        statuses=[json.loads(p.read_text())['status'] for kind in ['replication_v2','resident_v2','distributions_v2'] for p in (ROOT/'results'/kind).glob('*/status.json')]
        if len(statuses)>=3 and all(s.startswith('COMPLETE') or s in ['STOPPED_WITH_FAILURE','FAIL'] for s in statuses):break
        if time.monotonic()>deadline:atomic(folder/'status.json',dict(status='BLOCKED_UPSTREAM_TIMEOUT',updated=now()));return
        atomic(folder/'status.json',dict(status='WAITING_FOR_EXISTING_GPU_QUEUE',updated=now()));time.sleep(30)
    lock=(ROOT.parent/'engineering_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);dest.mkdir(parents=True,exist_ok=True)
    try:
        atomic(folder/'status.json',dict(status='RUNNING',updated=now()))
        with (folder/'launch.private.log').open('w') as log:
            p=subprocess.Popen([sys.executable,'-u','-m','scope_v2.shape_cause','worker','--plan',str(plan.resolve()),'--minimal',str(minimal.resolve()),'--dest',str(dest.resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:rc=p.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
                rc=124
        if not (dest/'manifest.json').exists():seal(dest,dict(status='TIMEOUT' if rc==124 else 'FAIL',reason='bounded worker stopped; partial files retained',publication='PENDING'))
        from scope_v2.replication import verify
        r=verify(dest);db=sqlite3.connect(folder/'registry.sqlite');db.execute('CREATE TABLE IF NOT EXISTS publications(id PRIMARY KEY,status,manifest_sha,updated)')
        db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',('probe','PENDING',digest((dest/'manifest.json').read_bytes()),now()));db.commit();db.close()
        with zipfile.ZipFile(ROOT.parent/'SCOPE_BF16_SHAPE_CAUSE_V2.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in dest.iterdir():z.write(f,f.relative_to(ROOT).as_posix())
        atomic(folder/'status.json',dict(status='COMPLETE' if not rc else 'STOPPED_WITH_FAILURE',result=r['status'],updated=now()))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--minimal',type=Path,required=True);p.add_argument('--dest',type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a.plan,a.minimal,a.dest)
    else:run(a.plan,a.minimal)
