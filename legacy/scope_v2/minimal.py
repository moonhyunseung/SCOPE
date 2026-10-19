"""One-case native-operation shape reproducer; exact frozen teacher inputs required."""
import argparse,copy,json,os
from pathlib import Path
from scope_v2.diagnose import ROOT,MODEL,REV,Trace
from scope_v2.mechanism import InputTrace,modules,matmul_probe,projection_probe
from scope.engineering import metric

def reproduce(plan,case_id,step,operation,dtype='bfloat16'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4);torch.manual_seed(20261018);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    import fcntl
    from scope.runner import gpu_gate
    config=json.loads(Path(plan).read_text())
    lock=(ROOT.parent/'engineering_gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_gate(config)
    torch.cuda.set_per_process_memory_fraction(96*2**30/torch.cuda.get_device_properties(0).total_memory)
    case=next(x for x in config['cases'] if x['case_id']==case_id)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,local_files_only=True,trust_remote_code=False,
        dtype=getattr(torch,dtype),attn_implementation='eager').eval().to('cuda')
    inp=torch.tensor([case['tokens']['prefix_ids']],device='cuda');t=torch.tensor([case['tokens']['continuation_ids']],device='cuda')
    with torch.inference_mode():
        past=model(inp,use_cache=True,logits_to_keep=1).past_key_values
        for s in range(step):past=model(t[:,s:s+1],past_key_values=past,use_cache=True,logits_to_keep=1).past_key_values
        with InputTrace(model) as tc:cached=model(t[:,step:step+1],past_key_values=past,use_cache=True,logits_to_keep=1)
        whole=torch.cat([inp,t[:,:step+1]],1)
        with InputTrace(model) as tf:full=model(whole,use_cache=False,logits_to_keep=1)
        if operation=='mlp_down':
            x=tc.inputs[operation];p=projection_probe(modules(model)[operation],x,whole.shape[1])
            a,b=p['one'],p['many'];native=tc.values['L00.mlp_down'];native_full=tf.values['L00.mlp_down']
            upstream_exact=bool(torch.equal(x,tf.inputs[operation]))
        else:
            k=tc.values['L00.cache_k'].cuda().to(model.dtype);v=tc.values['L00.cache_v'].cuda().to(model.dtype)
            q=tc.values['L00.rope_q'].unsqueeze(-2).cuda().to(model.dtype);rep=q.shape[1]//k.shape[1]
            if operation=='QK':
                aleft=q;right=k.repeat_interleave(rep,1).transpose(-2,-1);key='L00.scores'
                upstream_exact=all(torch.equal(tc.values[n],tf.values[n]) for n in ['L00.rope_q','L00.cache_k'])
            else:
                aleft=tc.values['L00.probabilities'].unsqueeze(-2).cuda().to(model.dtype);right=v.repeat_interleave(rep,1);key='L00.attention_out'
                upstream_exact=all(torch.equal(tc.values[n],tf.values[n]) for n in ['L00.probabilities','L00.cache_v'])
            p,a=matmul_probe(aleft,right,whole.shape[1])
            b=torch.matmul(aleft.expand(*aleft.shape[:-2],whole.shape[1],aleft.shape[-1]).contiguous(),right)[...,-1:,:]
            if operation=='QK':
                scale=model.model.layers[0].self_attn.scaling;a=a*scale;b=b*scale
            a=a.squeeze(-2).cpu();b=b.squeeze(-2).cpu();native=tc.values[key];native_full=tf.values[key]
        result=dict(case_id=case_id,step=step,operation=operation,dtype=dtype,rows=whole.shape[1],
            token_sha256=case['tokens']['sha256'],model=MODEL,revision=REV,upstream_exact=upstream_exact,
            one_vs_many=metric(a,b),one_vs_native=metric(a.reshape_as(native),native),
            many_vs_native_full=metric(b.reshape_as(native_full),native_full),native_gap=metric(native,native_full),
            final_logits=metric(cached.logits[:,-1],full.logits[:,-1]))
        print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--case-id',required=True);p.add_argument('--step',type=int,choices=range(8),required=True)
    p.add_argument('--operation',choices=['mlp_down','QK','PV'],required=True);p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16');a=p.parse_args()
    reproduce(a.plan,a.case_id,a.step,a.operation,a.dtype)
