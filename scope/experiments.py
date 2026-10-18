"""Isolated GPU workers: explicit references, actual-model capture and bounded replays."""
from __future__ import annotations
import argparse, gc, hashlib, importlib, importlib.metadata, inspect, json, math, os, platform, statistics, subprocess, sys, threading, time, traceback
from pathlib import Path
from scope.runner import ROOT, atomic, canonical, command, digest, gpu_gate, now, verify

class Unsupported(RuntimeError): pass
def versions():
    result={}
    for x in ["torch","triton","transformers","numpy","pytest","huggingface_hub","flash_attn","vllm"]:
        try:result[x]=importlib.metadata.version(x)
        except importlib.metadata.PackageNotFoundError:result[x]=None
    return result
def preflight(c):
    inventory=gpu_gate(c)
    rc,processes=command(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader"])
    import torch,triton
    if not torch.cuda.is_available() or not torch.version.cuda:raise RuntimeError("CUDA PyTorch required")
    if "H200" not in torch.cuda.get_device_name(0):raise RuntimeError("wrong CUDA device")
    env={"recorded":now(),"python":sys.version,"platform":platform.platform(),"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"hbm_bytes":torch.cuda.get_device_properties(0).total_memory,"budget_gib":c["max_allocated_hbm_gib"],"reserve_gib":c["reserve_hbm_gib"],"git_sha":command(["git","rev-parse","HEAD"])[1].strip(),"git_publication":"requires existing Git credentials and author identity; unverified until push/read-back"}
    base=ROOT/"results"/"preflight";base.mkdir(parents=True,exist_ok=True)
    for name,value in [("ENVIRONMENT",env),("GPU_INVENTORY",{"gpu_rows":inventory,"process_inventory":processes.strip(),"captured_before_workload":True}),("SOFTWARE_MANIFEST",versions())]:
        path=base/(name+".json")
        if path.exists() and json.loads(path.read_text())!=value:
            path=base/(name+"-"+str(time.time_ns())+".json")
        atomic(path,value)
    return env
def errors(x,y):
    import torch
    diff=(x.float()-y.float())
    return {"max_abs":float(diff.abs().max()),"relative_l2":float(diff.norm()/y.float().norm().clamp_min(1e-12))}
def p0(dest,c):
    import torch
    import torch.nn.functional as F
    from scope.reference import reference_decode_attention,symmetric_int4_groupwise
    preflight(c)
    rc,log=command([sys.executable,"-m","pytest","-q"],timeout=120)
    (dest/"pytest.log").write_text(log)
    if rc:raise RuntimeError("upstream/infrastructure tests failed; see pytest.log")
    rows=[]
    torch.manual_seed(c["seed"])
    for device in ["cpu","cuda"]:
        for hq,hkv in [(4,4),(6,2),(4,1)]:
            q=torch.randn(1,hq,1,32,device=device)
            k=torch.randn(1,hkv,17,32,device=device);v=torch.randn_like(k)
            for append in range(3):
                k=torch.cat([k,torch.randn(1,hkv,1,32,device=device)],2)
                v=torch.cat([v,torch.randn(1,hkv,1,32,device=device)],2)
                kq=symmetric_int4_groupwise(k,16);vq=symmetric_int4_groupwise(v,16)
                ref=reference_decode_attention(q,kq.decoded,vq.decoded,v_scales=vq.scales)
                target=F.scaled_dot_product_attention(q,kq.decoded.repeat_interleave(hq//hkv,1),vq.decoded.repeat_interleave(hq//hkv,1))
                err=errors(ref.y,target)
                if not torch.allclose(ref.y,target,atol=5e-4,rtol=5e-4):raise RuntimeError("P0 SDPA mismatch")
                rows.append({"device":device,"hq":hq,"hkv":hkv,"append":append,"length":k.shape[2],**err,"status":"PASS"})
    return {"status":"PASS","metrics":rows,"provenance":{"input_kind":"synthetic","seed":c["seed"],"software":versions()},"interpretation":"Synthetic CPU/H200 MHA/GQA/MQA attention and three-step cache append checks passed. Model-specific RoPE and logits trajectory are gated in P1. Masked attention coverage is in tests; no model-fidelity or speed claim."}
def describe(x):
    import torch
    from scope.observations import tensor_summary
    d=tensor_summary(x,max_samples=8192)
    sample=x.detach().reshape(-1)
    sample=sample[::max(1,(sample.numel()+8191)//8192)][:8192].float()
    _,exp=torch.frexp(sample)
    unique,counts=torch.unique(exp,return_counts=True)
    d["sample_exponents"]={str(int(a)):int(b) for a,b in zip(unique.cpu(),counts.cpu())}
    return d
def pair_stats(a,b):
    import torch
    a,b=torch.broadcast_tensors(a,b)
    step=max(1,(a.numel()+8191)//8192)
    a=a.reshape(-1)[::step][:8192].float();b=b.reshape(-1)[::step][:8192].float()
    x=a-a.mean();y=b-b.mean();den=x.norm()*y.norm()
    return {"n":a.numel(),"correlation":float((x*y).sum()/den) if float(den)>0 else None,"mean_product":float((a*b).mean())}
def native_eager_reference(q,k,v,mask,scale):
    """Reproduce original eager dtype/rounding; FP32 diagnostics are separate."""
    import torch
    rep=q.shape[1]//k.shape[1]
    kh=k.repeat_interleave(rep,1);vh=v.repeat_interleave(rep,1)
    z=torch.matmul(q,kh.transpose(-2,-1))*scale
    if mask is not None:
        z=z.masked_fill(~mask,float("-inf")) if mask.dtype==torch.bool else z+mask[...,:k.shape[-2]]
    prob=torch.softmax(z,dim=-1,dtype=torch.float32).to(q.dtype)
    return z,prob,torch.matmul(prob,vh)

def capture(dest,c,p):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    from scope.reference import symmetric_int4_groupwise
    from scope.observations import probability_summary,code_histogram
    model_spec=p["model"];mid=model_spec["id"];rev=model_spec["revision"]
    cfg=AutoConfig.from_pretrained(mid,revision=rev,trust_remote_code=False)
    model_type=cfg.model_type
    if model_type not in ["qwen2","gpt_neox","opt"]:raise Unsupported("unimplemented actual-model adapter: "+model_type)
    length=p["length"]
    if length+c["decode_steps"]>getattr(cfg,"max_position_embeddings",10**9):raise Unsupported("requested context plus decode exceeds model position limit")
    module=importlib.import_module("transformers.models."+model_type+".modeling_"+model_type)
    original=getattr(module,"eager_attention_forward",None)
    if original is None:raise Unsupported("installed model has no eager function")
    tokenizer=AutoTokenizer.from_pretrained(mid,revision=rev,trust_remote_code=False)
    text=c["inputs"][p["category"]]
    # Public, fixed authored text repeated to the frozen exact token population.
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids:raise RuntimeError("empty tokenizer input")
    tokens=(ids*((length+len(ids)-1)//len(ids)))[:length]
    inputs=torch.tensor([tokens],device="cuda")
    model=AutoModelForCausalLM.from_pretrained(mid,revision=rev,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation="eager").eval().to("cuda")
    attention_modules=[(name,m) for name,m in model.named_modules() if type(m).__name__.endswith("Attention")]
    if not attention_modules:raise Unsupported("no attention modules matched")
    selected={id(attention_modules[i][1]):attention_modules[i][0] for i in sorted({0,len(attention_modules)//2,len(attention_modules)-1})}
    metrics=[];states=[];step=[-1];active=[False];sample=[False];head_counts=set()
    def observed(attention,q,k,v,mask,*args,**kwargs):
        result=original(attention,q,k,v,mask,*args,**kwargs)
        if not active[0] or q.shape[-2]!=1 or id(attention) not in selected:return result
        bound=inspect.signature(original).bind_partial(attention,q,k,v,mask,*args,**kwargs);bound.apply_defaults()
        scale=bound.arguments.get("scaling")
        if scale is None:scale=getattr(attention,"scaling",q.shape[-1]**-0.5)
        rep=q.shape[1]//k.shape[1]
        if q.shape[1]%k.shape[1]:raise Unsupported("noncontiguous/irregular head mapping")
        head_counts.add((q.shape[1],k.shape[1]))
        kh=k.float().repeat_interleave(rep,1);vh=v.float().repeat_interleave(rep,1)
        z=(q.float()@kh.transpose(-2,-1))*scale
        if mask is not None:
            if mask.dtype==torch.bool:z=z.masked_fill(~mask,float("-inf"))
            else:z=z+mask[...,:k.shape[-2]].float()
        prob=z.softmax(-1);y=prob@vh
        native=result[0].transpose(1,2)
        diagnostic_error=errors(y,native)
        native_z,native_p,native_ref=native_eager_reference(q,k,v,mask,scale)
        native_error=errors(native_ref,native)
        if not torch.equal(native_ref,native):
            torch.save({"q":q.detach().cpu(),"k":k.detach().cpu(),"v":v.detach().cpu(),"mask":mask.detach().cpu() if mask is not None else None,"scale":float(scale),"native":native.detach().cpu()},dest/"failure_reproducer.pt")
            raise RuntimeError("same-dtype native eager reference mismatch: "+str(native_error))
        group=32 if k.shape[-1]%32==0 else k.shape[-1]
        kq=symmetric_int4_groupwise(k,group);vq=symmetric_int4_groupwise(v,group)
        w=prob.squeeze(2).unsqueeze(-1)*vq.scales.repeat_interleave(rep,1)
        yq=((q.float()@kq.decoded.repeat_interleave(rep,1).transpose(-2,-1))*scale)
        if mask is not None:yq=yq+(mask[...,:k.shape[-2]].float() if mask.dtype!=torch.bool else torch.where(mask,0.,float("-inf")))
        yq=yq.softmax(-1)@vq.decoded.repeat_interleave(rep,1)
        _,_,quantized_native=native_eager_reference(q,kq.decoded.to(q.dtype),vq.decoded.to(q.dtype),mask,scale)
        abs_pv=prob@vh.abs()
        cancel=float((y.abs()/abs_pv.clamp_min(1e-12)).mean())
        layer=selected[id(attention)]
        # Scalars/head and deterministic bounded tensor samples; no per-layer tensor archive.
        for head in range(q.shape[1]):
            kvhead=head//rep
            row={"model":mid,"revision":rev,"layer":layer,"head":head,"kv_head":kvhead,"hq":q.shape[1],"hkv":k.shape[1],"L":k.shape[-2],"decode_step":step[0],"category":p["category"],"native_relative_l2":errors(native_ref[:,head],native[:,head])["relative_l2"],"fp32_diagnostic_vs_native_relative_l2":errors(y[:,head],native[:,head])["relative_l2"],"quantized_relative_l2":errors(quantized_native[:,head],native[:,head])["relative_l2"],"fp32_quantized_relative_l2":errors(yq[:,head],y[:,head])["relative_l2"],**probability_summary(prob[:,head:head+1]),"status":"PASS"}
            metrics.append(row)
        detail={"layer":layer,"decode_step":step[0],"Q":describe(q),"K":describe(k),"V":describe(v),"Z":describe(z),"P":describe(prob),"Y":describe(y),"P_native":describe(result[1]),"Y_native":describe(native),"same_dtype_reference_error":native_error,"fp32_diagnostic_vs_native_error":diagnostic_error,"S_K":describe(kq.scales),"S_V":describe(vq.scales),"W":describe(w),"joint_P_scale":pair_stats(prob.squeeze(2).unsqueeze(-1),vq.scales.repeat_interleave(rep,1)),"joint_probability_V":pair_stats(prob.squeeze(2).unsqueeze(-1),vh),"joint_QK_products":pair_stats(q.float().squeeze(2).unsqueeze(2),kh),"mean_cancellation_ratio":cancel,"C_K":code_histogram(kq.codes),"C_V":code_histogram(vq.codes),"quantizer":"illustrative signed [-7,7], FP32 scales, group "+str(group)}
        states.append(detail)
        if not sample[0]:
            torch.save({"q":q.detach().cpu(),"k":k.detach().cpu(),"v":v.detach().cpu(),"mask":mask.detach().cpu() if mask is not None else None,"scale":float(scale),"model":mid,"revision":rev,"group":group,"layer":layer},dest/"sample.pt");sample[0]=True
        return result
    module.eager_attention_forward=observed
    torch.manual_seed(c["seed"]);trajectory=[]
    try:
        with torch.inference_mode():
            out=model(input_ids=inputs,use_cache=True)
            past=out.past_key_values;next_token=out.logits[:,-1:].argmax(-1)
            for position in range(c["decode_steps"]):
                step[0]=position;active[0]=True
                before=int(past.get_seq_length())
                inputs=torch.cat([inputs,next_token],1)
                out=model(input_ids=next_token,past_key_values=past,use_cache=True)
                past=out.past_key_values
                if int(past.get_seq_length())!=before+1:raise RuntimeError("cache append length mismatch")
                active[0]=False
                if position==0:
                    full=model(input_ids=inputs,use_cache=False)
                    error=errors(out.logits[:,-1],full.logits[:,-1])
                    trajectory.append(error)
                    if error["relative_l2"]>c["trajectory_relative_l2"]:raise RuntimeError("cached/full logits trajectory mismatch")
                    del full
                next_token=out.logits[:,-1:].argmax(-1)
    finally:module.eager_attention_forward=original
    if not metrics or not sample[0]:raise Unsupported("actual model did not call instrumented eager path")
    # Instrumentation is removed before full-model baseline timings.
    model_decode_baseline=[]
    for repeat in range(c["performance_repeats"]):
        with torch.inference_mode():
            replay_inputs=torch.tensor([tokens],device="cuda")
            replay=model(input_ids=replay_inputs,use_cache=True)
            replay_cache=replay.past_key_values
            token=replay.logits[:,-1:].argmax(-1)
            for warm in range(2):
                replay=model(input_ids=token,past_key_values=replay_cache,use_cache=True)
                replay_cache=replay.past_key_values;token=replay.logits[:,-1:].argmax(-1)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            timings=[]
            for step_index in range(6):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record()
                replay=model(input_ids=token,past_key_values=replay_cache,use_cache=True)
                replay_cache=replay.past_key_values;token=replay.logits[:,-1:].argmax(-1)
                b.record();b.synchronize();timings.append(float(a.elapsed_time(b)))
            storages={}
            for layer in replay_cache.layers:
                for name in ["keys","values"]:
                    tensor=getattr(layer,name,None)
                    if tensor is not None:storages[tensor.untyped_storage().data_ptr()]=tensor.untyped_storage().nbytes()
            model_decode_baseline.append({"repeat":repeat,"backend":"uninstrumented eager BF16","median_decode_ms":statistics.median(timings),"mean_decode_ms":statistics.mean(timings),"decode_tokens_per_second":1000/statistics.mean(timings),"timed_tokens":6,"initial_context":length,"final_cache_length":int(replay_cache.get_seq_length()),"cache_tensor_storage_bytes":sum(storages.values()),"peak_total_allocated_bytes":torch.cuda.max_memory_allocated(),"scope":"full-model dense-cache decode forward plus greedy token choice; prefill excluded; no compression integration","raw_ms":timings})
            del replay,replay_cache
    atomic(dest/"state_statistics.json",states)
    architecture=[{"hq":a,"hkv":b,"kind":"MHA" if a==b else ("MQA" if b==1 else "GQA")} for a,b in sorted(head_counts)]
    return {"status":"PASS","metrics":metrics,"model_decode_baseline":model_decode_baseline,"provenance":{"model_id":mid,"model_revision":rev,"model_type":model_type,"config_sha256":digest(cfg.to_json_string().encode()),"native_function_source_sha256":digest(inspect.getsource(original).encode()),"architecture_observed":architecture,"backend":"transformers eager BF16; FP32 diagnostic replay","input_dataset_id":"scope-public-repeated-"+p["category"]+"-v1","input_token_sha256":digest(canonical(tokens).encode()),"seed":c["seed"],"dtype":"bfloat16","native_contract":"BF16 matmul and scale, mask addition in native dtype, FP32 softmax cast to BF16, BF16 PV","batch":1,"context":length,"cache_append_steps":c["decode_steps"],"cached_full_logits_checks":trajectory,"software":versions(),"gpu":"NVIDIA H200","native_values":"Q,K,V, returned eager BF16 probability P and attention output Y","reconstructed_values":"FP32 diagnostic Z/P/Y/W; same-dtype Z/P/Y replay; illustrative quantizer codes/scales. FP32 difference is precision sensitivity, not executor error","trace_overhead":"not measured; no trace-mode latency reported"},"interpretation":"Actual pretrained-model decode capture with model-specific position handling and cache append. Native attention output checked by exact same-dtype replay; cached/full logits checked separately. FP32 diagnostic differences are measured precision sensitivity. Illustrative quantization is not the model's production cache or a quality-certified transformation."}
def pack(x,group):
    import torch
    from scope.reference import symmetric_int4_groupwise
    quant=symmetric_int4_groupwise(x,group)
    u=(quant.codes.to(torch.int16)+8).to(torch.uint8)
    packed=u[...,0::2]|(u[...,1::2]<<4)
    return packed,quant.scales,quant.decoded
def restore(packed,scales,group,dtype):
    import torch
    codes=torch.stack([(packed&15).to(torch.int16)-8,(packed>>4).to(torch.int16)-8],-1).flatten(-2)
    decoded=(codes.float().reshape(*codes.shape[:-1],-1,group)*scales.unsqueeze(-1)).reshape(codes.shape)
    return decoded.to(dtype)
def read_sample(folder,ident):
    import torch
    sample=folder/ident/"sample.pt"
    data=torch.load(sample,map_location="cuda",weights_only=True)
    return data
def attention(q,k,v,mask,scale):
    import torch.nn.functional as F
    rep=q.shape[1]//k.shape[1]
    return F.scaled_dot_product_attention(q,k.repeat_interleave(rep,1),v.repeat_interleave(rep,1),attn_mask=mask,scale=scale,is_causal=False)
def transform(dest,folder,p):
    import torch
    s=read_sample(folder,p["capture"]);q,k,v=s["q"],s["k"],s["v"]
    pk,sk,dk=pack(k,s["group"]);pv,sv,dv=pack(v,s["group"])
    rk=restore(pk,sk,s["group"],torch.float32);rv=restore(pv,sv,s["group"],torch.float32)
    if not torch.equal(rk,dk) or not torch.equal(rv,dv):raise RuntimeError("lossless packed layout roundtrip failed")
    original=attention(q,k,v,s["mask"],s["scale"])
    decoded=attention(q,dk.to(q.dtype),dv.to(q.dtype),s["mask"],s["scale"])
    error=errors(decoded,original)
    return {"status":"PASS","metrics":[{"capture":p["capture"],"signed_offset_pack_roundtrip":"PASS","representation_relative_l2":error["relative_l2"],"representation_max_abs":error["max_abs"],"status":"PASS"}],"hypotheses":[{"id":"T1-offset-pack-dense-restore","identity":"signed c = unsigned_nibble - 8; decoded c*s","input":"real post-position Q/K/V","persistent":"two uint8 nibbles plus FP32 group scales","intermediate":"temporary dense BF16","output":"BF16 SDPA","fidelity":"exact packed-to-FP32 decoded identity; BF16 replay consistency is separate","primitive":"vector bitwise + CUDA SDPA","bottleneck":"restoration traffic and launches","falsification":"roundtrip mismatch or total path slower than dense baseline"}],"interpretation":"Offset-packed layout roundtrip verified against decoded-quantized reference. Representation loss against original attention is reported separately. This does not establish end-model quality."}
def benchmark(dest,c,folder,p):
    import torch
    s=read_sample(folder,p["capture"]);q,k,v=s["q"],s["k"],s["v"]
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin.record();pk,sk,dk=pack(k,s["group"]);pv,sv,dv=pack(v,s["group"]);end.record();end.synchronize()
    packing_initialization_ms=float(begin.elapsed_time(end))
    mask,scale=s["mask"],s["scale"]
    original_native=native_eager_reference(q,k,v,mask,scale)[2]
    decoded_native=native_eager_reference(q,dk.to(q.dtype),dv.to(q.dtype),mask,scale)[2]
    def dense_eager():return native_eager_reference(q,k,v,mask,scale)[2]
    def dense_sdpa():return attention(q,k,v,mask,scale)
    def restore_eager():return native_eager_reference(q,restore(pk,sk,s["group"],q.dtype),restore(pv,sv,s["group"],q.dtype),mask,scale)[2]
    def restore_sdpa():return attention(q,restore(pk,sk,s["group"],q.dtype),restore(pv,sv,s["group"],q.dtype),mask,scale)
    variants=[("dense_eager",dense_eager,original_native),("dense_sdpa",dense_sdpa,original_native),("packed_restore_eager",restore_eager,decoded_native),("packed_restore_sdpa",restore_sdpa,decoded_native)]
    execution_errors={name:errors(fn(),ref) for name,fn,ref in variants}
    quality=errors(decoded_native,original_native)
    rows=[];passing=[];failed=[]
    for name,fn,ref in variants:
        if execution_errors[name]["relative_l2"]>c["native_reference_relative_l2"]:
            failed.append(name)
            rows.append({"capture":p["capture"],"variant":name,"status":"FAIL","failure_reason":"same-representation executor differs from native BF16 eager contract; timing prohibited","execution_relative_l2":execution_errors[name]["relative_l2"]})
        else:passing.append((name,fn))
    for repeat in range(c["performance_repeats"]):
        order=list(passing)
        if repeat%2:order.reverse()
        for variant,fn in order:
            for _ in range(c["warmup"]):fn()
            torch.cuda.synchronize()
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(c["timed_iterations"]):
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();result=fn();b.record();b.synchronize();times.append(float(a.elapsed_time(b)));del result
            peak=torch.cuda.max_memory_allocated()
            atomic(dest/(variant+"-repeat"+str(repeat)+".json"),times)
            allocated=(sum(x.untyped_storage().nbytes() for x in [pk,pv,sk,sv]) if variant.startswith("packed") else k.untyped_storage().nbytes()+v.untyped_storage().nbytes())
            rows.append({"capture":p["capture"],"variant":variant,"repeat":repeat,"median_ms":statistics.median(times),"mean_ms":statistics.mean(times),"p95_ms":sorted(times)[math.ceil(.95*len(times))-1],"iterations":len(times),"initial_packing_ms":packing_initialization_ms if variant.startswith("packed") else 0.0,"storage_allocation_bytes":allocated,"baseline_allocated_bytes":baseline,"peak_allocated_bytes":peak,"transient_peak_delta_bytes":peak-baseline,"execution_relative_l2":execution_errors[variant]["relative_l2"],"representation_relative_l2":quality["relative_l2"] if variant.startswith("packed") else 0.0,"status":"PASS"})
    return {"status":"PASS" if passing else "FAIL","metrics":rows,"failed_variants":failed,"provenance":{"model":s["model"],"revision":s["revision"],"layer":s["layer"],"shape_q":list(q.shape),"shape_k":list(k.shape),"dtype":str(q.dtype),"gpu":torch.cuda.get_device_name(),"software":versions(),"clocks":command(["nvidia-smi","--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw","--format=csv,noheader"])[1].strip(),"scope":"single-layer decode replay; includes unpack, dequantize, GQA materialization and attention; excludes one-time packing (separately recorded), full model and sampling","reference":"same BF16 eager contract as model; representation loss separate from executor error"},"interpretation":"Completed operator screening block; PASS at block level means results finalized, not every variant passed. Failed variants are retained and never timed. CUDA event latency, tensor storage and allocator peaks are measurements. Resident-KV capacity/full-model compressed-cache frontier is NOT measured. Native full-model dense baselines are in P1."}
def aggregate(dest,folder,block):
    records=[]
    for p in sorted(folder.glob("*/summary.json")):
        data=json.loads(p.read_text());records.append(data)
    if block=="P7":
        for p in folder.glob("*/manifest.json"):verify(p.parent)
    source="P1" if block in ["P2","P5"] else "P4"
    chosen=[x for x in records if x["id"].split("-")[0]==source]
    rows=[]
    for x in chosen:
        result=x["result"]
        if block=="P5":
            rows.append({"id":x["id"],"status":x["status"],"model":result.get("provenance",{}).get("model_id"),"architecture":json.dumps(result.get("provenance",{}).get("architecture_observed",[]))})
        elif block=="P2":
            if x["status"]=="PASS":
                samples=json.loads((folder/x["id"]/"state_statistics.json").read_text())
                for s in samples:rows.append({"capture":x["id"],"layer":s["layer"],"step":s["decode_step"],"cancellation_ratio":s["mean_cancellation_ratio"],"P_scale_correlation":s["joint_P_scale"]["correlation"],"P_V_correlation":s["joint_probability_V"]["correlation"],"status":"PASS"})
            else:rows.append({"capture":x["id"],"status":x["status"]})
        elif block=="P6":
            rows.extend([{"id":x["id"],**r} for r in result.get("metrics",[])] or [{"id":x["id"],"status":x["status"]}])
    if block=="P7":rows=[{"id":x["id"],"status":x["status"],"integrity":"PASS"} for x in records]
    passing=sum(x["status"]=="PASS" for x in chosen)
    status="PASS" if block=="P7" or passing else "BLOCKED"
    return {"status":status,"metrics":rows,"interpretation":{"P2":"Conditional and joint sampled statistics from actual captures; correlations do not prove universal invariants. Samples are deterministic and repetitive public text limits population coverage.","P5":"Cross-model coverage inventory; architecture is observed from actual Q/K/V head counts. MQA, hybrid, MLA, paged/vLLM, FlashAttention and custom Triton are UNSUPPORTED/NOT_TESTED in this pilot unless explicitly recorded elsewhere.","P6":"Exploratory paired pilot summaries only. Repeats are three timing windows, not independent model populations. No confirmatory factorial design, quality bound, uncertainty interval or resident-capacity frontier established.","P7":"Existing artifact hashes verified; condition failures retained. This is an integrity audit, not independent GPU replication or OSDI-class completeness."}[block]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--campaign",required=True);ap.add_argument("--job",required=True);args=ap.parse_args()
    folder=Path(args.campaign);meta=json.loads((folder/"campaign.json").read_text());c=meta["config"];dest=folder/args.job
    from scope.runner import connect
    spec=json.loads(connect(folder).execute("SELECT spec FROM jobs WHERE id=?",(args.job,)).fetchone()[0])
    controller=os.getppid()
    def watchdog():
        while True:
            time.sleep(5)
            if os.getppid()!=controller:os._exit(143)
    threading.Thread(target=watchdog,daemon=True).start()
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(c["max_allocated_hbm_gib"]*2**30/torch.cuda.get_device_properties(0).total_memory,0)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.manual_seed(c["seed"]);torch.set_num_threads(c["cpu_threads"])
        b=spec["block"]
        with torch.inference_mode():
            if b=="P0":out=p0(dest,c)
            elif b=="P1":out=capture(dest,c,spec["params"])
            elif b=="P3":out=transform(dest,folder,spec["params"])
            elif b=="P4":out=benchmark(dest,c,folder,spec["params"])
            else:out=aggregate(dest,folder,b)
    except Unsupported as e:out={"status":"UNSUPPORTED","detail":str(e)}
    except Exception as e:
        traceback.print_exc()
        out={"status":"OOM" if "out of memory" in str(e).lower() else "FAIL","detail":str(e)[:1000]}
    atomic(dest/"output.json",out)
    print(out["status"],out.get("detail",""),flush=True)
    return 0 if out["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
