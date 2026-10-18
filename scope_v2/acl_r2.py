"""Three preregistered additive diagnostics requested by the ACL reviewer.

Tensor loops never call an LLM. No historical sources/results are edited.
"""
from __future__ import annotations
import argparse,contextlib,gc,hashlib,inspect,json,os,sqlite3,subprocess,sys,time,traceback,zipfile
from pathlib import Path
from scope.runner import atomic,canonical,digest,gpu_gate,now
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/acl_r2_contract_v2.json'
STAGES=['quality_gate_v2','lowering_diagnosis_v2','operand_screen_v2']

def code_hash():
    names=['scope_v2/acl_r2.py','scope_v2/lowering_probe.py','configs/acl_r2_contract_v2.json',
           'scope_v2/persistent.py','scope_v2/packed_attention.py','scope_v2/resident.py','scope/engineering.py']
    return digest(b''.join(n.encode()+b'\0'+(ROOT/n).read_bytes() for n in names))

def thash(x):
    return digest(x.detach().contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes())

def independent_quant(x):
    """Direct dense math, deliberately no production encode/unpack calls."""
    import torch
    grouped=x.float().reshape(*x.shape[:-1],x.shape[-1]//32,32)
    scale=torch.maximum(grouped.abs().amax(-1),torch.tensor(1e-12,device=x.device))/7
    codes=torch.clamp(torch.round(grouped/scale[...,None]),-7,7).to(torch.int16)
    decoded=(codes.float()*scale[...,None]).reshape_as(x).to(x.dtype)
    return decoded,codes.reshape_as(x),scale

def independent_unpack(layer,key):
    import torch
    p=(layer.pk if key else layer.pv)[...,:layer.length,:]
    s=(layer.sk if key else layer.sv)[...,:layer.length,:]
    low=(p.to(torch.int32)%16)-8;high=torch.div(p.to(torch.int32),16,rounding_mode='floor')-8
    c=torch.empty((*p.shape[:-1],p.shape[-1]*2),device=p.device,dtype=torch.float32)
    c[...,0::2]=low;c[...,1::2]=high
    return (c.reshape(*c.shape[:-1],-1,32)*s[...,None]).reshape_as(c).to(layer.dtype)

def independent_cache(states):
    from transformers.cache_utils import Cache,DynamicLayer
    class Layer(DynamicLayer):
        def update(self,k,v,*a,**kw):
            return super().update(independent_quant(k)[0],independent_quant(v)[0],*a,**kw)
    layers=[]
    for k,v in states:
        layer=Layer();layer.keys=independent_quant(k)[0].cuda();layer.values=independent_quant(v)[0].cuda()
        layer.dtype=k.dtype;layer.device=layer.keys.device;layer.is_initialized=True;layers.append(layer)
    return Cache(layers=layers)

def cache_states(cache):
    from scope_v2.persistent import PackedLayer
    return [(independent_unpack(l,True).cpu(),independent_unpack(l,False).cpu()) if isinstance(l,PackedLayer)
            else (l.keys[...,:l.get_seq_length(),:].cpu(),l.values[...,:l.get_seq_length(),:].cpu()) for l in cache.layers]

def state_hash(states):return digest(canonical([[thash(k),thash(v)] for k,v in states]).encode())

def numerical(x,ref):
    import torch
    a=x.detach().double().cpu();b=ref.detach().double().cpu()
    assert torch.isfinite(a).all() and torch.isfinite(b).all(),'nonfinite tensor'
    norm=float(b.norm());diff=a-b
    return dict(relative_l2=float(diff.norm())/max(norm,1e-30),max_abs=float(diff.abs().max()),
                reference_norm=norm,near_zero_reference=norm<=1e-30,exact=bool(torch.equal(a,b)))

def logits_comparison(x,ref,target):
    import torch
    m=numerical(x,ref);a=x.double();b=ref.double();la=a.log_softmax(-1);lb=b.log_softmax(-1)
    m.update(kl_reference_to_output=float((lb.exp()*(lb-la)).sum(-1).mean()),top1_agreement=float((a.argmax(-1)==b.argmax(-1)).double().mean()),
             natural_next_token_nll_output=float(-la[...,target].mean()),natural_next_token_nll_reference=float(-lb[...,target].mean()))
    assert all(__import__('math').isfinite(v) for v in m.values() if isinstance(v,float))
    return m

def tf32_round(x):
    import torch
    x=x.float().contiguous();assert torch.isfinite(x).all()
    bits=x.view(torch.int32);rounded=((bits+4095+((bits>>13)&1))&(-8192)).view(torch.float32)
    assert torch.isfinite(rounded).all(),'TF32 emulation overflow'
    return rounded

def window(doc):
    tokens=doc['windows']['qwen1_5b']['tokens'];prefix=tokens[-521:-9];teacher=tokens[-9:]
    assert len(prefix)==512 and len(teacher)==9
    return prefix,teacher,digest(canonical(dict(prefix=prefix,teacher=teacher)).encode())

def prepare(manifest,out,repair_parent=None):
    c=json.loads(CONFIG.read_text());data=json.loads(manifest.read_text())
    assert digest(manifest.read_bytes())==c['input_manifest_sha256']
    m=next(m for m in data['models'] if m['key']=='qwen1_5b');assert m['id']==c['model'] and m['revision']==c['revision']
    inputs=[]
    for source in c['sources']:
        d=next(d for d in data['documents'] if d['document_id']==source);assert d['split']=='screening'
        _,_,h=window(d)
        previous=[]
        for p in (ROOT/'results/distributions_v2/12fbc139be256644/blocks').glob('*/summary.json'):
            s=json.loads(p.read_text())
            if s['spec']['model']=='qwen1_5b' and s['spec']['document_id']==source and s['spec']['length']==512:previous.append(s)
        assert len(previous)==1 and previous[0]['token_window_sha256']==h and all(previous[0]['native_controls_exact'])
        inputs.append(dict(document_id=source,domain=d['domain'],token_window_sha256=h))
    frozen=dict(contract=c,source_sha256=code_hash(),inputs=inputs)
    if repair_parent is not None:
        assert len(repair_parent)==16 and all(x in '0123456789abcdef' for x in repair_parent)
        old=ROOT/'results/quality_gate_v2'/repair_parent/'blocks/quality';assert verify(old)['status']=='CONTROL_FAIL'
        evidence=manifest.parent.parent/'acl_r2_repair1_evidence.json'
        proof=json.loads(evidence.read_text());assert proof['numeric_equal'] and not proof['raw_bytes_equal']
        frozen['repair']=dict(index=1,parent_run=repair_parent,parent_failure_manifest_sha256=digest((old/'manifest.json').read_bytes()),evidence_sha256=digest(evidence.read_bytes()),
            reason='Independent direct quantizer must materialize integer codes before dequantization; signed floating zero is not an INT4 code. Numerical endpoints, sources and thresholds unchanged.')
    if out.exists():assert json.loads(out.read_text())==frozen,'preregistration drift'
    else:atomic(out,frozen)
    print(json.dumps(dict(preregistered_sha256=digest(out.read_bytes()),run=digest(canonical(frozen).encode())[:16],inputs=inputs)))

def setup(manifest,prepared):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['TRITON_CACHE_DIR']=str(ROOT.parent/'triton_cache_continuation_v2')
    import torch,transformers,platform
    from transformers import AutoModelForCausalLM
    frozen=json.loads(prepared.read_text());c=frozen['contract'];data=json.loads(manifest.read_text())
    assert frozen['source_sha256']==code_hash() and digest(manifest.read_bytes())==c['input_manifest_sha256']
    gpu_gate(dict(reserve_hbm_gib=12,minimum_free_hbm_gib=32,reserve_disk_gib=24))
    torch.set_num_threads(4);torch.manual_seed(20261019);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(96*2**30/torch.cuda.get_device_properties(0).total_memory)
    model=AutoModelForCausalLM.from_pretrained(c['model'],revision=c['revision'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').eval().cuda()
    assert model.config.num_hidden_layers==28 and model.config.num_attention_heads==12
    docs=[]
    for x in frozen['inputs']:
        d=next(d for d in data['documents'] if d['document_id']==x['document_id']);assert window(d)[2]==x['token_window_sha256'];docs.append(d)
    runtime=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=False,bf16_reduction_flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    return model,docs,frozen,runtime

def prefill(model,prefix):
    import torch
    cache=model(input_ids=torch.tensor([prefix],device='cuda'),use_cache=True,logits_to_keep=1).past_key_values
    states=cache_states(cache);del cache;gc.collect();torch.cuda.empty_cache();return states

def decode_path(model,states,prefix,teacher,path,hook=None,steps=4):
    import torch
    from scope_v2.persistent import build_cache,execution
    cache=None
    if path=='independent_dense_quantized':cache=independent_cache(states)
    elif path!='original_full':cache=build_cache(states,1,'dense' if path=='native_cached' else 'chunked' if path=='int4_chunked' else 'global')
    outputs=[];hashes=[]
    ctx=execution(model,'chunked',validate_mask=True) if path=='int4_chunked' else contextlib.nullcontext()
    with ctx:
        for step in range(steps):
            if hook is not None:hook['step']=step
            ids=torch.tensor([prefix+teacher[:step+1]] if path=='original_full' else [[teacher[step]]],device='cuda')
            out=model(input_ids=ids,past_key_values=cache,use_cache=path!='original_full',logits_to_keep=1)
            outputs.append(out.logits[:,-1].cpu())
            if cache is not None:
                assert cache.get_seq_length()==512+step+1;hashes.append(state_hash(cache_states(cache)))
            del out
    del cache;gc.collect();torch.cuda.empty_cache();return outputs,hashes

def quality(model,docs,frozen,dest):
    import torch
    prefix,teacher,_=window(docs[0]);states=prefill(model,prefix)
    paths=['original_full','native_cached','independent_dense_quantized','int4_chunked','int4_global'];saved={};cache_hashes={};rows=[];replays=[]
    for replay in range(2):
        current={};ch={}
        for path in paths:current[path],ch[path]=decode_path(model,states,prefix,teacher,path)
        for path in paths:
            if replay:
                assert all(torch.equal(a,b) for a,b in zip(current[path],saved[path])),'logit replay drift'
                assert ch[path]==cache_hashes[path],'cache replay drift'
        for path in ['int4_chunked','int4_global']:assert ch[path]==ch['independent_dense_quantized'],'independent decoded-cache parity failure'
        for ref,path,kind in [('original_full','native_cached','baseline'),('native_cached','independent_dense_quantized','representation_cached'),('original_full','independent_dense_quantized','representation_total'),('independent_dense_quantized','int4_chunked','execution'),('independent_dense_quantized','int4_global','execution')]:
            for step in range(4):rows.append(dict(replay=replay,step=step,path=path,reference=ref,comparison=kind,**logits_comparison(current[path][step],current[ref][step],teacher[step+1])))
        replays.append(dict(replay=replay,logit_sha256={p:[thash(x) for x in current[p]] for p in paths},cache_sha256=ch))
        saved=current;cache_hashes=ch
    total=[x['relative_l2'] for x in rows if x['comparison']=='representation_total'];base=[x['relative_l2'] for x in rows if x['comparison']=='baseline'];execution=[x['relative_l2'] for x in rows if x['comparison']=='execution']
    assert max(execution)<=.025,'independent quantized execution gate failure'
    result=dict(status='REPRESENTATION_FIDELITY_FAIL' if max(total)>.025 else 'COLLECTED',baseline_status='BASELINE_DIVERGENCE' if max(base)>.025 else 'BASELINE_SCREEN_PASS',
        representation_max_relative_l2=max(total),baseline_max_relative_l2=max(base),execution_max_relative_l2=max(execution),independent_cache_parity_exact=True,replays_exact=True,
        independent_source_units=1,repeated_steps=4,replays=2,execution_gate=.025,representation_screen=.025,
        scope='Local logit fidelity screen and natural next-token NLL only. No corpus perplexity/task accuracy claim. Distances are not additive error components.')
    write_csv(dest/'comparisons.csv',rows);atomic(dest/'replay_controls.json',replays);return result

def write_csv(path,rows):
    import csv
    keys=sorted(set().union(*(set(r) for r in rows))) if rows else ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def native_capture(original,att,q,k,v,mask,args,kw):
    import torch
    saved=[];softmax=torch.nn.functional.softmax
    def observed(z,*a,**kws):saved.append(z.detach().clone());return softmax(z,*a,**kws)
    torch.nn.functional.softmax=observed
    try:out=original(att,q,k,v,mask,*args,**kw)
    finally:torch.nn.functional.softmax=softmax
    assert len(saved)==1 and out[1] is not None,'native Z/P capture'
    return out,saved[0],out[1]

def trace_path(model,states,prefix,teacher,mode,reference,rows,append_rows):
    import torch,types
    import transformers.models.qwen2.modeling_qwen2 as module
    from scope_v2.persistent import build_cache,Operand,PackedLayer
    from torch.nn.attention import sdpa_kernel,SDPBackend
    original=module.eager_attention_forward;layers=[m for m in model.modules() if type(m).__name__.endswith('Attention')]
    lookup={id(m):i for i,m in enumerate(layers)};state={'step':0};trace={};positions={};hooks=[]
    quant=mode in ['quant_reference','int4_triton'];cache=independent_cache(states) if mode=='quant_reference' else build_cache(states,1,'triton' if quant else 'dense_static')
    for index,layer in enumerate(cache.layers):
        old=layer.update
        def update(self,k,v,*a,_old=old,_index=index,**kw):
            before=int(self.get_seq_length());answer=_old(k,v,*a,**kw);after=int(self.get_seq_length())
            kk=independent_unpack(self,True) if isinstance(self,PackedLayer) else self.keys[...,:after,:]
            vv=independent_unpack(self,False) if isinstance(self,PackedLayer) else self.values[...,:after,:]
            expected_k=independent_quant(k)[0] if quant else k;expected_v=independent_quant(v)[0] if quant else v
            entry=dict(method=mode,step=state['step'],layer=_index,before=before,after=after,position_expected=512+state['step'],
                       append_k_exact=bool(torch.equal(kk[...,before:after,:],expected_k)),append_v_exact=bool(torch.equal(vv[...,before:after,:],expected_v)))
            assert before==entry['position_expected'] and after==before+1 and entry['append_k_exact'] and entry['append_v_exact'],'cache append parity'
            append_rows.append(entry);return answer
        layer.update=types.MethodType(update,layer)
    def pre(att,args,kw):
        bound=inspect.signature(att.forward).bind_partial(*args,**kw).arguments
        pe=bound.get('position_embeddings');pi=bound.get('position_ids');cp=bound.get('cache_position')
        positions[id(att)]=dict(position_embeddings_sha256=[thash(x) for x in pe] if isinstance(pe,tuple) else 'UNKNOWN',
                               position_ids_sha256=thash(pi) if isinstance(pi,torch.Tensor) else 'UNKNOWN',
                               cache_position_sha256=thash(cp) if isinstance(cp,torch.Tensor) else 'UNKNOWN')
    for att in layers:hooks.append(att.register_forward_pre_hook(pre,with_kwargs=True))
    def wrapped(att,q,k,v,mask,*args,**kw):
        index=lookup[id(att)];step=state['step'];packed=isinstance(k,Operand)
        if packed:
            dk=k.decode();dv=v.decode();assert torch.equal(dk,independent_unpack(k.layer,True)) and torch.equal(dv,independent_unpack(v.layer,False)),'packed decode independent parity'
        else:dk,dv=k,v
        assert q.shape[-2]==1 and (mask is None or bool((mask==0).all())),'unsupported decode mask'
        eager,z,p=native_capture(original,att,q,dk,dv,mask,args,kw)
        signature=dict(Q=thash(q),K=thash(dk),V=thash(dv),mask=thash(mask) if mask is not None else None,
                       **positions.get(id(att),{}),scale=float(kw.get('scaling',att.scaling)),length=dk.shape[-2])
        entry=dict(method=mode,step=step,layer=index,scale=signature['scale'],kv_length=signature['length'],mask_zero_or_none=True,
                   packed_decode_parity=True if packed else None,backend_choice='MATH' if mode=='static_sdpa_math' else 'AUTO' if mode=='static_sdpa' else mode,
                   internal_algorithm_id='UNKNOWN')
        if mode in ['native_reference','quant_reference']:
            y=eager[0];trace[(step,index)]=dict(signature=signature,Z=z.cpu(),P=p.cpu(),Y=y.cpu())
        elif mode.startswith('static_sdpa'):
            ctx=sdpa_kernel(SDPBackend.MATH) if mode=='static_sdpa_math' else contextlib.nullcontext()
            with ctx:
                y=torch.nn.functional.scaled_dot_product_attention(q,dk,dv,attn_mask=mask,dropout_p=0.,is_causal=False,scale=signature['scale'],enable_gqa=q.shape[1]!=dk.shape[1]).transpose(1,2).contiguous()
            entry.update(internal_ZP='UNKNOWN_FUSED_API',local_first_visible_operator='Y' if not torch.equal(y,eager[0]) else 'NO_VISIBLE_DIFFERENCE')
        else:
            from scope_v2.lowering_probe import observe
            debug,dz,dp,exact,actual=observe(q,k.layer,signature['scale']);y=actual.transpose(1,2).contiguous()
            entry['diagnostic_clone_Y_exact']=exact
            if exact:
                mz=numerical(dz,z);mp=numerical(dp,p);entry.update(z_relative_l2=mz['relative_l2'],z_exact=mz['exact'],p_relative_l2=mp['relative_l2'],p_exact=mp['exact'],
                    internal_ZP='OBSERVATIONALLY_CONTROLLED_CLONE_NOT_INSTRUCTION_IDENTITY',local_first_visible_operator='Z' if not mz['exact'] else 'P' if not mp['exact'] else 'Y' if not torch.equal(y,eager[0]) else 'NO_VISIBLE_DIFFERENCE')
            else:entry.update(internal_ZP='UNKNOWN_CLONE_CONTROL_FAILED',local_first_visible_operator='Y_BOUNDARY_ONLY')
            del debug,dz,dp,actual
        if reference is not None:
            ref=reference[(step,index)];entry.update(**{'input_'+name+'_exact':signature.get(name)==value for name,value in ref['signature'].items()},
                same_operand_y_relative_l2=numerical(y,eager[0])['relative_l2'],same_operand_y_exact=bool(torch.equal(y,eager[0])),
                trajectory_y_relative_l2=numerical(y,ref['Y'])['relative_l2'],trajectory_y_exact=bool(torch.equal(y.cpu(),ref['Y'])),
                native_same_operand_Z_vs_reference_relative_l2=numerical(z,ref['Z'])['relative_l2'],native_same_operand_P_vs_reference_relative_l2=numerical(p,ref['P'])['relative_l2'])
            rows.append(entry)
        return (y,eager[1]) if mode in ['native_reference','quant_reference'] else (y,None)
    module.eager_attention_forward=wrapped;logits=[];cache_digests=[]
    try:
        for step in range(4):
            state['step']=step;out=model(input_ids=torch.tensor([[teacher[step]]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
            logits.append(out.logits[:,-1].cpu());cache_digests.append(state_hash(cache_states(cache)));del out
    finally:
        module.eager_attention_forward=original
        for h in hooks:h.remove()
    del cache;gc.collect();torch.cuda.empty_cache();return trace,logits,cache_digests

def diagnosis(model,docs,frozen,dest):
    import torch
    prefix,teacher,_=window(docs[0]);states=prefill(model,prefix);rows=[];append_rows=[];steps=[];runs={};controls=[]
    run_id=digest(canonical(frozen).encode())[:16]
    quality_controls=json.loads((ROOT/'results/quality_gate_v2'/run_id/'blocks/quality/replay_controls.json').read_text())
    for replay in range(2):
        refs={}
        for mode,qpath in [('native_reference','native_cached'),('quant_reference','independent_dense_quantized')]:
            trace,logits,ch=trace_path(model,states,prefix,teacher,mode,None,rows,append_rows)
            assert [thash(x) for x in logits]==quality_controls[replay]['logit_sha256'][qpath],'native trace observer changed quality reference'
            assert ch==quality_controls[replay]['cache_sha256'][qpath]
            refs[mode]=(trace,logits);controls.append(dict(replay=replay,method=mode,quality_reference_logit_and_cache_exact=True))
        for mode in ['static_sdpa','static_sdpa_math','int4_triton']:
            trace,logits,ch=trace_path(model,states,prefix,teacher,mode,refs['quant_reference' if mode=='int4_triton' else 'native_reference'][0],rows,append_rows)
            if replay:assert [thash(x) for x in logits]==runs[mode]['logits'] and ch==runs[mode]['cache'],'candidate repeat drift'
            runs[mode]=dict(logits=[thash(x) for x in logits],cache=ch)
            ref=refs['quant_reference' if mode=='int4_triton' else 'native_reference'][1]
            for step in range(4):steps.append(dict(replay=replay,method=mode,step=step,**logits_comparison(logits[step],ref[step],teacher[step+1])))
        for x in rows:
            if 'replay' not in x:x['replay']=replay
        for x in append_rows:
            if 'replay' not in x:x['replay']=replay
    first=[]
    for replay in range(2):
        for method in ['static_sdpa','static_sdpa_math','int4_triton']:
            selected=[x for x in rows if x['method']==method and x['replay']==replay and not x['same_operand_y_exact']]
            first.append(dict(replay=replay,method=method,first=min(selected,key=lambda x:(x['step'],x['layer'])) if selected else None))
    write_csv(dest/'first_divergence.csv',rows);write_csv(dest/'stepwise_errors.csv',steps)
    atomic(dest/'operand_parity.json',dict(append=append_rows,reference_controls=controls,candidate_replay_hashes=runs))
    return dict(status='COLLECTED',independent_source_units=1,replays=2,steps=4,first_divergences=first,
        native_observer_controls_exact=True,append_parity_exact=True,replay_exact=True,
        gate_d={m:('GATE_D_FAIL' if max(x['relative_l2'] for x in steps if x['method']==m)>.025 else 'PASS') for m in runs},
        scope='Same-operand local lowering errors versus recurrent trajectory divergence. SDPA Z/P internals UNKNOWN. Triton Z/P diagnostic clone accepted only where Y is bitwise production-equal; compiler internal identity and cuBLAS algorithm IDs UNKNOWN. No timing winner or automatic repair.')

def operand_rows(p,v,common):
    import torch
    # CPU FP32 contractions isolate operand-format arithmetic from actual MMA.
    decoded,c,sv=independent_quant(v.detach());p=p.detach().float().cpu();decoded=decoded.cpu();c=c.cpu();sv=sv.cpu()
    w=p[:,None]*sv;codes=c.reshape(len(p),-1,32).float()
    y_ref=p@decoded.float();factorized=(w[...,None]*codes).sum(0).flatten()
    h=tf32_round(w);lo=tf32_round(w-h)
    candidates=[('fp32',w,None),('tf32_rne_single',h,None),('tf32_rne_hi_plus_lo',h,lo)]
    cancellation_den=(w[...,None]*codes).abs().sum(0).flatten();valid=cancellation_den>1e-30
    cancellation=factorized[valid].abs()/cancellation_den[valid]
    quantiles=torch.quantile(cancellation,torch.tensor([0.,.01,.05,.5,.95,1.])).tolist() if len(cancellation) else None
    result=[]
    for name,a,b in candidates:
        represented=a if b is None else a+b
        y=(a[...,None]*codes).sum(0).flatten()
        if b is not None:y=y+(b[...,None]*codes).sum(0).flatten()
        mw=numerical(represented,w);my=numerical(y,y_ref);mf=numerical(y,factorized)
        result.append(dict(common,format=name,w_relative_l2=mw['relative_l2'],w_max_abs=mw['max_abs'],w_near_zero=mw['near_zero_reference'],
            pv_relative_l2=my['relative_l2'],pv_max_abs=my['max_abs'],pv_reference_norm=my['reference_norm'],pv_near_zero=my['near_zero_reference'],
            pv_vs_factorized_fp32_relative_l2=mf['relative_l2'],bf16_decode_rounding_gap_relative_l2=numerical(factorized,y_ref)['relative_l2'],
            zero_cancellation_denominator_channels=int((~valid).sum()),cancellation_quantile_levels='0,.01,.05,.5,.95,1',
            cancellation_quantiles=json.dumps(quantiles),contraction='sum_t W[t,g]*C[t,g,d]; TF32 parts separately accumulated in FP32',
            hardware_execution='FORMAT_EMULATION_NOT_TF32_MMA'))
    return result

def operand(model,docs,frozen,dest):
    import torch
    import transformers.models.qwen2.modeling_qwen2 as module
    original=module.eager_attention_forward;layers=[m for m in model.modules() if type(m).__name__.endswith('Attention')];lookup={id(m):i for i,m in enumerate(layers)}
    rows=[];controls=[];old_samples={}
    for f in (ROOT/'results/distributions_v2/12fbc139be256644/blocks').glob('*/sketch-*.jsonl'):
        # Scope to selected Qwen cells using their sealed summary first.
        s=json.loads((f.parent/'summary.json').read_text())
        if s['spec']['model']!='qwen1_5b' or s['spec']['length']!=512:continue
        for line in f.read_text().splitlines():
            x=json.loads(line)
            if x['query_head'] in [0,6,11]:old_samples[(x['document_id'],x['step'],x['layer'],x['query_head'])]=x
    for doc in docs:
        prefix,teacher,_=window(doc);states=prefill(model,prefix);replay_hashes=[]
        for replay in range(2):
            current=[];state={'step':-1}
            def wrapped(att,q,k,v,mask,*args,**kw):
                out=original(att,q,k,v,mask,*args,**kw);i=lookup[id(att)];step=state['step']
                if step in [0,7] and i in [0,14,27]:
                    assert out[1] is not None and q.shape[0]==1 and q.shape[-2]==1
                    for head in [0,6,11]:
                        key=(doc['document_id'],step,i,head);kv=head//(q.shape[1]//k.shape[1])
                        p=out[1][0,head,0].detach();vv=v[0,kv].detach()
                        from scope_v2.distributions import joints
                        _,cv,sv=independent_quant(vv)
                        old=old_samples[key];observed=joints(p.float().cpu(),sv.cpu(),cv.cpu())['reconstructed_p_sv_c_cancellation_ratio']
                        previous=old['joint']['reconstructed_p_sv_c_cancellation_ratio']
                        assert (observed is None and previous is None) or abs(observed-previous)<=1e-6,'historical FP32 joint reference changed'
                        current.extend(operand_rows(p,vv,dict(document_id=doc['document_id'],domain=doc['domain'],step=step,layer=i,head=head,replay=replay)))
                return out
            module.eager_attention_forward=wrapped
            try:outputs,cache_hashes=decode_path(model,states,prefix,teacher,'native_cached',hook=state,steps=8)
            finally:module.eager_attention_forward=original
            hashes=dict(logits=[thash(x) for x in outputs],cache=cache_hashes)
            if replay:
                assert hashes==replay_hashes[0],'native operand-observer replay drift'
                a=[{k:v for k,v in x.items() if k!='replay'} for x in current];b=[{k:v for k,v in x.items() if k!='replay'} for x in rows if x['document_id']==doc['document_id']]
                assert a==b,'operand metric replay drift'
            else:rows.extend(current)
            replay_hashes.append(hashes)
        # Independent unobserved native replay provides observer control.
        unobserved,ch=decode_path(model,states,prefix,teacher,'native_cached',steps=8)
        assert dict(logits=[thash(x) for x in unobserved],cache=ch)==replay_hashes[0],'operand observer altered native execution'
        controls.append(dict(document_id=doc['document_id'],replays_exact=True,unobserved_logits_cache_exact=True,historical_fp32_joint_tolerance=1e-6))
    assert len(rows)==2*2*3*3*3
    write_csv(dest/'operand_residual.csv',[{k:v for k,v in x.items() if not k.startswith('pv_')} for x in rows])
    write_csv(dest/'pv_residual.csv',rows);atomic(dest/'observer_controls.json',controls)
    return dict(status='COLLECTED',source_units=2,selected_head_layer_step_units=36,format_rows=len(rows),replays=2,observer_controls_exact=True,
        residuals={name:dict(max_w_relative_l2=max(x['w_relative_l2'] for x in rows if x['format']==name),max_pv_relative_l2=max(x['pv_relative_l2'] for x in rows if x['format']==name),max_pv_vs_factorized_fp32_relative_l2=max(x['pv_vs_factorized_fp32_relative_l2'] for x in rows if x['format']==name)) for name in ['fp32','tf32_rne_single','tf32_rne_hi_plus_lo']},
        scope='Two screening sources only. Heads/layers/steps are repeated observations. CPU FP32 format emulation with native P and actual INT4-dequantized V. Dequantization rounding gap retained; not MMA throughput or model-quality evidence.')

def seal(dest,result,frozen):
    dest.mkdir(parents=True,exist_ok=True);result.update(source_sha256=frozen['source_sha256'],preregistration_sha256=digest(canonical(frozen).encode()),publication='PENDING')
    atomic(dest/'summary.json',result);atomic(dest/'preregistered_contract.json',frozen)
    (dest/'report.md').write_text('# ACL round2 additive diagnostic\n\n'+json.dumps(result,indent=2)+'\n\nRaw logits, token IDs and activation tensors were not saved to this block. Historical sources, result identities, original failures and .025 Gate D are preserved.\n')
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in dest.iterdir() if p.is_file() and p.name not in ['manifest.json','SHA256SUMS.txt']}
    atomic(dest/'manifest.json',dict(files=files,source_sha256=frozen['source_sha256'],sealed=now()));(dest/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in sorted(files.items())))

def verify(dest):
    meta=json.loads((dest/'manifest.json').read_text())
    for name,v in meta['files'].items():
        assert Path(name).name==name;data=(dest/name).read_bytes();assert len(data)==v['bytes'] and digest(data)==v['sha256']
    return json.loads((dest/'summary.json').read_text())

def worker(stage,manifest,prepared,dest):
    frozen=json.loads(prepared.read_text());result=dict(status='RUNNING',stage=stage)
    dest.mkdir(parents=True,exist_ok=True)
    try:
        model,docs,frozen,runtime=setup(manifest,prepared)
        import torch
        with torch.inference_mode():result.update({'quality_gate_v2':quality,'lowering_diagnosis_v2':diagnosis,'operand_screen_v2':operand}[stage](model,docs,frozen,dest))
        result['runtime']=runtime
    except Exception as exc:
        result.update(status='CONTROL_FAIL' if isinstance(exc,AssertionError) else 'FAIL',error_type=type(exc).__name__)
        (dest.parent.parent/(stage+'.private.log')).write_text(traceback.format_exc())
    seal(dest,result,frozen)
    if result['status'] in ['FAIL','CONTROL_FAIL']:raise RuntimeError('sealed additive diagnostic failure; bounded repair required')

def run(manifest,prepared):
    import fcntl
    frozen=json.loads(prepared.read_text());assert frozen['source_sha256']==code_hash();identity=digest(canonical(frozen).encode())[:16]
    lock=(ROOT.parent/'engineering_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        for stage,label in zip(STAGES,['quality','first-divergence','operand']):
            folder=ROOT/'results'/stage/identity;folder.mkdir(parents=True,exist_ok=True);dest=folder/'blocks'/label
            if (dest/'manifest.json').exists():
                previous=verify(dest)
                if previous['status'] in ['FAIL','CONTROL_FAIL','TIMEOUT']:break
                continue
            atomic(folder/'campaign.json',dict(run=identity,source_sha256=frozen['source_sha256'],prepared_sha256=digest(prepared.read_bytes()),stage=stage,budget_seconds=frozen['contract']['limits'][stage]))
            atomic(folder/'status.json',dict(status='RUNNING',updated=now()))
            with (folder/'worker.private.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','scope_v2.acl_r2','worker','--stage',stage,'--manifest',str(manifest.resolve()),'--prepared',str(prepared.resolve()),'--dest',str(dest.resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:rc=proc.wait(timeout=frozen['contract']['limits'][stage])
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    rc=124
            if not (dest/'manifest.json').exists():seal(dest,dict(status='TIMEOUT' if rc==124 else 'FAIL',stage=stage,reason='own bounded worker terminated; no accepted result'),frozen)
            result=verify(dest);db=sqlite3.connect(folder/'registry.sqlite');db.execute('CREATE TABLE IF NOT EXISTS publications(id PRIMARY KEY,status,manifest_sha,updated)')
            db.execute('INSERT OR REPLACE INTO publications VALUES(?,?,?,?)',(label,'PENDING',digest((dest/'manifest.json').read_bytes()),now()));db.commit();db.close()
            atomic(folder/'status.json',dict(status='STOPPED_WITH_FAILURE' if rc else 'COMPLETE',result=result['status'],updated=now()))
            with zipfile.ZipFile(ROOT.parent/f'SCOPE_{stage}_{identity}.zip','w',zipfile.ZIP_DEFLATED) as z:
                for p in dest.iterdir():z.write(p,p.relative_to(ROOT).as_posix())
            if rc:break
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','worker']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--stage',choices=STAGES);p.add_argument('--dest',type=Path);p.add_argument('--repair-parent-run');a=p.parse_args()
    if a.mode=='prepare':prepare(a.manifest,a.prepared,a.repair_parent_run)
    elif a.mode=='run':run(a.manifest,a.prepared)
    else:worker(a.stage,a.manifest,a.prepared,a.dest)
