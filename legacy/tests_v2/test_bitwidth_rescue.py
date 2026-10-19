import torch
from scope_v2.bitwidth_rescue import independent_quant8, checked_value, mixed_cache, byte_equal, decision, ROLES


def test_int8_signed_codes_ties_even_zero_and_oracle_bytes():
    x=torch.zeros(1,1,1,32,dtype=torch.bfloat16)
    x[...,:6]=torch.tensor([.5,1.5,-.5,-1.5,127.,-0.0],dtype=torch.bfloat16)
    y,c,s=independent_quant8(x)
    assert c.dtype==torch.int8 and s.dtype==torch.float32
    assert c.flatten()[:6].tolist()==[0,2,0,-2,127,0]
    assert byte_equal(y,checked_value(x,True)) and not torch.signbit(y[...,5]).any()
    assert torch.signbit(checked_value(x,False)[...,5]).all()
    z=independent_quant8(torch.zeros_like(x))
    assert torch.isfinite(z[0]).all() and (z[1]==0).all() and (z[2]>0).all()


def test_int8_mixed_cache_initialization_prefix_and_four_appends():
    k=torch.linspace(-3.5,5.7,512*128).reshape(1,1,512,128).bfloat16()
    v=(k.flip(-1)*.37).bfloat16()
    for role,flags in ROLES.items():
        controls=[];layer=mixed_cache([(k,v)],role,controls,device='cpu').layers[0]
        assert byte_equal(layer.keys,checked_value(k,flags[0]))
        assert byte_equal(layer.values,checked_value(v,flags[1]))
        for step in range(4):
            nk,nv=k[...,step:step+1,:],v[...,step:step+1,:]
            layer.update(nk,nv)
            assert byte_equal(layer.keys[...,-1:,:],checked_value(nk,flags[0]))
            assert byte_equal(layer.values[...,-1:,:],checked_value(nv,flags[1]))
        assert layer.get_seq_length()==516 and len(controls)==4
        assert byte_equal(layer.keys[...,:512,:],checked_value(k,flags[0]))


def test_two_references_and_unresolved_baseline_have_distinct_states():
    assert decision(.025,.025,.025)['local_candidate_status']=='LOCAL_SCREEN_PASS'
    assert decision(.01,.01,.03)['local_candidate_status']=='BASELINE_DIVERGENCE_UNRESOLVED'
    assert decision(.01,.03,.03)['local_candidate_status']=='FULL_REFERENCE_FAIL'
    assert decision(.03,.01,.02)['local_candidate_status']=='REPRESENTATION_CACHED_FAIL'
