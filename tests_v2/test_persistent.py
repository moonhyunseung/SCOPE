import copy
import torch
from test_diagnose import tiny
from scope.engineering import quantize
from scope_v2.persistent import encode,build_cache,payload,execution

def test_codes_match_frozen_quantizer_and_actual_storage():
    torch.manual_seed(31);x=torch.randn(1,2,7,8)
    for bits in [4,8]:
        p,s=encode(x,bits,4);ref,scales,_=quantize(x,bits,4)
        assert torch.equal(p,ref) and torch.equal(s,scales)
    cache=build_cache([(x,x)],3,'chunked',group=4,device='cpu')
    info=payload(cache)
    assert info['dense_persistent_fields']==0 and info['unique_storages']==4
    assert info['storage_bytes']==sum(t.numel()*t.element_size() for t in [cache.layers[0].pk,cache.layers[0].pv,cache.layers[0].sk,cache.layers[0].sv])
    assert cache.layers[0].pk.stride(0)>0 and cache.get_seq_length()==7

def test_actual_model_persistent_cache_equals_dense_quant_reference():
    model=tiny();ids=torch.tensor([[1,2,3,4,5]])
    with torch.inference_mode():
        pre=model(ids[:,:-1],use_cache=True).past_key_values
        cpu=[(l.keys.clone(),l.values.clone()) for l in pre.layers]
        results=[]
        for kind in ['dense_quant_reference','global','chunked']:
            cache=build_cache(cpu,2,kind,group=4,device='cpu')
            with execution(model,'chunked',chunk=1):
                out=model(ids[:,-1:].repeat(2,1),past_key_values=cache,use_cache=True).logits
            results.append(out)
            assert cache.get_seq_length()==5
            cache.crop(-1);assert cache.get_seq_length()==4
        assert torch.equal(results[0],results[1])
        torch.testing.assert_close(results[0],results[2],rtol=1e-6,atol=1e-6)
