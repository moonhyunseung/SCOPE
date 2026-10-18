import torch
from scope_v2.mechanism import InputTrace,modules,matmul_probe
from test_diagnose import tiny

def test_mechanism_inputs_and_linear_proxy_replay_native():
    model=tiny();ids=torch.tensor([[1,2,3]])
    with torch.inference_mode():
        past=model(ids[:,:-1],use_cache=True).past_key_values
        with InputTrace(model) as trace:model(ids[:,-1:],past_key_values=past,use_cache=True)
        for name,module in modules(model).items():
            y=module(trace.inputs[name]).select(-2,0)
            assert torch.equal(y.cpu(),trace.values['L00.'+name])

def test_matmul_fixed_inputs_shapes():
    a=torch.randn(1,4,1,8);b=torch.randn(1,4,8,13)
    r,y=matmul_probe(a,b,7)
    assert y.shape==(1,4,1,13) and r['rows']==7
    assert torch.equal(y,torch.matmul(a,b))
    assert r['one_vs_same_weight_fp32']['max_abs']==0
