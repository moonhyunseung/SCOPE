import torch
from scope_v2.distributions import marginal,joints,observe

def test_distribution_preserves_zero_sign_and_pairing():
    r=marginal(torch.tensor([0.,-1.,1.,2.]))
    assert r['zero_fraction']==.25 and r['negative_fraction']==.25
    assert sum(r['frexp_exponent_counts'].values())==3
    p=torch.tensor([.75,.25]);s=torch.tensor([[2.],[4.]])
    c=torch.ones(2,32);c[1]=-1
    j=joints(p,s,c)
    assert abs(j['w']['mean']-1.25)<1e-7
    assert abs(j['reconstructed_p_sv_c_cancellation_ratio']-.2)<1e-7
    assert sum(map(sum,j['p_vs_s_log2_histogram']))==2

def test_native_observation_preserves_actual_logits_and_cache():
    import copy
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from scope.population_collect import cache_equal
    torch.manual_seed(3)
    config=Qwen2Config(vocab_size=64,hidden_size=64,intermediate_size=96,num_hidden_layers=3,num_attention_heads=2,num_key_value_heads=1)
    config._attn_implementation='eager';model=Qwen2ForCausalLM(config).eval()
    with torch.inference_mode():
        prefix=model(torch.tensor([[1,2,3]]),use_cache=True).past_key_values
        a=model(torch.tensor([[4]]),past_key_values=copy.deepcopy(prefix),use_cache=True)
        records=[]
        with observe(model,dict(document_id='test',domain='test'),0,records) as stats:
            b=model(torch.tensor([[4]]),past_key_values=copy.deepcopy(prefix),use_cache=True)
    assert torch.equal(a.logits,b.logits) and cache_equal(a.past_key_values,b.past_key_values)
    assert stats==dict(calls=3,selected_calls=3) and len(records)==6
    assert all(r['native']['P']['nonfinite']==0 for r in records)
