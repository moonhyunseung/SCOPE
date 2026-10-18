import copy,importlib
import pytest
import torch
from scope.ablation import interaction,override,cache_equal,cache_hash

def tiny():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(111)
    c=Qwen2Config(vocab_size=41,hidden_size=64,intermediate_size=128,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
    c._attn_implementation='eager'
    return Qwen2ForCausalLM(c).eval()

def test_factorial_additive_has_zero_interaction():
    y=torch.tensor([[2.,4.,8.]])
    _,r=interaction(y,y+1,y+2,y+3)
    assert r['relative_l2']==0 and r['max_abs']==0

def test_factorial_nonadditive_vector_is_measured_directly():
    y=torch.tensor([[2.,0.]])
    _,r=interaction(y,y,y,y+torch.tensor([[0.,1.]]))
    assert r['relative_l2']==.5 and r['max_abs']==1

def test_native_cache_snapshot_is_immutable_and_fork_independent():
    model=tiny()
    with torch.inference_mode():
        cache=model(input_ids=torch.tensor([[1,2,3]]),use_cache=True).past_key_values
        old=cache_hash(cache);fork=copy.deepcopy(cache)
        assert cache_equal(cache,fork)
        model(input_ids=torch.tensor([[4]]),past_key_values=fork,use_cache=True)
        assert cache_hash(cache)==old
        assert not cache_equal(cache,fork)

@pytest.mark.parametrize('engine',['original','eager'])
def test_all_layer_noop_matches_original_cached_logits_exactly(engine):
    model=tiny()
    with torch.inference_mode():
        cache=model(input_ids=torch.tensor([[1,2,3]]),use_cache=True).past_key_values
        x=torch.tensor([[4]])
        ref=model(input_ids=x,past_key_values=copy.deepcopy(cache),use_cache=True).logits
        with override(model,'qwen2',dict(name='noop',engine=engine,K=False,V=False)) as s:
            out=model(input_ids=x,past_key_values=copy.deepcopy(cache),use_cache=True).logits
        assert s['calls']==3
        assert torch.equal(ref,out)

def test_hook_restores_original_even_on_exception():
    model=tiny();module=importlib.import_module('transformers.models.qwen2.modeling_qwen2');before=module.eager_attention_forward
    with pytest.raises(ValueError):
        with override(model,'qwen2',dict(name='noop',engine='original',K=False,V=False)):raise ValueError('test')
    assert module.eager_attention_forward is before
