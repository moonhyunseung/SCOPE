import json
from pathlib import Path
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from scope_v2.diagnose import Trace, compare_traces, projection_probe, seal, verify

def tiny():
    torch.manual_seed(17)
    c=Qwen2Config(vocab_size=96,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,head_dim=8)
    c._attn_implementation='eager'
    return Qwen2ForCausalLM(c).eval()

def test_native_instrumentation_is_exact_and_restored():
    import transformers.models.qwen2.modeling_qwen2 as m
    model=tiny();ids=torch.tensor([[1,2,3,4,5]])
    original=m.eager_attention_forward;softmax=torch.nn.functional.softmax
    with torch.inference_mode():
        ref=model(ids,use_cache=False).logits
        with Trace(model,probe_rows=5) as trace: got=model(ids,use_cache=False).logits
    assert torch.equal(ref,got)
    assert m.eager_attention_forward is original
    assert torch.nn.functional.softmax is softmax
    assert trace.values['L00.scores'].shape==(1,4,5)
    assert trace.values['L00.cache_k'].shape==(1,2,5,8)
    assert len(trace.probes)==3

def test_matched_cached_full_structure():
    model=tiny();ids=torch.tensor([[1,2,3,4,5]])
    with torch.inference_mode():
        past=model(ids[:,:-1],use_cache=True).past_key_values
        with Trace(model) as a: model(ids[:,-1:],past_key_values=past,use_cache=True)
        with Trace(model) as b: model(ids,use_cache=False)
    rows=compare_traces(a,b,'test','float32',0,'cached_full')
    structure=[r for r in rows if r['stage'] in ['position_ids','rope_cos','rope_sin'] or r['stage'].endswith(('.mask','.scaling'))]
    assert all(r['exact'] for r in structure)
    assert rows[-1]['stage']=='logits'

def test_projection_control_has_identical_input_and_current_weights():
    m=torch.nn.Linear(32,64);x=torch.randn(1,5,32)
    with torch.inference_mode():r=projection_probe(m,x,17)
    assert r['rows']==17 and r['input'].shape==(1,32)
    assert torch.equal(r['one'],m(x[:,-1:]).reshape(1,64))
    assert r['one_vs_same_weight_fp32']['max_abs']==0

def test_restores_on_exception():
    import transformers.models.qwen2.modeling_qwen2 as m
    original=m.eager_attention_forward
    with pytest.raises(ValueError):
        with Trace(tiny()):raise ValueError('test')
    assert m.eager_attention_forward is original

def test_seal_tamper_is_rejected(tmp_path):
    seal(tmp_path,dict(status='FAIL',error='test'))
    assert verify(tmp_path)['status']=='FAIL'
    (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(RuntimeError,match='HASH_MISMATCH'):verify(tmp_path)

def test_aggregate_repeated_identifier_columns_and_denominators(tmp_path):
    import sqlite3
    from scope_v2.diagnose import aggregate,csv_write
    dest=tmp_path/'id'/'attempt';dest.mkdir(parents=True)
    csv_write(dest/'stages.csv',[])
    r=dict(status='COLLECTED',case_id='id',document_id='doc',dtype='bfloat16',stage_rows=0,
        dense_fidelity='FAIL',controls_exact=True,append_ok=True,structural_exact=True,control_checks=9,
        projection_shape_sensitivity_demonstrated=True,unresolved_exhaustive_cause=True,
        steps=[dict(case_id='id',dtype='bfloat16',step=i) for i in range(8)])
    seal(dest,r)
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE jobs(id,status,attempt)')
    db.execute('INSERT INTO jobs VALUES(?,?,?)',('id','COLLECTED','attempt'))
    db.execute('CREATE TABLE publications(id PRIMARY KEY,status,manifest_sha,updated)')
    s=aggregate(tmp_path,db,dict(cases=[{}]),False)
    import csv
    assert s['collected']==1 and s['counts']['bfloat16_FAIL']==1
    assert len(list(csv.DictReader((tmp_path/'analysis/metrics.csv').open())))==8
