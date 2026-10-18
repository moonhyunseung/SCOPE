import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from scope.experiments import pack,restore,attention
from scope.runner import jobs_for,connect,atomic,verify,digest

@pytest.mark.parametrize("group",[16,32])
def test_pack_roundtrip_for_realistic_ranges(group):
    torch.manual_seed(311)
    x=torch.randn(2,3,19,64)*torch.linspace(.001,100,64)
    x[0,0,0]=0
    packed,scales,decoded=pack(x,group)
    assert torch.equal(restore(packed,scales,group,torch.float32),decoded)
    assert packed.untyped_storage().nbytes()+scales.untyped_storage().nbytes()<x.untyped_storage().nbytes()

@pytest.mark.parametrize("boolean",[False,True])
def test_mask_and_gqa_replay(boolean):
    torch.manual_seed(20)
    q=torch.randn(1,6,1,32);k=torch.randn(1,2,17,32);v=torch.randn_like(k)
    mask=torch.ones(1,1,1,17,dtype=torch.bool);mask[...,3:6]=False
    actual_mask=mask if boolean else torch.where(mask,0.,float("-inf"))
    got=attention(q,k,v,actual_mask,.17)
    z=(q@k.repeat_interleave(3,1).transpose(-2,-1))*.17
    z=z.masked_fill(~mask,float("-inf"))
    expected=z.softmax(-1)@v.repeat_interleave(3,1)
    torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-6)

def test_registry_transactions_and_failed_rows_survive(tmp_path):
    db=connect(tmp_path)
    db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)",("test","{}","OOM","PENDING",1,"now"))
    db.commit();db.close()
    db=connect(tmp_path)
    assert db.execute("SELECT status,publication FROM jobs").fetchone()==("OOM","PENDING")

def test_plan_dependency_closure_and_no_condition_removed():
    config={"models":[{"id":"one","revision":"a"}],"lengths":[512,2048],"categories":["code","prose"]}
    first=jobs_for(config)
    assert first==jobs_for(config)
    assert sum(x["block"]=="P1" for x in first)==4
    names={x["id"] for x in first}
    assert all(set(x["deps"])<=names for x in first)
    assert not any(x["id"] in x["deps"] for x in first)

def test_corruption_is_detected(tmp_path):
    for name in ["manifest.json","summary.json"]:
        atomic(tmp_path/name,{})
    (tmp_path/"metrics.csv").write_text("id,status\na,PASS\n")
    (tmp_path/"report.md").write_text("example\n")
    (tmp_path/"SHA256SUMS.txt").write_text("".join(digest(p.read_bytes())+"  "+p.name+"\n" for p in tmp_path.iterdir() if p.name!="SHA256SUMS.txt"))
    verify(tmp_path)
    (tmp_path/"metrics.csv").write_text("corrupted")
    with pytest.raises(RuntimeError,match="HASH_MISMATCH"):verify(tmp_path)

@pytest.mark.parametrize("family",["qwen2","gpt_neox","opt"])
def test_same_dtype_replay_matches_upstream_model_function(family):
    import importlib
    from types import SimpleNamespace
    from scope.experiments import native_eager_reference
    upstream=importlib.import_module("transformers.models."+family+".modeling_"+family).eager_attention_forward
    hq,hkv=(6,2) if family=="qwen2" else (4,4)
    torch.manual_seed(37)
    q=(torch.randn(1,hq,1,32)*15).to(torch.bfloat16)
    k=(torch.randn(1,hkv,19,32)*15).to(torch.bfloat16)
    v=torch.randn(1,hkv,19,32).to(torch.bfloat16)
    mask=torch.zeros(1,1,1,19,dtype=torch.bfloat16);mask[...,2:4]=float("-inf")
    module=SimpleNamespace(num_key_value_groups=hq//hkv,training=False)
    actual=upstream(module,q,k,v,mask,scaling=.17)
    z,prob,y=native_eager_reference(q,k,v,mask,.17)
    assert torch.equal(y,actual[0].transpose(1,2))
    assert torch.equal(prob,actual[1])
