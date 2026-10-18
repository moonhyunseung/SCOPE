import json
import pytest
import torch
from scope.engineering import metric, accepted, logits_metric, quantize, unpack, write_row

@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('group',[32,64,128])
def test_packed_identity_and_reference(bits,group):
    torch.manual_seed(83)
    x=torch.randn(2,3,7,128,dtype=torch.bfloat16)
    p,s,d=quantize(x,bits,group)
    assert torch.equal(unpack(p,s,bits,group,torch.float32),d)
    assert torch.equal(unpack(p,s,bits,group,torch.bfloat16),d.to(torch.bfloat16))
    assert accepted(metric(unpack(p,s,bits,group,torch.bfloat16),d.to(torch.bfloat16)))
    assert p.untyped_storage().nbytes()+s.untyped_storage().nbytes()<x.untyped_storage().nbytes()

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_cannot_pass(bad):
    m=metric(torch.tensor([bad]),torch.ones(1))
    assert not accepted(m)
    assert m['relative_l2'] is None
    json.dumps(m,allow_nan=False)

def test_gate_boundary_and_actual_difference():
    assert accepted({'finite':True,'relative_l2':.025})
    assert not accepted({'finite':True,'relative_l2':.02501})
    assert metric(torch.tensor([4.,0.]),torch.tensor([2.,0.]))['relative_l2']==1

def test_logits_identical():
    a=torch.tensor([[1.,2.,-3.]])
    m=logits_metric(a,a)
    assert m['relative_l2']==0 and m['top1_agreement']==1 and m['kl_reference_to_output']==0

def test_json_lines_are_independent(tmp_path):
    p=tmp_path/'rows.jsonl'
    write_row(p,{'a':1});write_row(p,{'a':2})
    assert [json.loads(s)['a'] for s in p.read_text().splitlines()]==[1,2]
