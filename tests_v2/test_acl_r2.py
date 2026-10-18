import torch
from scope_v2.acl_r2 import independent_quant,tf32_round,operand_rows
from scope.engineering import quantize,unpack

def test_independent_dense_definition_matches_signed_packing_and_zero_groups():
    x=torch.cat([torch.zeros(32),torch.linspace(-3,3,96)]).reshape(1,1,1,128).bfloat16()
    direct,c,s=independent_quant(x);packed,scale,_=quantize(x,4,32)
    assert torch.equal(direct,unpack(packed,scale,4,32,x.dtype))
    assert c.min()>=-7 and c.max()<=7 and torch.isfinite(s).all()

def test_tf32_round_ties_even_signed_and_two_part_precision():
    x=torch.tensor([1+2**-11,1+3*2**-11,-1-2**-11,-1-3*2**-11])
    assert torch.equal(tf32_round(x),torch.tensor([1.,1+2**-9,-1.,-1-2**-9]))
    x=torch.linspace(-3.7,4.3,257);hi=tf32_round(x);lo=tf32_round(x-hi)
    assert (hi+lo-x).norm()<.001*(hi-x).norm()

def test_zero_pv_denominator_is_explicit_and_finite():
    rows=operand_rows(torch.tensor([.5,.5]),torch.zeros(2,128,dtype=torch.bfloat16),{})
    assert len(rows)==3
    for r in rows:
        assert r['pv_near_zero'] and r['pv_relative_l2']==0 and r['zero_cancellation_denominator_channels']==128
        assert r['cancellation_quantiles']=='null'
