import torch
from scope_v2.resident import compare_logits,plan

def test_gate_uses_each_request_and_does_not_hide_outlier_in_global_norm():
    a=torch.tensor([[1000.,0.],[1.,0.]])
    b=torch.tensor([[1000.,0.],[1.,.1]])
    r=compare_logits(a,b)
    assert abs(r['max_request_relative_l2']-.1)<1e-7
    assert r['max_request_relative_l2']>.025
    assert r['top1_agreement']==1.

def test_hardware_plan_does_not_draw_confirmation_sources():
    data={'documents':[dict(document_id='screen-a',domain='prose'),dict(document_id='screen-b',domain='code'),dict(document_id='confirm',domain='prose')]}
    jobs=plan(data,'frontier')
    assert len(jobs)==72
    assert {j['document_id'] for j in jobs}=={'screen-a','screen-b'}
    assert len(plan(data,'pilot'))==6
