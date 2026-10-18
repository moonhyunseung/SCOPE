import copy
import torch
from test_diagnose import tiny
from scope_v2.replication import intervention,jobs_for

def test_original_backend_replay_and_packing_are_exact():
    model=tiny();ids=torch.tensor([[1,2,3,4,5]])
    spec=dict(bits=4,group=4,layer='all',head=None)
    with torch.inference_mode():
        past=model(ids[:,:-1],use_cache=True).past_key_values
        native=model(ids[:,-1:],past_key_values=copy.deepcopy(past),use_cache=True).logits
        results={}
        for name in ['native','K0V0','K1V1','packed11']:
            with intervention(model,spec,name) as hook:
                results[name]=model(ids[:,-1:],past_key_values=copy.deepcopy(past),use_cache=True).logits
            assert hook['calls']==2
    assert torch.equal(native,results['native']) and torch.equal(native,results['K0V0'])
    assert torch.equal(results['K1V1'],results['packed11'])

def test_head_layer_selection_and_plan_pairing():
    model=tiny();ids=torch.tensor([[1,2,3]])
    with torch.inference_mode():
        past=model(ids[:,:-1],use_cache=True).past_key_values
        with intervention(model,dict(bits=8,group=4,layer='first',head=0),'K1V0') as hook:
            model(ids[:,-1:],past_key_values=past,use_cache=True)
        assert hook['target_layers']==hook['calls']==1
    docs=[dict(document_id=str(i),domain='prose' if i%2==0 else 'code',split='screening' if i<16 else 'confirmatory') for i in range(128)]
    jobs=jobs_for(dict(models=[dict(key='model')],documents=docs,confirmation_length=1536))
    assert len(jobs)==160
    assert sum(s['split']=='confirmatory' for s in jobs)==112
    assert all(s['length']==1536 and s['dtype']=='bfloat16' for s in jobs if s['split']=='confirmatory')
