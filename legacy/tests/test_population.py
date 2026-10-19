import pytest
from scope.population_inputs import window,shingles,near_duplicate
from scope.population_stats import bootstrap,paired
from scope.population import planned

def test_natural_length_pairing():
    ids=list(range(3000));w,start=window(ids,'a')
    assert len(w)==2057 and w==ids[start:start+2057]
    assert w[-521:-9]==w[:2048][-512:]
    assert w[-9:]==ids[start+2048:start+2057]
    assert window(ids,'a')==(w,start)
    with pytest.raises(ValueError):window(list(range(2056)),'short')

def test_near_duplicate_control():
    a=shingles(list(range(2057)))
    assert near_duplicate(a,[a])
    assert not near_duplicate(a,[shingles(list(range(10000,12057)))])

def test_paired_uses_document_identity_not_row_order():
    rows=[dict(document_id='a',v='K',x=9),dict(document_id='b',v='V',x=3),dict(document_id='a',v='V',x=2),dict(document_id='b',v='K',x=5),dict(document_id='missing',v='K',x=99)]
    ids,x=paired(rows,lambda r:r['v']=='K',lambda r:r['v']=='V','x')
    assert ids==['a','b'] and x==[7,2]

def test_bootstrap_document_count_and_constant_interval():
    r=bootstrap([4]*32)
    assert r['n']==32 and r['ci']==[4,4] and r['mean']==4
    assert bootstrap([4])['ci'] is None
    assert bootstrap([])['n']==0
    assert bootstrap([1,4,7])==bootstrap([1,4,7])

def test_plan_denominators_and_failure_retention():
    docs=[dict(document_id=str(i),domain='prose' if i%2==0 else 'code',split='confirmatory' if (i//2)%2 else 'exploratory',dtype_audit=i<64) for i in range(512)]
    jobs=planned(dict(documents=docs))
    assert len(jobs)==1152
    assert len({s['document_id'] for s in jobs})==512
    assert sum(s['dtype']=='float32' for s in jobs)==128
    assert len({(s['document_id'],s['length'],s['dtype']) for s in jobs})==1152


def test_failed_conditions_remain_denominator(tmp_path):
    import json
    from scope.population_stats import analyze
    (tmp_path/'campaign.json').write_text(json.dumps(dict(planned_independent_units=512)))
    spec=dict(document_id='failed',category='code',split='confirmatory',length=2048,dtype='bfloat16')
    analyze(tmp_path,[dict(id='bad',attempt='a',spec=spec,result=dict(status='CONTROL_FAIL'))])
    summary=json.loads((tmp_path/'analysis/SUMMARY.json').read_text())
    assert summary['attempted_conditions']==1 and summary['failed_conditions']==1
    assert summary['observed_valid_primary_source_units']==0 and summary['planned_source_units']==512
    assert summary['primary'][0]['n']==0 and not summary['complete']


def test_dtype_batches_are_homogeneous_even_small_smoke():
    from scope.population import batches_for
    jobs=[dict(dtype=d) for d in ['bfloat16']*4+['float32']*4]
    batches=batches_for(jobs)
    assert len(batches)==2 and all(len({s['dtype'] for s in batch})==1 for batch in batches)
    assert sum(map(len,batches))==8
