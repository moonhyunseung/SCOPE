import pytest
from scope_v2.confirmation_audit import paired_models

def test_shared_source_variation_cancels_in_joint_model_comparison():
    left={str(i):i*1000.+.125 for i in range(56)};right={str(i):i*1000. for i in range(56)}
    result=paired_models(left,right,set(left),repeats=1000)
    assert result['n']==56 and result['mean']==.125 and result['ci']==[.125,.125]

def test_missing_model_source_is_not_silently_filtered():
    with pytest.raises(ValueError,match='incomplete paired'):
        paired_models({'a':1.},{'b':1.},{'a','b'},repeats=10)
