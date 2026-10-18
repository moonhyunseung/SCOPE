import torch
from scope_v2.role_fidelity import role_value, checked_value, mixed_cache, byte_equal, decision


def fixture():
    k = torch.linspace(-3, 3, 512*32).reshape(1, 1, 512, 32).bfloat16()
    v = (k.flip(-1) * .37).bfloat16()
    k[..., 0] = -1e-6
    return k, v


def test_mixed_roles_apply_at_initialization_and_every_append():
    k, v = fixture()
    for role, flags in [('R0', (False, False)), ('R1', (True, False)), ('R2', (False, True)), ('R3', (True, True))]:
        controls = []
        cache = mixed_cache([(k, v)], role, controls, device='cpu')
        layer = cache.layers[0]
        assert byte_equal(layer.keys, checked_value(k, flags[0]))
        assert byte_equal(layer.values, checked_value(v, flags[1]))
        for step in range(4):
            nk, nv = k[..., step:step+1, :], v[..., step:step+1, :]
            layer.update(nk, nv)
            assert byte_equal(layer.keys[..., -1:, :], role_value(nk, flags[0]))
            assert byte_equal(layer.values[..., -1:, :], role_value(nv, flags[1]))
            assert byte_equal(layer.keys[..., :512, :], role_value(k, flags[0]))
        assert layer.get_seq_length() == 516 and len(controls) == 4
        assert all(x['append_exact'] and x['prefix_exact'] for x in controls)


def test_baseline_cancellation_is_not_accepted_and_boundary_is_inclusive():
    assert decision(.025, .025, .025)['local_candidate_status'] == 'LOCAL_SCREEN_PASS'
    assert decision(.01, .01, .03)['local_candidate_status'] == 'BASELINE_DIVERGENCE_UNRESOLVED'
    assert decision(.02500001, .01, .02)['cached_gate'] == 'FAIL'
    assert decision(.01, .02500001, .02)['full_gate'] == 'FAIL'


def test_unselected_negative_zero_preserved_selected_integer_zero_canonicalized():
    x = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
    x[..., 0] = -0.0
    assert byte_equal(checked_value(x, False), x)
    assert torch.signbit(checked_value(x, False)[..., 0]).all()
    assert not torch.signbit(checked_value(x, True)[..., 0]).any()
