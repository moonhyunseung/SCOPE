import torch
import torch.nn.functional as F

from scope.reference import reference_decode_attention, symmetric_int4_groupwise
from scope.observations import code_histogram, probability_summary, trace_summary


def test_quantizer_exact_decoding_and_zero_groups():
    x = torch.zeros((1, 2, 3, 64), dtype=torch.float32)
    q = symmetric_int4_groupwise(x, 32)
    assert q.codes.dtype == torch.int8
    assert q.scales.shape == (1, 2, 3, 2)
    assert torch.equal(q.decoded, x)
    assert sum(code_histogram(q.codes).values()) == x.numel()


def test_int4_groupwise_semantics():
    torch.manual_seed(13)
    x = torch.randn(1, 2, 5, 64)
    q = symmetric_int4_groupwise(x, 32)
    expected = (q.codes.reshape(1, 2, 5, 2, 32).float() * q.scales.unsqueeze(-1)).reshape(x.shape)
    assert torch.equal(q.decoded, expected)
    assert (q.codes >= -7).all() and (q.codes <= 7).all()


def test_gqa_attention_matches_sdpa_and_w_contract():
    torch.manual_seed(8)
    q = torch.randn(2, 6, 1, 32)
    k = torch.randn(2, 2, 19, 32)
    v = torch.randn(2, 2, 19, 32)
    kq = symmetric_int4_groupwise(k, 16)
    vq = symmetric_int4_groupwise(v, 16)
    t = reference_decode_attention(q, kq.decoded, vq.decoded, v_scales=vq.scales)
    kh = kq.decoded.repeat_interleave(3, dim=1)
    vh = vq.decoded.repeat_interleave(3, dim=1)
    target = F.scaled_dot_product_attention(q, kh, vh, is_causal=False)
    assert t.z.shape == (2, 6, 1, 19)
    assert t.p.shape == (2, 6, 1, 19)
    assert t.y.shape == (2, 6, 1, 32)
    assert t.w.shape == (2, 6, 19, 2)
    assert torch.allclose(t.y, target, atol=1e-5, rtol=1e-5)
    expected_w = t.p.squeeze(-2).unsqueeze(-1) * vq.scales.repeat_interleave(3, dim=1)
    assert torch.allclose(t.w, expected_w, atol=0, rtol=0)
    assert probability_summary(t.p)["max_row_sum_error"] < 1e-6
    summary = trace_summary(t, kq, vq)
    assert "W=P_times_S_V" in summary["tensor_statistics"]


def test_shape_contract_rejects_bad_gqa():
    q = torch.randn(1, 5, 1, 32)
    k = torch.randn(1, 2, 3, 32)
    try:
        reference_decode_attention(q, k, k)
    except ValueError:
        pass
    else:
        raise AssertionError("expected GQA head-divisibility check")
