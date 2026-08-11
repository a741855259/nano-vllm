"""Unit tests for the stateless GQA attention. Run: python tests/test_attention.py"""
import torch
import torch.nn.functional as F

from nanovllm.layers.attention import Attention


def make_attn():
    # GQA: 4 q heads, 2 kv heads, head_dim 8
    return Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=2)


def test_output_shape():
    attn = make_attn()
    torch.manual_seed(0)
    q = torch.randn(5, 4, 8)
    k = torch.randn(5, 2, 8)
    v = torch.randn(5, 2, 8)
    o = attn(q, k, v)
    assert o.shape == (5, 4 * 8), o.shape
    print("  output shape: OK")


def test_causal_masking():
    # With is_causal=True, query at position i must not attend to keys at j > i.
    # Mutating a FUTURE key leaves outputs for earlier positions unchanged, while
    # the last position's output (which sees that key) does change.
    attn = make_attn()
    torch.manual_seed(1)
    n = 6
    q = torch.randn(n, 4, 8)
    k = torch.randn(n, 2, 8)
    v = torch.randn(n, 2, 8)
    o_ref = attn(q, k, v)

    k2 = k.clone()
    k2[-1] = torch.randn(2, 8) * 10.0
    o_mut = attn(q, k2, v)

    diff_early = (o_ref[:n - 1] - o_mut[:n - 1]).abs().max().item()
    diff_last = (o_ref[-1] - o_mut[-1]).abs().max().item()
    assert diff_early < 1e-5, f"early output changed under future-key mutation: {diff_early}"
    assert diff_last > 1e-4, f"last output did not change under its own key mutation: {diff_last}"
    print("  causal masking: OK")


def test_gqa_matches_manual():
    # The repeat_interleave grouping must match a manual GQA reference.
    attn = make_attn()
    torch.manual_seed(3)
    n = 4
    q = torch.randn(n, 4, 8)
    k = torch.randn(n, 2, 8)
    v = torch.randn(n, 2, 8)
    o = attn(q, k, v)  # [n, 32]

    kr = k.repeat_interleave(2, dim=1)   # [n, 4, 8]: heads [kv0,kv0,kv1,kv1]
    vr = v.repeat_interleave(2, dim=1)
    ref = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), kr.transpose(0, 1).unsqueeze(0),
        vr.transpose(0, 1).unsqueeze(0), scale=8 ** -0.5, is_causal=True,
    ).transpose(1, 2).reshape(n, 32)
    assert torch.allclose(o, ref, atol=1e-5), "GQA repeat grouping mismatch"
    print("  GQA grouping: OK")


def test_stateless():
    # No carried state: the same input always yields the same output, and
    # calling repeatedly does not accumulate anything.
    attn = make_attn()
    torch.manual_seed(2)
    q = torch.randn(3, 4, 8); k = torch.randn(3, 2, 8); v = torch.randn(3, 2, 8)
    o1 = attn(q, k, v)
    o2 = attn(q, k, v)
    assert torch.allclose(o1, o2), "attention is not stateless"
    assert not any(hasattr(attn, attr) for attr in ("k_cache", "v_cache")), "leftover cache state"
    print("  stateless: OK")


def main():
    print("Running attention unit tests...")
    test_output_shape()
    test_causal_masking()
    test_gqa_matches_manual()
    test_stateless()
    print("All attention tests passed.")


if __name__ == "__main__":
    main()
