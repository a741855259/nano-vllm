"""Unit tests for the SDPA append-and-attend attention. Run: python tests/test_attention.py"""
import torch

from nanovllm.layers.attention import Attention


def make_attn():
    # GQA: 4 q heads, 2 kv heads, head_dim 8
    attn = Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=2)
    return attn


def _run(attn, n):
    torch.manual_seed(0)
    q = torch.randn(n, 4, 8)
    k = torch.randn(n, 2, 8)
    v = torch.randn(n, 2, 8)
    return attn(q, k, v)


def test_output_shape_and_cache_growth():
    attn = make_attn()
    o = _run(attn, 5)  # prefill
    assert o.shape == (5, 4 * 8), o.shape
    assert attn.k_cache.shape == (5, 2, 8), attn.k_cache.shape
    # decode step
    o1 = _run(attn, 1)
    assert o1.shape == (1, 4 * 8), o1.shape
    assert attn.k_cache.shape == (6, 2, 8), attn.k_cache.shape
    print("  shapes / cache growth: OK")


def test_causal_masking():
    # In prefill, query at position i must not attend to keys at positions > i.
    # Changing a future key (position > i) leaves output[i] unchanged.
    attn = make_attn()
    torch.manual_seed(1)
    n = 6
    q = torch.randn(n, 4, 8)
    k = torch.randn(n, 2, 8)
    v = torch.randn(n, 2, 8)
    o_ref = attn(q, k, v)

    # mutate the LAST key only; outputs for positions 0..n-2 must be identical
    k2 = k.clone()
    k2[-1] = torch.randn(2, 8) * 10.0
    attn.k_cache = None
    o_mut = attn(q, k2, v)

    diff_early = (o_ref[:n - 1] - o_mut[:n - 1]).abs().max().item()
    diff_last = (o_ref[-1] - o_mut[-1]).abs().max().item()
    assert diff_early < 1e-5, f"early output changed under future-key mutation: {diff_early}"
    assert diff_last > 1e-4, f"last output did not change under its own key mutation: {diff_last}"
    print("  causal masking: OK")


def test_decode_matches_prefill_last_row():
    # The decode output for the newest token must equal the last row of a full
    # prefill over the same (L+1) tokens: the new query sits at the last position
    # and attends to ALL cached keys. This is the regression test for the
    # is_causal upper-left-alignment bug (decode must NOT mask out older keys).
    attn = make_attn()
    torch.manual_seed(2)
    L = 5
    q = torch.randn(L, 4, 8); k = torch.randn(L, 2, 8); v = torch.randn(L, 2, 8)
    attn(q, k, v)  # prefill L tokens, fills cache
    q1 = torch.randn(1, 4, 8); k1 = torch.randn(1, 2, 8); v1 = torch.randn(1, 2, 8)
    o_decode = attn(q1, k1, v1)                       # [1, 32]

    attn2 = make_attn()
    qf = torch.cat([q, q1]); kf = torch.cat([k, k1]); vf = torch.cat([v, v1])
    o_full = attn2(qf, kf, vf)                        # [L+1, 32]
    assert torch.allclose(o_decode, o_full[-1:], atol=1e-5), \
        f"decode != prefill last row: max diff {(o_decode - o_full[-1:]).abs().max()}"
    print("  decode == prefill-last-row: OK")


def test_reset():
    attn = make_attn()
    _run(attn, 3)
    assert attn.k_cache is not None
    attn.k_cache = None
    attn.v_cache = None
    assert attn.k_cache is None
    # first forward after reset replaces instead of concatenating
    o = _run(attn, 2)
    assert attn.k_cache.shape == (2, 2, 8), attn.k_cache.shape
    print("  reset: OK")


def main():
    print("Running attention unit tests...")
    test_output_shape_and_cache_growth()
    test_causal_masking()
    test_decode_matches_prefill_last_row()
    test_reset()
    print("All attention tests passed.")


if __name__ == "__main__":
    main()
