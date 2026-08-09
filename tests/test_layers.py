"""Unit tests for the pure-math layers. Run: python tests/test_layers.py"""
import torch
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding


def test_silu_and_mul():
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    act = SiluAndMul()
    gate, up = x.chunk(2, -1)
    expected = F.silu(gate) * up
    out = act(x)
    assert torch.allclose(out, expected, atol=1e-6), "SiluAndMul mismatch"
    print("  SiluAndMul: OK")


def test_rmsnorm():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    norm = RMSNorm(32, eps=1e-6)
    # F.rms_norm normalizes by the same rms with the same eps and applies weight
    expected = F.rms_norm(x.float(), (32,), weight=norm.weight.float(), eps=1e-6).to(x.dtype)
    out = norm(x)
    assert torch.allclose(out.float(), expected.float(), atol=1e-5), "RMSNorm mismatch"
    print("  RMSNorm: OK")


def test_rmsnorm_with_residual():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    residual = torch.randn(4, 32)
    norm = RMSNorm(32, eps=1e-6)
    added = x + residual
    expected = F.rms_norm(added.float(), (32,), weight=norm.weight.float(), eps=1e-6).to(x.dtype)
    out, new_residual = norm(x, residual)
    assert torch.allclose(out.float(), expected.float(), atol=1e-5), "RMSNorm(add) output mismatch"
    assert torch.allclose(new_residual.float(), added.float(), atol=1e-5), "RMSNorm(add) residual mismatch"
    print("  RMSNorm(residual): OK")


def test_rope_identity_at_zero():
    # RoPE at position 0 is identity (cos=1, sin=0).
    rope = RotaryEmbedding(head_size=16, rotary_dim=16, max_position_embeddings=64, base=10000.0)
    q = torch.randn(1, 4, 16)
    k = torch.randn(1, 2, 16)
    positions = torch.tensor([0])
    q2, k2 = rope(positions, q.clone(), k.clone())
    assert torch.allclose(q2, q, atol=1e-6), "RoPE(q) not identity at pos 0"
    assert torch.allclose(k2, k, atol=1e-6), "RoPE(k) not identity at pos 0"
    print("  RoPE identity@0: OK")


def test_rope_preserves_norm():
    # RoPE is a rotation, so per-head norms are preserved.
    rope = RotaryEmbedding(head_size=16, rotary_dim=16, max_position_embeddings=64, base=10000.0)
    q = torch.randn(2, 4, 16)
    positions = torch.tensor([3, 7])
    q2, _ = rope(positions, q.clone(), q.clone())
    n1 = q.float().norm(dim=-1)
    n2 = q2.float().norm(dim=-1)
    assert torch.allclose(n1, n2, atol=1e-4), f"RoPE changed norm: {n1} vs {n2}"
    print("  RoPE norm-preserving: OK")


def main():
    print("Running layer unit tests...")
    test_silu_and_mul()
    test_rmsnorm()
    test_rmsnorm_with_residual()
    test_rope_identity_at_zero()
    test_rope_preserves_norm()
    print("All layer tests passed.")


if __name__ == "__main__":
    main()
