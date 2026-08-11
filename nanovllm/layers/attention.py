import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Grouped-query self-attention, stateless.

    No KV cache: the caller feeds the full token sequence each step and we
    recompute attention over all of it. Since q_len == kv_len on every call we
    can always use a plain causal mask (is_causal=True) -- no paged cache, no
    alignment subtleties, no carried state.
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: [N, num_heads, head_dim], k/v: [N, num_kv_heads, head_dim]
        n = q.size(0)

        # SDPA expects [batch, heads, seq, dim].
        q_ = q.transpose(0, 1).unsqueeze(0)                          # [1, H, N, D]
        k_ = k.transpose(0, 1).unsqueeze(0)                          # [1, kvH, N, D]
        v_ = v.transpose(0, 1).unsqueeze(0)                          # [1, kvH, N, D]

        # Repeat kv heads for GQA (each kv head serves num_heads // num_kv_heads q heads).
        rep = self.num_heads // self.num_kv_heads
        if rep != 1:
            k_ = k_.repeat_interleave(rep, dim=1)
            v_ = v_.repeat_interleave(rep, dim=1)

        o = F.scaled_dot_product_attention(q_, k_, v_, scale=self.scale, is_causal=True)  # [1, H, N, D]
        o = o.transpose(1, 2).reshape(n, self.num_heads * self.head_dim)                  # [N, H*D]
        return o
