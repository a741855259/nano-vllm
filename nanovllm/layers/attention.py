import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Grouped-query attention with a per-sequence "append-and-attend" KV cache.

    Each layer owns its own k_cache / v_cache, which are None until the first
    forward and are reset between sequences by ModelRunner.reset_kv_cache().
    On every step the new k/v are appended and F.scaled_dot_product_attention is
    run over the whole cache with is_causal=True, which is correct for both
    prefill (q has L tokens, kv has L) and decode (q has 1 token at the last
    position, kv has the running total).
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: [N, num_heads, head_dim], k/v: [N, num_kv_heads, head_dim]
        n = q.size(0)
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=0)
            self.v_cache = torch.cat([self.v_cache, v], dim=0)
        kv_len = self.k_cache.size(0)

        # SDPA expects [batch, heads, seq, dim].
        q_ = q.transpose(0, 1).unsqueeze(0)                          # [1, H, N, D]
        k_ = self.k_cache.transpose(0, 1).unsqueeze(0)               # [1, kvH, L, D]
        v_ = self.v_cache.transpose(0, 1).unsqueeze(0)               # [1, kvH, L, D]

        # Repeat kv heads for GQA (each kv head serves num_heads // num_kv_heads q heads).
        rep = self.num_heads // self.num_kv_heads
        if rep != 1:
            k_ = k_.repeat_interleave(rep, dim=1)
            v_ = v_.repeat_interleave(rep, dim=1)

        # Prefill (n == kv_len): a causal mask over the L x L is needed.
        # Decode (n < kv_len): the single new query sits at the last position and
        # should attend to ALL cached keys, so no mask. PyTorch's is_causal=True
        # uses upper-left alignment and would wrongly mask a short query to only
        # the first key -- so we only enable it when q_len == kv_len.
        is_causal = (n == kv_len)
        o = F.scaled_dot_product_attention(q_, k_, v_, scale=self.scale, is_causal=is_causal)  # [1, H, N, D]
        o = o.transpose(1, 2).reshape(n, self.num_heads * self.head_dim)                       # [N, H*D]
        return o
