import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Grouped-query self-attention with a minimal, single-sequence KV cache.

    The first call is a causal prefill over the whole prompt. Later calls must
    contain one token; its key/value are appended to the cache and its query
    attends to the complete cache. This intentionally omits paging, prefix
    sharing and cache preallocation.
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None

    @property
    def cache_len(self) -> int:
        return 0 if self.k_cache is None else self.k_cache.size(0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: [Q, num_heads, head_dim], k/v: [Q, num_kv_heads, head_dim]
        q_len = q.size(0)
        is_prefill = self.k_cache is None
        if is_prefill:
            self.k_cache = k
            self.v_cache = v
        else:
            assert q_len == 1, "cached decode expects exactly one new token"
            self.k_cache = torch.cat((self.k_cache, k), dim=0)
            self.v_cache = torch.cat((self.v_cache, v), dim=0)

        k = self.k_cache
        v = self.v_cache

        # SDPA expects [batch, heads, seq, dim].
        q_ = q.transpose(0, 1).unsqueeze(0)                          # [1, H, N, D]
        k_ = k.transpose(0, 1).unsqueeze(0)                          # [1, kvH, N, D]
        v_ = v.transpose(0, 1).unsqueeze(0)                          # [1, kvH, N, D]

        # Repeat kv heads for GQA (each kv head serves num_heads // num_kv_heads q heads).
        rep = self.num_heads // self.num_kv_heads
        if rep != 1:
            k_ = k_.repeat_interleave(rep, dim=1)
            v_ = v_.repeat_interleave(rep, dim=1)

        # During decode q_len == 1 and the query is the final sequence position,
        # so an SDPA causal mask would incorrectly let it see only key 0.
        o = F.scaled_dot_product_attention(
            q_, k_, v_, scale=self.scale, is_causal=is_prefill,
        )
        o = o.transpose(1, 2).reshape(q_len, self.num_heads * self.head_dim)
        return o
