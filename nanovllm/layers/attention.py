import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


class Attention(nn.Module):
    """Grouped-query attention backed by a minimal paged KV cache."""

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        assert self.k_cache is not None and self.v_cache is not None
        assert context.slot_mapping is not None and context.block_table is not None

        # Write new K/V into physical cache slots selected by the block table.
        q_len = q.size(0)
        assert context.slot_mapping.numel() == q_len
        flat_k_cache = self.k_cache.flatten(0, 1)
        flat_v_cache = self.v_cache.flatten(0, 1)
        flat_k_cache.index_copy_(0, context.slot_mapping, k)
        flat_v_cache.index_copy_(0, context.slot_mapping, v)

        # SDPA does not consume a block table directly, so gather the logical
        # sequence from physical pages before attention. A paged kernel can
        # replace this gather later without changing allocation or scheduling.
        logical_positions = torch.arange(context.context_len, device=q.device)
        logical_blocks = torch.div(logical_positions, context.block_size, rounding_mode="floor")
        physical_slots = (
            context.block_table[logical_blocks] * context.block_size
            + logical_positions.remainder(context.block_size)
        )
        k = flat_k_cache[physical_slots]
        v = flat_v_cache[physical_slots]

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
            q_, k_, v_, scale=self.scale, is_causal=context.is_prefill,
        )
        o = o.transpose(1, 2).reshape(q_len, self.num_heads * self.head_dim)
        return o
