import torch
from torch import nn
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

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
        assert context.slot_mapping is not None
        assert context.block_tables is not None and context.context_lens is not None
        assert context.cu_seqlens_q is not None

        # Write new K/V into physical cache slots selected by the block table.
        num_tokens = q.size(0)
        assert context.slot_mapping.numel() == num_tokens
        flat_k_cache = self.k_cache.flatten(0, 1)
        flat_v_cache = self.v_cache.flatten(0, 1)
        flat_k_cache.index_copy_(0, context.slot_mapping, k)
        flat_v_cache.index_copy_(0, context.slot_mapping, v)

        if not context.is_prefill:
            # Decode has one query token per request. The kernel follows each
            # block table and reads the paged cache directly, avoiding the
            # dense K/V gather used by the SDPA prefill path below.
            batch_size = context.context_lens.numel()
            assert num_tokens == batch_size
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                self.k_cache,
                self.v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
            return o.reshape(num_tokens, self.num_heads * self.head_dim)

        # Prefill remains on SDPA for now. SDPA does not consume a block table,
        # so gather each logical sequence from its physical pages.
        batch_size = context.context_lens.numel()
        logical_positions = torch.arange(context.max_seqlen_k, device=q.device)
        logical_blocks = torch.div(logical_positions, context.block_size, rounding_mode="floor")
        block_ids = context.block_tables[:, logical_blocks]
        valid_keys = logical_positions.unsqueeze(0) < context.context_lens.unsqueeze(1)
        physical_slots = (
            block_ids * context.block_size + logical_positions.remainder(context.block_size)
        )
        physical_slots = physical_slots.masked_fill(~valid_keys, 0)
        k = flat_k_cache[physical_slots]
        v = flat_v_cache[physical_slots]

        # SDPA expects [batch, heads, seq, dim]. K/V are gathered from each
        # sequence's page table into a padded batch.
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)

        # Repeat kv heads for GQA (each kv head serves num_heads // num_kv_heads q heads).
        rep = self.num_heads // self.num_kv_heads
        if rep != 1:
            k_ = k_.repeat_interleave(rep, dim=1)
            v_ = v_.repeat_interleave(rep, dim=1)

        query_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
        seq_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=q.device), query_lens,
        )
        query_offsets = (
            torch.arange(num_tokens, device=q.device)
            - torch.repeat_interleave(context.cu_seqlens_q[:-1], query_lens)
        )
        q_padded = q.new_zeros(
            batch_size, context.max_seqlen_q, self.num_heads, self.head_dim,
        )
        q_padded[seq_ids, query_offsets] = q
        q_ = q_padded.transpose(1, 2)

        query_positions = torch.arange(context.max_seqlen_q, device=q.device)
        valid_queries = query_positions.unsqueeze(0) < query_lens.unsqueeze(1)
        causal = logical_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attn_mask = (
            valid_queries[:, None, :, None]
            & valid_keys[:, None, None, :]
            & causal[None, None, :, :]
        )

        o = F.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=attn_mask, scale=self.scale,
        )
        o = o.transpose(1, 2)
        o = o[seq_ids, query_offsets]
        return o.reshape(num_tokens, self.num_heads * self.head_dim)
