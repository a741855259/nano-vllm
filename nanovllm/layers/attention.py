import torch
from torch import nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

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

        if context.is_prefill:
            # Full prefill has no cached prefix, so Q/K/V share the same packed
            # sequence boundaries. FlashAttention handles variable lengths and
            # GQA directly without padding, explicit masks, or K/V head copies.
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_q,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
            return o.reshape(num_tokens, self.num_heads * self.head_dim)

        # Decode has one query token per request. The kernel follows each block
        # table and reads the paged cache directly.
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
