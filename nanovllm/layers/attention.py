import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,        # [N, Hkv, Dh]
    key_stride,     # Hkv*Dh
    value_ptr,      # [N, Hkv, Dh]
    value_stride,   # Hkv*Dh
    k_cache_ptr,    # [B, S, Hkv, Dh]
    v_cache_ptr,    # [B, S, Hkv, Dh]
    slot_mapping_ptr,   # [N]
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # idx 是标量，表示当前处理的 token 编号
    idx = tl.program_id(0)
    # slot 也是标量，表示目标物理 cache slot
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    # Triton arange requires a power-of-two range, while MiniMind's
    # Hkv * head_dim is 4 * 96 = 384. 
    # 需要取到最近的2幂次(即512)，剩余的mask掉
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    key_offsets = idx * key_stride + offsets
    value_offsets = idx * value_stride + offsets
    # 第idx个Token 这一层的k/v
    key = tl.load(key_ptr + key_offsets, mask=mask)
    value = tl.load(value_ptr + value_offsets, mask=mask)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    # N   = 本轮 token 总数
    # Hq  = num_attention_heads
    # Hkv = num_key_value_heads
    # Dh  = head_dim
    # D   = Hkv * Dh
    # B   = KV cache block 数量
    # S   = block_size
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # Triton 中 pointer 本身不携带这些 shape 信息
    # 因此 kernel 必须通过 stride 和 D 自行进行地址计算
    # 启动 N 个program
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
        triton.next_power_of_2(D),
    )


class Attention(nn.Module):
    """Grouped-query attention backed by a minimal paged KV cache."""

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # warmup时还不知道能分配多少 KV block，所以不能创建正式 cache
        # 让每个 Attention 层先持有空 tensor:
        self.register_buffer("k_cache", torch.tensor([]), persistent=False)
        self.register_buffer("v_cache", torch.tensor([]), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        num_tokens = q.size(0)
        # 判断 cache 是否已经分配：
        # warmup 阶段k_cache和v_cache都未分配，所以不执行 Triton KV 写入。
        # prefill attention 仍然可以运行，因为第一次完整 prefill 不需要读取 paged KV cache，可以直接对当前的 packed Q/K/V 执行
        if self.k_cache.numel() and self.v_cache.numel():
            assert context.slot_mapping is not None
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            assert context.cu_seqlens_q is not None
            if context.block_tables is None:
                # No cached prefix: packed Q/K/V share sequence boundaries.
                attn_k = k
                attn_v = v
                cu_seqlens_k = context.cu_seqlens_q
            else:
                # Chunked prefill: this layer's new K/V have just been written
                # above, so paged K/V now contains both the prefix and chunk.
                assert context.cu_seqlens_k is not None
                attn_k = self.k_cache
                attn_v = self.v_cache
                cu_seqlens_k = context.cu_seqlens_k
            o = flash_attn_varlen_func(
                q,
                attn_k,
                attn_v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
            return o.reshape(num_tokens, self.num_heads * self.head_dim)

        # Decode has one query token per request. The kernel follows each block
        # table and reads the paged cache directly.
        assert context.block_tables is not None and context.context_lens is not None
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
