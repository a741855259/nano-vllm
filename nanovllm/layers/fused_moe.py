from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _top1_routed_matmul_kernel(
    a_ptr,
    weight_ptr,
    expert_ids_ptr,
    routed_weights_ptr,
    output_ptr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    multiply_routed_weight: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    token_id = tl.program_id(0)
    n_block_id = tl.program_id(1)
    expert_id = tl.load(expert_ids_ptr + token_id)
    n_offsets = n_block_id * block_n + tl.arange(0, block_n)
    k_offsets = tl.arange(0, block_k)

    a_ptrs = (
        a_ptr
        + token_id * stride_am
        + k_offsets[None, :] * stride_ak
    )
    weight_ptrs = (
        weight_ptr
        + expert_id * stride_we
        + k_offsets[:, None] * stride_wk
        + n_offsets[None, :] * stride_wn
    )
    accumulator = tl.zeros((1, block_n), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(k_size, block_k)):
        remaining = k_size - k_block * block_k
        a = tl.load(
            a_ptrs,
            mask=k_offsets[None, :] < remaining,
            other=0.0,
        )
        weight = tl.load(
            weight_ptrs,
            mask=(k_offsets[:, None] < remaining)
            & (n_offsets[None, :] < n_size),
            other=0.0,
        )
        accumulator += tl.dot(a, weight)
        a_ptrs += block_k * stride_ak
        weight_ptrs += block_k * stride_wk

    if multiply_routed_weight:
        routed_weight = tl.load(routed_weights_ptr + token_id)
        accumulator *= routed_weight

    output_ptrs = (
        output_ptr
        + token_id * stride_om
        + n_offsets[None, :] * stride_on
    )
    tl.store(
        output_ptrs,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=n_offsets[None, :] < n_size,
    )


def top1_routed_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    expert_ids: torch.Tensor,
    routed_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multiply each token by the selected expert's weight matrix."""
    if not x.is_cuda:
        raise ValueError("top1_routed_matmul requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported routed matmul dtype: {x.dtype}")
    if weight.ndim != 3 or weight.shape[2] != x.shape[1]:
        raise ValueError("expert weight and input dimensions do not match")
    if expert_ids.shape != (x.shape[0], 1):
        raise ValueError("top1 routed matmul requires expert_ids shape [M, 1]")
    if routed_weights is not None and routed_weights.shape != expert_ids.shape:
        raise ValueError("routed weights and expert ids must have matching shapes")

    x = x.contiguous()
    expert_ids = expert_ids.contiguous()
    output = torch.empty(
        (x.shape[0], weight.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    block_n = 64
    block_k = 64
    grid = (x.shape[0], triton.cdiv(weight.shape[1], block_n))
    _top1_routed_matmul_kernel[grid](
        x,
        weight,
        expert_ids,
        routed_weights if routed_weights is not None else expert_ids,
        output,
        n_size=weight.shape[1],
        k_size=weight.shape[2],
        # [M, K]
        # K
        stride_am=x.stride(0),
        # 1
        stride_ak=x.stride(1),
        # E：专家数, M：token 数, N：输出维度, K：输入维度
        # [E,N,K]
        # N*K
        stride_we=weight.stride(0),
        # K
        stride_wn=weight.stride(1),
        # 1
        stride_wk=weight.stride(2),
        # [M,N]
        # N
        stride_om=output.stride(0),
        # 1
        stride_on=output.stride(1),
        multiply_routed_weight=routed_weights is not None,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=3,
    )
    return output


class FusedExperts(nn.Module):
    """Packed SwiGLU experts with a CUDA Top-1 decode kernel."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # packed专家权重矩阵
        # w13: gate_proj + up_proj
        self.w13_weight = nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size,
            hidden_size,
        ))
        # w2: down_proj
        self.w2_weight = nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
        ))

    # 按专家ID逐步加载到同一个packed专家权重矩阵
    def load_expert_weight(
        self,
        expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
    ) -> None:
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(f"invalid expert id {expert_id}")
        if projection == "gate_proj":
            target = self.w13_weight.data[expert_id, :self.intermediate_size]
        elif projection == "up_proj":
            target = self.w13_weight.data[expert_id, self.intermediate_size:]
        elif projection == "down_proj":
            target = self.w2_weight.data[expert_id]
        else:
            raise ValueError(f"unsupported expert projection {projection!r}")
        if target.shape != loaded_weight.shape:
            raise ValueError(
                f"expected {projection} shape {tuple(target.shape)}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight)

    def _grouped_forward(
        self,
        x: torch.Tensor,
        routed_weights: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        for expert_id in range(self.num_experts):
            token_ids, topk_ids = torch.where(expert_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            gate_up = F.linear(x[token_ids], self.w13_weight[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                self.w2_weight[expert_id],
            )
            weight = routed_weights[token_ids, topk_ids, None].to(x.dtype)
            output.index_add_(0, token_ids, expert_output * weight)
        return output

    def forward(
        self,
        x: torch.Tensor,
        routed_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        if not is_prefill and x.is_cuda and expert_ids.shape[1] == 1:
            gate_up = top1_routed_matmul(x, self.w13_weight, expert_ids)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            return top1_routed_matmul(
                hidden,
                self.w2_weight,
                expert_ids,
                routed_weights,
            )
        return self._grouped_forward(x, routed_weights, expert_ids)
