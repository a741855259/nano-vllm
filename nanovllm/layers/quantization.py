from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def quantize_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize a 2D weight by input-dimension groups."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if group_size <= 0 or group_size % 2:
        raise ValueError("group_size must be a positive even integer")
    if in_features % group_size:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size "
            f"({group_size})"
        )

    groups = weight.float().reshape(out_features, -1, group_size)
    max_abs = groups.abs().amax(dim=-1)
    scales = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 7)
    quantized = torch.round(groups / scales.unsqueeze(-1)).clamp_(-7, 7)
    quantized = quantized.to(torch.int8).reshape(out_features, in_features)
    low = quantized[:, 0::2].to(torch.uint8) & 0x0F
    high = (quantized[:, 1::2].to(torch.uint8) & 0x0F) << 4
    return (low | high).contiguous(), scales.to(torch.float16).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack two's-complement INT4 pairs to an int8 tensor."""
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise ValueError("packed weight must be a 2D uint8 tensor")
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(1)
    signed = nibbles.to(torch.int8)
    return torch.where(signed >= 8, signed - 16, signed)


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference dequantizer used by CPU fallback and correctness tests."""
    quantized = unpack_int4(packed)
    if quantized.shape[1] % group_size:
        raise ValueError("unpacked input dimension must be divisible by group_size")
    expected = (quantized.shape[0], quantized.shape[1] // group_size)
    if tuple(scales.shape) != expected:
        raise ValueError(f"expected scales shape {expected}, got {tuple(scales.shape)}")
    scale_values = scales.to(torch.float32).repeat_interleave(group_size, dim=1)
    return (quantized.float() * scale_values).to(dtype)


@triton.jit
def _int4_linear_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            weight_ptr + offs_n[:, None] * (K // 2) + (offs_k[None, :] // 2),
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0,
        )
        nibble = tl.where(
            (offs_k[None, :] & 1) == 0,
            packed & 0x0F,
            (packed >> 4) & 0x0F,
        ).to(tl.int32)
        signed = tl.where(nibble < 8, nibble, nibble - 16).to(tl.float32)
        scales = tl.load(
            scale_ptr
            + offs_n[:, None] * (K // GROUP_SIZE)
            + offs_k[None, :] // GROUP_SIZE,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        weight = (signed * scales).to(x.dtype)
        accumulator += tl.dot(x, tl.trans(weight))

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def int4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a W4A16 linear, using a fused Triton kernel on CUDA."""
    if x.shape[-1] != packed_weight.shape[1] * 2:
        raise ValueError("input and packed weight dimensions do not match")
    if not x.is_cuda:
        weight = dequantize_int4(
            packed_weight,
            scales,
            group_size,
            dtype=x.dtype,
        )
        return F.linear(x, weight, bias)
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"INT4 CUDA linear requires FP16/BF16 input, got {x.dtype}")

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x_2d.shape
    n = packed_weight.shape[0]
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 64))
    _int4_linear_kernel[grid](
        x_2d,
        packed_weight,
        scales,
        bias if bias is not None else output,
        output,
        M=m,
        N=n,
        K=k,
        GROUP_SIZE=group_size,
        HAS_BIAS=bias is not None,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
    )
    return output.reshape(*x.shape[:-1], n)


class LinearMethod(ABC):

    @abstractmethod
    def create_weights(self, layer, input_size: int, output_size: int):
        raise NotImplementedError

    @abstractmethod
    def load_weight(
        self,
        layer,
        loaded_weight: torch.Tensor,
        output_offset: int,
        output_size: int,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QuantizationConfig(ABC):

    @abstractmethod
    def get_quant_method(self, layer, prefix: str) -> LinearMethod | None:
        """Select the weight and compute method for one linear layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethod):

    def create_weights(self, layer, input_size: int, output_size: int):
        layer.weight = torch.nn.Parameter(torch.empty(output_size, input_size))

    def load_weight(self, layer, loaded_weight, output_offset, output_size):
        layer.weight.data.narrow(0, output_offset, output_size).copy_(loaded_weight)

    def apply(self, layer, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, layer.weight, layer.bias)


class Int4WeightOnlyLinearMethod(LinearMethod):

    def __init__(self, group_size: int = 128):
        if group_size <= 0 or group_size % 2:
            raise ValueError("group_size must be a positive even integer")
        self.group_size = group_size

    def create_weights(self, layer, input_size: int, output_size: int):
        if input_size % self.group_size:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by INT4 group_size "
                f"({self.group_size})"
            )
        layer.weight = torch.nn.Parameter(
            torch.empty(output_size, input_size // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_buffer(
            "weight_scale",
            torch.empty(output_size, input_size // self.group_size, dtype=torch.float16),
        )

    def load_weight(self, layer, loaded_weight, output_offset, output_size):
        if tuple(loaded_weight.shape) != (output_size, layer.input_size):
            raise ValueError(
                f"expected weight shape {(output_size, layer.input_size)}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        packed, scales = quantize_int4(loaded_weight, self.group_size)
        layer.weight.data.narrow(0, output_offset, output_size).copy_(packed)
        layer.weight_scale.narrow(0, output_offset, output_size).copy_(scales)

    def apply(self, layer, x: torch.Tensor) -> torch.Tensor:
        return int4_linear(
            x,
            layer.weight,
            layer.weight_scale,
            self.group_size,
            layer.bias,
        )


class Int4QuantizationConfig(QuantizationConfig):

    def __init__(self, group_size: int = 128):
        if group_size <= 0 or group_size % 2:
            raise ValueError("group_size must be a positive even integer")
        self.group_size = group_size

    def get_quant_method(self, layer, prefix: str) -> LinearMethod:
        return Int4WeightOnlyLinearMethod(self.group_size)


def get_quantization_config(
    quantization: str | None,
    group_size: int = 128,
) -> QuantizationConfig | None:
    if quantization is None:
        return None
    if quantization == "int4":
        return Int4QuantizationConfig(group_size)
    raise ValueError(f"unsupported quantization: {quantization!r}")
