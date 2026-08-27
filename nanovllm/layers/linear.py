import torch
from torch import nn

from nanovllm.layers.quantization import (
    QuantizationConfig,
    UnquantizedLinearMethod,
)

# Single-GPU only -- no tensor parallelism. The "Parallel" naming is inherited
# from nano-vllm/vLLM (where these classes shard across TP ranks); here they are
# plain linears. What remains meaningful is the *weight fusion*: q/k/v are fused
# into a single qkv_proj and gate/up into a single gate_up_proj, so each is one
# matmul, and the weight_loader places the sub-weights at offsets (not sharding).


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        self.prefix = prefix
        self.quant_method = (
            UnquantizedLinearMethod()
            if quant_config is None
            else quant_config.get_quant_method(self, prefix)
        )
        if self.quant_method is None:
            raise ValueError(f"quantization does not support linear layer {prefix!r}")
        self.quant_method.create_weights(self, input_size, output_size)
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param is self.bias:
            param.data.copy_(loaded_weight)
        else:
            self.quant_method.load_weight(self, loaded_weight, 0, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ColumnParallelLinear(LinearBase):
    # Single-GPU: a plain linear. Base class for the fused projections below.

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(input_size, output_size, bias, quant_config, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x)


class MergedColumnParallelLinear(ColumnParallelLinear):
    # Fuses two same-input projections (gate, up) into one weight, placing each
    # at an offset along the output dim. One fused matmul, not tensor parallelism.

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size,
            sum(output_sizes),
            bias,
            quant_config,
            prefix,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        if param is self.bias:
            param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight)
        else:
            self.quant_method.load_weight(
                self, loaded_weight, shard_offset, shard_size,
            )


class QKVParallelLinear(ColumnParallelLinear):
    # Fuses q, k, v projections into one weight: q at offset 0, k after q, v after k.

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        output_size = (num_heads + 2 * num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias, quant_config, prefix)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (
                self.num_heads * self.head_size
                + self.num_kv_heads * self.head_size
            )
        if param is self.bias:
            param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight)
        else:
            self.quant_method.load_weight(
                self, loaded_weight, shard_offset, shard_size,
            )


class RowParallelLinear(LinearBase):
    # Single-GPU: a plain linear (no input-dim sharding, no all-reduce).

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(input_size, output_size, bias, quant_config, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x)
