import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, with an optional fused residual add.

    Variance is computed in float32 then cast back to the original dtype, matching
    the reference implementation (and HF's Qwen3 RMSNorm).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x

    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Non-in-place so the returned residual never aliases the normed output
        # (matters when x is already float32; nano-vllm's in-place version relies
        # on bf16 production inputs + a fused torch.compile kernel).
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
