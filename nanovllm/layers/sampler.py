import torch
from torch import nn


class Sampler(nn.Module):
    """Batched greedy or temperature sampling.

    Temperature sampling uses the same exponential-noise trick as nano-vllm:
    argmax(softmax(logits/T) / exponential(1)).
    """

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        greedy_tokens = logits.argmax(dim=-1)
        greedy = temperatures <= 0
        safe_temperatures = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        logits = logits.float().div_(safe_temperatures.unsqueeze(1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy, greedy_tokens, sample_tokens)
