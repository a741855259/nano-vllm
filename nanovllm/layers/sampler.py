import torch
from torch import nn


class Sampler(nn.Module):
    """Greedy (temperature <= 0) or temperature sampling, one sequence at a time.

    Temperature sampling uses the same exponential-noise trick as nano-vllm:
    argmax(softmax(logits/T) / exponential(1)).
    """

    def forward(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits.float().div_(temperature)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
