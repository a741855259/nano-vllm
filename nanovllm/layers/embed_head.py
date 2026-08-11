import torch
from torch import nn
import torch.nn.functional as F


class VocabParallelEmbedding(nn.Module):
    # Single-GPU: a plain token embedding (no vocab sharding / masking / all-reduce).

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return F.embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):
    # Single-GPU: a plain LM head projection. The caller (ModelRunner) slices
    # hidden states to the last token before calling compute_logits.

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight)
