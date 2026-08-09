import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import TP_SIZE, TP_RANK


class VocabParallelEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = TP_RANK
        self.tp_size = TP_SIZE
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # tp_size == 1: no masking / all-reduce needed
        return F.embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # The caller (ModelRunner) is responsible for slicing x to the last token
        # before calling compute_logits, so this is a plain projection.
        return F.linear(x, self.weight)
