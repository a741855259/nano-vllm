import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_seqs: int = 1
    max_model_len: int = 4096
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    eos: int = -1
    hf_config: object | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model), f"model path does not exist: {self.model}"
        assert self.max_num_seqs > 0
        assert self.kvcache_block_size > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        min_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        if self.num_kvcache_blocks == -1:
            self.num_kvcache_blocks = min_blocks
        assert self.num_kvcache_blocks >= min_blocks, (
            f"num_kvcache_blocks must be at least {min_blocks} for max_model_len"
        )
