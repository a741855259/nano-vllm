import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_model_len: int = 4096
    eos: int = -1
    hf_config: object | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model), f"model path does not exist: {self.model}"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
