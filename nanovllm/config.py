import os
from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.models.configs import register_model_configs


register_model_configs()


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 3
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    vision_model: str | None = None
    max_num_images_per_prompt: int = 1
    quantization: str | None = None
    quantization_group_size: int = 128
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    eos: int = -1
    hf_config: object | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model), f"model path does not exist: {self.model}"
        assert self.max_num_batched_tokens > 0
        assert self.max_num_seqs > 0
        assert self.max_num_images_per_prompt > 0
        if self.quantization not in (None, "int4"):
            raise ValueError(
                f"unsupported quantization {self.quantization!r}; "
                "supported values are None and 'int4'"
            )
        if self.quantization_group_size <= 0:
            raise ValueError("quantization_group_size must be positive")
        if self.quantization_group_size % 2:
            raise ValueError("quantization_group_size must be even")
        assert self.kvcache_block_size > 0
        assert self.kvcache_block_size % 256 == 0, (
            "kvcache_block_size must be a multiple of 256 for paged FlashAttention"
        )
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if getattr(self.hf_config, "model_type", None) == "minimind-v":
            if not self.vision_model or not os.path.isdir(self.vision_model):
                raise ValueError(
                    "MiniMindVLM requires a local vision_model directory"
                )
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
