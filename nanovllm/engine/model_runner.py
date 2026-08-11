import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.attention import Attention
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model


class ModelRunner:
    """Builds the model on GPU (bf16), loads safetensors weights, and runs
    single-sequence generation with a minimal contiguous KV cache."""

    def __init__(self, config: Config):
        self.config = config
        hf_config = config.hf_config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.model.eval()
        self.sampler = Sampler()
        self.active_seq_id: int | None = None
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def reset_cache(self):
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.reset_cache()
        self.active_seq_id = None

    def _cache_len(self) -> int:
        for module in self.model.modules():
            if isinstance(module, Attention):
                return module.cache_len
        return 0

    @torch.inference_mode()
    def run(self, seq: Sequence) -> int:
        if self.active_seq_id != seq.seq_id:
            self.reset_cache()
            self.active_seq_id = seq.seq_id

        cache_len = self._cache_len()
        if cache_len == 0:
            input_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")
            positions = torch.arange(len(seq), dtype=torch.int64, device="cuda")
        else:
            # The token sampled after the previous call is the only token not
            # yet represented in the cache.
            assert cache_len == len(seq) - 1, (
                f"KV cache has {cache_len} tokens, expected {len(seq) - 1}"
            )
            input_ids = torch.tensor([seq.last_token], dtype=torch.int64, device="cuda")
            positions = torch.tensor([len(seq) - 1], dtype=torch.int64, device="cuda")

        hidden_states = self.model(input_ids, positions)
        # Prefill returns one hidden state per prompt token; only the final one
        # is sampled. Decode already has length one, so this works for both.
        logits = self.model.compute_logits(hidden_states[-1:].contiguous())
        token_id = self.sampler(logits, seq.temperature)
        return token_id.item()
