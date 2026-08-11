import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model


class ModelRunner:
    """Builds the model on GPU (bf16), loads safetensors weights, and runs
    single-sequence generation. No KV cache, no tensor-parallel, no CUDA graphs,
    no continuous batching."""

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
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    @torch.inference_mode()
    def run(self, seq: Sequence) -> int:
        # Stateless: recompute over the full sequence on every step (no KV cache).
        input_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")
        positions = torch.arange(len(seq), dtype=torch.int64, device="cuda")
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states[-1:].contiguous())
        token_id = self.sampler(logits, seq.temperature)
        return token_id.item()
