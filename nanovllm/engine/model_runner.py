import torch

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.attention import Attention
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """Builds the model on GPU (bf16), loads safetensors weights, and runs
    single-sequence generation with a minimal paged KV cache."""

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
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks)
        self.allocate_kv_cache()
        self.active_seq: Sequence | None = None
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def allocate_kv_cache(self):
        """Allocate one cache pool and bind each attention layer to a view."""
        config = self.config
        hf_config = config.hf_config
        num_layers = hf_config.num_hidden_layers
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads,
        )
        self.kv_cache = torch.empty(
            2,
            num_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        attentions = [module for module in self.model.modules() if isinstance(module, Attention)]
        assert len(attentions) == num_layers
        for layer_id, attention in enumerate(attentions):
            attention.k_cache = self.kv_cache[0, layer_id]
            attention.v_cache = self.kv_cache[1, layer_id]

    def reset_cache(self):
        if self.active_seq is not None:
            self.block_manager.deallocate(self.active_seq)
        self.active_seq = None
        reset_context()

    def _ensure_blocks(self, seq: Sequence):
        num_blocks = (len(seq) + self.block_size - 1) // self.block_size
        self.block_manager.ensure_capacity(seq, num_blocks)

    def _slot_mapping(
        self, positions: torch.Tensor, seq: Sequence,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_table = torch.tensor(seq.block_table, dtype=torch.int64, device="cuda")
        logical_blocks = torch.div(positions, self.block_size, rounding_mode="floor")
        slots = block_table[logical_blocks] * self.block_size + positions.remainder(self.block_size)
        return slots, block_table

    @torch.inference_mode()
    def run(self, seq: Sequence) -> int:
        if self.active_seq is not seq:
            self.reset_cache()
            self.active_seq = seq

        is_prefill = not seq.block_table
        if is_prefill:
            input_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")
            positions = torch.arange(len(seq), dtype=torch.int64, device="cuda")
        else:
            # The token sampled after the previous call is the only token not
            # yet represented in the cache.
            assert seq.num_cached_tokens == len(seq) - 1, (
                f"KV cache has {seq.num_cached_tokens} tokens, expected {len(seq) - 1}"
            )
            input_ids = torch.tensor([seq.last_token], dtype=torch.int64, device="cuda")
            positions = torch.tensor([len(seq) - 1], dtype=torch.int64, device="cuda")

        self._ensure_blocks(seq)
        slot_mapping, block_table = self._slot_mapping(positions, seq)
        set_context(is_prefill, slot_mapping, block_table, len(seq), self.block_size)
        try:
            hidden_states = self.model(input_ids, positions)
        finally:
            reset_context()
        seq.num_cached_tokens = len(seq)
        # Prefill returns one hidden state per prompt token; only the final one
        # is sampled. Decode already has length one, so this works for both.
        logits = self.model.compute_logits(hidden_states[-1:].contiguous())
        token_id = self.sampler(logits, seq.temperature)
        return token_id.item()
