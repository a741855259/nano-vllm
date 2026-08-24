import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.attention import Attention
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """Builds the model on GPU (bf16), loads safetensors weights, and runs
    batches of sequences with a minimal paged KV cache."""

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
        self.enforce_eager = config.enforce_eager
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
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

    def _prepare_batch(self, seqs: list[Sequence], is_prefill: bool):
        input_ids = []
        positions = []
        slot_mapping = []
        query_lens = []
        key_lens = []
        context_lens = []
        for seq in seqs:
            if is_prefill:
                start = seq.num_cached_tokens
                end = start + seq.num_scheduled_tokens
                assert start < end <= len(seq)
                seq_input_ids = seq.token_ids[start:end]
                seq_positions = list(range(start, end))
            else:
                assert seq.num_cached_tokens == len(seq) - 1, (
                    f"KV cache has {seq.num_cached_tokens} tokens, expected {len(seq) - 1}"
                )
                assert seq.num_scheduled_tokens == 1
                start = len(seq) - 1
                end = len(seq)
                seq_input_ids = [seq.last_token]
                seq_positions = [start]

            required_blocks = (end + self.block_size - 1) // self.block_size
            assert len(seq.block_table) >= required_blocks, (
                "scheduler did not allocate enough KV-cache blocks"
            )

            input_ids.extend(seq_input_ids)
            positions.extend(seq_positions)
            query_lens.append(len(seq_input_ids))
            key_lens.append(end)
            context_lens.append(end)
            slot_mapping.extend(
                seq.block_table[position // self.block_size] * self.block_size
                + position % self.block_size
                for position in seq_positions
            )

        # Full prefill can attend directly to its packed K/V. A later chunk (or
        # decode) must follow the block table to include the cached prefix.
        use_paged_kv = not is_prefill or any(seq.num_cached_tokens > 0 for seq in seqs)
        block_tables = None
        if use_paged_kv:
            max_num_blocks = max(len(seq.block_table) for seq in seqs)
            block_tables = [
                seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
                for seq in seqs
            ]
        cu_seqlens_q = [0]
        for query_len in query_lens:
            cu_seqlens_q.append(cu_seqlens_q[-1] + query_len)
        cu_seqlens_k = [0]
        for key_len in key_lens:
            cu_seqlens_k.append(cu_seqlens_k[-1] + key_len)

        device = self.kv_cache.device
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        positions = torch.tensor(positions, dtype=torch.int64, device=device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
        if block_tables is not None:
            block_tables = torch.tensor(block_tables, dtype=torch.int32, device=device)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
        set_context(
            is_prefill,
            slot_mapping,
            block_tables,
            context_lens_tensor,
            cu_seqlens_q,
            cu_seqlens_k,
            max(query_lens),
            max(key_lens),
            self.block_size,
        )
        return input_ids, positions, cu_seqlens_q

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs[-1]:
            return self.model(input_ids, positions)

        bs = input_ids.size(0)
        context = get_context()
        graph_bs = next(x for x in self.graph_bs if x >= bs)
        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return graph_vars["outputs"][:bs]

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int | None]:
        """Run one packed prefill or decode batch in a single model forward."""
        assert seqs, "seqs must not be empty"
        input_ids, positions, cu_seqlens_q = self._prepare_batch(seqs, is_prefill)
        try:
            hidden_states = self.run_model(input_ids, positions, is_prefill)
        finally:
            reset_context()

        last_hidden_states = hidden_states[cu_seqlens_q[1:] - 1]
        sample_indices = [
            i
            for i, seq in enumerate(seqs)
            if not is_prefill
            or seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)
        ]
        token_ids: list[int | None] = [None] * len(seqs)
        if sample_indices:
            sample_hidden_states = last_hidden_states[sample_indices].contiguous()
            logits = self.model.compute_logits(sample_hidden_states)
            temperatures = torch.tensor(
                [seqs[i].temperature for i in sample_indices],
                dtype=torch.float32,
                device=self.kv_cache.device,
            )
            sampled = self.sampler(logits, temperatures).tolist()
            for i, token_id in zip(sample_indices, sampled):
                token_ids[i] = token_id
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
