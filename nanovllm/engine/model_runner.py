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
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(
            max_num_batched_tokens // seq_len,
            self.config.max_num_seqs,
        )
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_layers = hf_config.num_hidden_layers
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads,
        )
        block_bytes = (
            2
            * num_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - (peak - current))
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
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

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_num_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
            for seq in seqs
        ]
        # 1.pin_memory=False时：
        # 普通 CPU 内存
        #     ↓ 隐式复制
        # 临时 pinned 内存
        #     ↓ DMA
        # GPU 内存
        # 2.pin_memory作用：
        # Pinned CPU 内存
        #     ↓ DMA
        # GPU 内存
        return torch.tensor(
            block_tables,
            dtype=torch.int32,
            # ModelRunner.__init__() 构造模型时设置了：torch.set_default_device("cuda")
            # 可能受默认设备影响，尝试直接创建在 CUDA 上
            device="cpu",
            pin_memory=True,
        ).cuda(non_blocking=True)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            assert start < end <= len(seq)
            if seq.block_table:
                required_blocks = (end + self.block_size - 1) // self.block_size
                assert len(seq.block_table) >= required_blocks, (
                    "scheduler did not allocate enough KV-cache blocks"
                )
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if not seq.block_table:  # warmup runs before the paged cache exists
                continue
            for position in range(start, end):
                slot_mapping.append(
                    seq.block_table[position // self.block_size] * self.block_size
                    + position % self.block_size
                )

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            positions, dtype=torch.int64, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        set_context(
            True,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        return input_ids, positions, cu_seqlens_q

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            assert seq.num_cached_tokens == len(seq) - 1, (
                f"KV cache has {seq.num_cached_tokens} tokens, expected {len(seq) - 1}"
            )
            assert seq.num_scheduled_tokens == 1
            position = len(seq) - 1
            required_blocks = (len(seq) + self.block_size - 1) // self.block_size
            assert len(seq.block_table) >= required_blocks, (
                "scheduler did not allocate enough KV-cache blocks"
            )
            input_ids.append(seq.last_token)
            positions.append(position)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[position // self.block_size] * self.block_size
                + position % self.block_size
            )

        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        positions = torch.tensor(
            positions, dtype=torch.int64, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, device="cpu", pin_memory=True,
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        return temperatures.cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs[-1]:
            return self.model.compute_logits(self.model(input_ids, positions))

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
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run one packed prefill or decode batch in a single model forward."""
        assert seqs, "seqs must not be empty"
        if is_prefill:
            input_ids, positions, _ = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        try:
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = self.sampler(logits, temperatures).tolist()
        finally:
            reset_context()
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
