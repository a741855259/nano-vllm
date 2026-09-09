from collections import OrderedDict

import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.quantization import get_quantization_config
from nanovllm.models.registry import resolve_model_class
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """Builds the model on GPU (bf16), loads checkpoint weights, and runs
    batches of sequences with a minimal paged KV cache."""

    def __init__(self, config: Config):
        self.config = config
        hf_config = config.hf_config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        model_cls = resolve_model_class(hf_config)
        quant_config = get_quantization_config(
            config.quantization,
            config.quantization_group_size,
        )
        model_kwargs = {"quant_config": quant_config}
        model_type = getattr(hf_config, "model_type", None)
        if model_type in {"minimind-v", "minimind-o"}:
            model_kwargs["vision_model_path"] = config.vision_model
        if model_type == "minimind-o":
            model_kwargs["audio_model_path"] = config.audio_model
        self.model = model_cls(hf_config, **model_kwargs)
        load_model(self.model, config.model)
        self.model.eval()
        self.encoder_cache = OrderedDict()
        self.max_encoder_cache_items = (
            config.max_num_seqs
            * (
                config.max_num_images_per_prompt
                + config.max_num_audios_per_prompt
            )
            * 2
        )
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
        if hasattr(getattr(self, "model", None), "vision_encoder"):
            image_size = self.model.vision_encoder.config.image_size
            dummy_pixels = torch.zeros(
                self.config.max_num_images_per_prompt,
                3,
                image_size,
                image_size,
                dtype=torch.float32,
            )
            self.model.embed_multimodal(
                modality="image",
                pixel_values=dummy_pixels,
            )
            del dummy_pixels
        if hasattr(getattr(self, "model", None), "audio_encoder"):
            audio_feature_tokens = min(
                self.config.max_audio_feature_tokens,
                self.config.max_num_batched_tokens,
                self.config.max_model_len,
            )
            dummy_features = torch.zeros(
                self.config.max_num_audios_per_prompt,
                audio_feature_tokens,
                560,
                dtype=torch.float32,
            )
            dummy_lengths = torch.full(
                (self.config.max_num_audios_per_prompt,),
                audio_feature_tokens,
                dtype=torch.long,
            )
            self.model.embed_multimodal(
                modality="audio",
                input_features=dummy_features,
                feature_lengths=dummy_lengths,
            )
            del dummy_features, dummy_lengths
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

    def _get_encoder_output(self, feature):
        cached = self.encoder_cache.get(feature.identifier)
        if cached is not None:
            self.encoder_cache.move_to_end(feature.identifier)
            return cached
        data = {
            key: value.cuda(non_blocking=True)
            for key, value in feature.data.items()
        }
        output = self.model.embed_multimodal(
            modality=feature.modality,
            **data,
        )[0].detach()
        self.encoder_cache[feature.identifier] = output
        self.encoder_cache.move_to_end(feature.identifier)
        while len(self.encoder_cache) > self.max_encoder_cache_items:
            self.encoder_cache.popitem(last=False)
        return output

    def _gather_multimodal_embeddings(self, seq, start: int, end: int):
        embeddings = []
        mask = [False] * (end - start)
        for feature in seq.mm_features:
            placeholder = feature.placeholder
            if not placeholder.overlaps(start, end):
                continue
            overlap_start = max(start, placeholder.offset)
            overlap_end = min(end, placeholder.offset + placeholder.length)
            embed_start = overlap_start - placeholder.offset
            embed_end = overlap_end - placeholder.offset
            encoder_output = self._get_encoder_output(feature)
            embeddings.append(encoder_output[embed_start:embed_end])
            for i in range(overlap_start - start, overlap_end - start):
                mask[i] = True
        return embeddings, mask

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        multimodal_embeddings = []
        is_multimodal = []
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
            chunk_embeddings, chunk_mask = self._gather_multimodal_embeddings(
                seq,
                start,
                end,
            )
            multimodal_embeddings.extend(chunk_embeddings)
            is_multimodal.extend(chunk_mask)
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
        is_multimodal = torch.tensor(
            is_multimodal,
            dtype=torch.bool,
            device="cpu",
            pin_memory=True,
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
        return input_ids, positions, (multimodal_embeddings, is_multimodal)

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
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if multimodal_embeddings:
            # 合并，文字embd向量 + 图像embd向量替换
            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            return self.model.compute_logits(
                self.model(None, positions, inputs_embeds=inputs_embeds),
            )
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
        multimodal_embeddings = None
        is_multimodal = None
        if is_prefill:
            input_ids, positions, multimodal_inputs = self.prepare_prefill(seqs)
            multimodal_embeddings, is_multimodal = multimodal_inputs
        else:
            input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        try:
            logits = self.run_model(
                input_ids,
                positions,
                is_prefill,
                multimodal_embeddings,
                is_multimodal,
            )
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
