from __future__ import annotations

import torch
from torch import nn
from transformers import SiglipVisionModel

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.layers.fused_moe import FusedExperts
from nanovllm.layers.linear import ColumnParallelLinear
from nanovllm.layers.quantization import QuantizationConfig
from nanovllm.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3Model,
    _add_prefix,
)
from nanovllm.utils.context import get_context


class UnfusedExperts(nn.ModuleList):

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__([
            Qwen3MLP(
                config.hidden_size,
                config.moe_intermediate_size,
                config.hidden_act,
                quant_config,
                prefix=_add_prefix(prefix, f"experts.{expert_id}"),
            )
            for expert_id in range(config.num_experts)
        ])

    def _forward_prefill(
        self,
        x: torch.Tensor,
        routed_weights: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self):
            token_ids, topk_ids = torch.where(expert_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(x[token_ids])
            expert_weight = routed_weights[token_ids, topk_ids, None].to(x.dtype)
            # index_add_ writes experts back in token order and reduces Top-K.
            output.index_add_(0, token_ids, expert_output * expert_weight)
        return output

    def _forward_decode(
        self,
        x: torch.Tensor,
        routed_weights: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Static shapes keep the fallback compatible with CUDA Graph capture.
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self):
            expert_weight = (
                routed_weights * (expert_ids == expert_id)
            ).sum(dim=-1, keepdim=True)
            output.add_(expert(x) * expert_weight.to(x.dtype))
        return output

    def forward(
        self,
        x: torch.Tensor,
        routed_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        if is_prefill:
            return self._forward_prefill(x, routed_weights, expert_ids)
        return self._forward_decode(x, routed_weights, expert_ids)


class MiniMindMoE(nn.Module):

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        if not 0 < self.top_k <= config.num_experts:
            raise ValueError(
                "num_experts_per_tok must be between 1 and num_experts"
            )

        # Routers stay in the model dtype even when experts are quantized.
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=_add_prefix(prefix, "gate"),
        )
        if quant_config is None and self.top_k == 1:
            self.experts = FusedExperts(
                config.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
        else:
            self.experts = UnfusedExperts(config, quant_config, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        scores = torch.softmax(self.gate(x), dim=-1)
        # Top-K is token-local, so independently packed requests do not mix.
        routed_weights, expert_ids = torch.topk(
            scores,
            self.top_k,
            dim=-1,
            sorted=False,
        )
        if self.renormalize:
            routed_weights = routed_weights / (
                routed_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        output = self.experts(
            x,
            routed_weights,
            expert_ids,
            is_prefill=get_context().is_prefill,
        )
        return output.reshape(original_shape)


class MiniMindDecoderLayer(Qwen3DecoderLayer):

    @staticmethod
    def build_mlp(
        config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:
        if getattr(config, "use_moe", False):
            return MiniMindMoE(
                config,
                quant_config,
                prefix=prefix,
            )
        return Qwen3DecoderLayer.build_mlp(config, quant_config, prefix)

    @staticmethod
    def get_attention_bias(config) -> bool:
        return False


class MiniMindModel(Qwen3Model):

    decoder_layer_cls = MiniMindDecoderLayer

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)


class MMVisionProjector(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MiniMindVLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    supports_multimodal = True

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        vision_model_path: str | None = None,
    ):
        super().__init__()
        if vision_model_path is None:
            raise ValueError("vision_model_path is required for MiniMindVLM")
        self.config = config
        self.model = MiniMindModel(
            config,
            quant_config,
            prefix=_add_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.vision_proj = MMVisionProjector(
            config.image_hidden_size,
            config.hidden_size,
        )
        target_device = torch.get_default_device()
        torch.set_default_device("cpu")
        try:
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                vision_model_path,
                local_files_only=True,
                dtype=torch.float16,
            ).eval()
        finally:
            torch.set_default_device(target_device)
        self.vision_encoder.to(target_device)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    # 视觉编码整体流程:
    # 1.图片先经过ImageProcessor(SiglipImageProcessor)得到mm_features图片像素pixel_values(float32)
    # 2.再输入VisionEncoder(SiglipVisionModel)得到hidden_states 视觉模型输出(float16)
    # 3.再经过MMVisionProjector投影映射到语音Embd空间, 作为主LLM的Vision EmbdToken(bfloat16)
    def embed_multimodal(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_dtype = next(self.vision_encoder.parameters()).dtype  # float16
        outputs = self.vision_encoder(pixel_values=pixel_values.to(vision_dtype))
        features = outputs.last_hidden_state
        projector_dtype = self.vision_proj.mlp[0].weight.dtype  # bfloat16
        features = self.vision_proj(features.to(projector_dtype))
        expected = self.config.image_token_len
        if features.size(1) != expected:
            raise ValueError(
                f"vision encoder returned {features.size(1)} tokens; expected {expected}"
            )
        return features

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor] | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required with multimodal embeddings")
        # 把当前 chunk 中的视觉特征拼平
        mm_embeds = torch.cat(multimodal_embeddings, dim=0)
        # 检查数量, 必须满足: 视觉 embedding 数量 == mask 中 True 的数量
        expected = int(is_multimodal.sum().item())
        if mm_embeds.size(0) != expected:
            raise ValueError(
                f"received {mm_embeds.size(0)} image embeddings for "
                f"{expected} placeholders"
            )
        # 最后将占位Token替换为视觉EmbdToken
        inputs_embeds[is_multimodal] = mm_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
