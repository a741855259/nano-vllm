from __future__ import annotations

import torch
from torch import nn
from transformers import SiglipVisionModel

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.layers.quantization import QuantizationConfig
from nanovllm.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Model,
    _add_prefix,
)


class MiniMindDecoderLayer(Qwen3DecoderLayer):

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
        if getattr(config, "use_moe", False):
            raise NotImplementedError("MiniMind MoE checkpoints are not supported")
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
