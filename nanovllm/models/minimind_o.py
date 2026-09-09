from __future__ import annotations

import contextlib
import io
import logging
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import SiglipVisionModel, logging as hf_logging

from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.layers.quantization import QuantizationConfig
from nanovllm.models.minimind_v import MMVisionProjector, MiniMindModel
from nanovllm.models.qwen3 import _add_prefix


class MMAudioProjector(nn.Module):

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


class SenseVoiceAudioProcessor:

    def __init__(self, frontend):
        self.frontend = frontend

    def __call__(
        self,
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        **kwargs,
    ):
        del sampling_rate, return_tensors, return_attention_mask, kwargs
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            features, feature_lengths = self.frontend(
                waveform,
                torch.tensor([waveform.size(1)]),
            )
        mask = torch.arange(features.size(1)) < feature_lengths[0]
        return SimpleNamespace(
            input_features=features,
            attention_mask=mask.long().unsqueeze(0),
        )


class MiniMindOmniThinker(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # 先不加载talker模块
    ignored_weight_prefixes = ("talker.",)
    supports_multimodal = True

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        vision_model_path: str | None = None,
        audio_model_path: str | None = None,
    ):
        super().__init__()
        if vision_model_path is None:
            raise ValueError("vision_model_path is required for MiniMindOmniThinker")
        if audio_model_path is None:
            raise ValueError("audio_model_path is required for MiniMindOmniThinker")
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
        self.audio_proj = MMAudioProjector(
            config.audio_hidden_size,
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
            self.audio_encoder, self.audio_processor = self._load_sensevoice(
                audio_model_path,
            )
        finally:
            torch.set_default_device(target_device)
        self.vision_encoder.to(target_device)
        self.audio_encoder.to(target_device)
        for encoder in (self.vision_encoder, self.audio_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _load_sensevoice(path):
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "MiniMind-O could not import FunASR or its dependencies: "
                f"{exc}. Install audio dependencies with `uv sync --extra omni`."
            ) from exc
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()):
            wrapped = AutoModel(
                model=path,
                trust_remote_code=True,
                disable_update=True,
                device="cpu",
            )
        encoder = wrapped.model.encoder.eval().float()
        frontend = wrapped.kwargs["frontend"].eval()
        return encoder, SenseVoiceAudioProcessor(frontend)

    def _embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder_dtype = next(self.vision_encoder.parameters()).dtype
        outputs = self.vision_encoder(
            pixel_values=pixel_values.to(encoder_dtype),
        )
        features = outputs.last_hidden_state
        projector_dtype = self.vision_proj.mlp[0].weight.dtype
        features = self.vision_proj(features.to(projector_dtype))
        if features.size(1) != self.config.image_token_len:
            raise ValueError(
                f"vision encoder returned {features.size(1)} tokens; expected "
                f"{self.config.image_token_len}"
            )
        return features

    def _embed_audio(
        self,
        input_features: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> torch.Tensor:
        encoder_dtype = next(self.audio_encoder.parameters()).dtype
        features, _ = self.audio_encoder(
            input_features.to(encoder_dtype),
            feature_lengths,
        )
        expected = int(feature_lengths[0].item())
        if features.size(1) < expected:
            raise ValueError(
                f"audio encoder returned {features.size(1)} tokens; expected at "
                f"least {expected}"
            )
        projector_dtype = self.audio_proj.mlp[0].weight.dtype
        return self.audio_proj(features[:, :expected].to(projector_dtype))

    def embed_multimodal(self, modality: str, **data) -> torch.Tensor:
        if modality == "image":
            return self._embed_image(data["pixel_values"])
        if modality == "audio":
            return self._embed_audio(
                data["input_features"],
                data["feature_lengths"],
            )
        raise ValueError(f"unsupported MiniMind-O modality: {modality}")

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
        embeddings = torch.cat(multimodal_embeddings, dim=0)
        expected = int(is_multimodal.sum().item())
        if embeddings.size(0) != expected:
            raise ValueError(
                f"received {embeddings.size(0)} multimodal embeddings for "
                f"{expected} placeholders"
            )
        inputs_embeds[is_multimodal] = embeddings.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
