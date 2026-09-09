from __future__ import annotations

import os

import numpy as np
import torch
from nanovllm.multimodal.inputs import MultiModalFeature
from nanovllm.multimodal.minimind_base import MiniMindProcessorBase


class MiniMindOProcessor(MiniMindProcessorBase):

    modality_arguments = {"image": "images", "audio": "audios"}

    def __init__(
        self,
        tokenizer,
        vision_model_path: str,
        config,
        audio_processor,
        max_num_images: int = 1,
        max_num_audios: int = 1,
        max_audio_feature_tokens: int = 1200,
        image_processor=None,
    ):
        super().__init__(
            tokenizer, vision_model_path, config, max_num_images, image_processor,
        )
        self.audio_token_id = config.audio_ids[0]
        self.audio_token = config.audio_special_token
        self.max_num_audios = max_num_audios
        self.max_audio_feature_tokens = max_audio_feature_tokens
        self.audio_processor = audio_processor

    @staticmethod
    def _normalize_audios(audios):
        if audios is None:
            return []
        if (
            isinstance(audios, tuple)
            and len(audios) == 2
            and isinstance(audios[1], (int, float))
        ):
            return [audios]
        return list(audios) if isinstance(audios, list) else [audios]

    @staticmethod
    def _load_audio(audio) -> tuple[np.ndarray, int]:
        if isinstance(audio, tuple) and len(audio) == 2:
            waveform, sample_rate = audio
            waveform = np.asarray(waveform, dtype=np.float32)
        elif isinstance(audio, np.ndarray):
            waveform = audio.astype(np.float32, copy=False)
            sample_rate = 16000
        elif isinstance(audio, (str, os.PathLike)):
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError(
                    "reading audio files requires the 'omni' optional dependencies"
                ) from exc
            waveform, sample_rate = sf.read(os.fspath(audio), dtype="float32")
        else:
            raise TypeError(
                "audio must be a numpy array, (waveform, sample_rate), or a path"
            )
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1:
            raise ValueError("audio waveform must be mono or channel-last stereo")
        if sample_rate != 16000:
            try:
                import librosa
            except ImportError as exc:
                raise RuntimeError(
                    "resampling audio requires the 'omni' optional dependencies"
                ) from exc
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=16000,
            )
            sample_rate = 16000
        return np.ascontiguousarray(waveform, dtype=np.float32), sample_rate

    def _process_audios(self, audios):
        if audios and self.audio_processor is None:
            raise RuntimeError("audio processor is unavailable")
        processed = []
        for audio in audios:
            waveform, sample_rate = self._load_audio(audio)
            result = self.audio_processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = result.input_features
            feature_length = int(result.attention_mask.sum().item())
            if feature_length <= 0:
                raise ValueError("audio produced no SenseVoice feature frames")
            if feature_length > self.max_audio_feature_tokens:
                raise ValueError(
                    f"audio produced {feature_length} feature tokens; maximum is "
                    f"{self.max_audio_feature_tokens}"
                )
            processed.append((input_features, feature_length))
        return processed

    def process(
        self,
        prompt: str,
        images=None,
        audios=None,
    ) -> tuple[list[int], list[MultiModalFeature]]:
        prompt, images = self._prepare_image_prompt(prompt, images)
        audios = self._normalize_audios(audios)
        if len(audios) > self.max_num_audios:
            raise ValueError(
                f"received {len(audios)} audios; maximum is {self.max_num_audios}"
            )
        self._validate_markers(prompt, "<audio>", len(audios), "audio")
        audio_data = self._process_audios(audios)
        lengths = [length for _, length in audio_data]
        prompt = self._expand_markers(
            prompt, "<audio>", self.audio_token, lengths,
        )
        # Locate both modalities only after every marker has been expanded.
        token_ids = self.tokenizer.encode(prompt)
        features = self._image_features(token_ids, images)
        audio_ranges = self._find_placeholders(
            token_ids, self.audio_token_id, lengths, "audio",
        )
        for (input_features, feature_length), placeholder in zip(
            audio_data, audio_ranges,
        ):
            features.append(self._make_feature(
                "audio",
                {
                    "input_features": input_features,
                    "feature_lengths": torch.tensor([feature_length]),
                },
                placeholder,
                input_features,
                feature_length,
            ))
        features.sort(key=lambda feature: feature.placeholder.offset)
        return token_ids, features
