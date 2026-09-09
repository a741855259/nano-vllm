import hashlib

import torch
try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor

from nanovllm.multimodal.base import MultiModalProcessor
from nanovllm.multimodal.inputs import MultiModalFeature, PlaceholderRange


class MiniMindProcessorBase(MultiModalProcessor):
    """Shared image preparation and token layout for MiniMind processors."""

    def __init__(
        self, tokenizer, vision_model_path, config, max_num_images=1,
        image_processor=None,
    ):
        self.tokenizer = tokenizer
        self.image_token_id = config.image_ids[0]
        self.image_token = config.image_special_token
        self.image_token_len = config.image_token_len
        self.max_num_images = max_num_images
        self.image_processor = image_processor
        if self.image_processor is None:
            self.image_processor = SiglipImageProcessor.from_pretrained(
                vision_model_path, local_files_only=True,
            )

    @staticmethod
    def _normalize_items(items):
        if items is None:
            return []
        return list(items) if isinstance(items, (list, tuple)) else [items]

    @staticmethod
    def _validate_markers(prompt, marker, count, modality):
        if prompt.count(marker) != count:
            raise ValueError(
                f"prompt contains {prompt.count(marker)} {marker} markers but "
                f"received {count} {modality} inputs"
            )

    @staticmethod
    def _expand_markers(prompt, marker, token, lengths):
        for length in lengths:
            prompt = prompt.replace(marker, token * length, 1)
        return prompt

    def _prepare_image_prompt(self, prompt, images):
        if self.image_token in prompt:
            raise ValueError(
                f"prompt must use <image>, not pre-expanded {self.image_token}"
            )
        images = self._normalize_items(images)
        if len(images) > self.max_num_images:
            raise ValueError(
                f"received {len(images)} images; maximum is {self.max_num_images}"
            )
        self._validate_markers(prompt, "<image>", len(images), "image")
        prompt = self._expand_markers(
            prompt, "<image>", self.image_token,
            [self.image_token_len] * len(images),
        )
        return prompt, images

    @staticmethod
    def _hash_tensor(modality: str, tensor: torch.Tensor, *metadata) -> str:
        value = tensor.detach().cpu().contiguous()
        h = hashlib.sha256()
        h.update(modality.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        for item in metadata:
            h.update(str(item).encode())
        h.update(value.numpy().tobytes())
        return f"{modality}:{h.hexdigest()}"

    @staticmethod
    def _find_placeholders(
        token_ids: list[int],
        token_id: int,
        expected_lengths: list[int],
        modality: str,
    ) -> list[PlaceholderRange]:
        ranges = []
        i = 0
        item_idx = 0
        while i < len(token_ids):
            if token_ids[i] != token_id:
                i += 1
                continue
            start = i
            while i < len(token_ids) and token_ids[i] == token_id:
                i += 1
            run_length = i - start
            consumed = 0
            while consumed < run_length and item_idx < len(expected_lengths):
                expected = expected_lengths[item_idx]
                if consumed + expected > run_length:
                    break
                ranges.append(PlaceholderRange(start + consumed, expected))
                consumed += expected
                item_idx += 1
            if consumed != run_length:
                raise ValueError(
                    f"{modality} placeholder run has {run_length} tokens and "
                    "cannot be matched to the supplied inputs"
                )
        if item_idx != len(expected_lengths):
            raise ValueError(
                f"prompt contains {item_idx} {modality} placeholders but received "
                f"{len(expected_lengths)} {modality} inputs"
            )
        return ranges

    def _make_feature(self, modality, data, placeholder, tensor, *metadata):
        return MultiModalFeature(
            identifier=self._hash_tensor(modality, tensor, *metadata),
            placeholder=placeholder,
            modality=modality,
            data=data,
        )

    def _image_features(self, token_ids, images):
        ranges = self._find_placeholders(
            token_ids, self.image_token_id,
            [self.image_token_len] * len(images), "image",
        )
        if not images:
            return []
        pixels = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return [
            self._make_feature(
                "image", {"pixel_values": pixels[i:i + 1]},
                placeholder, pixels[i:i + 1],
            )
            for i, placeholder in enumerate(ranges)
        ]

