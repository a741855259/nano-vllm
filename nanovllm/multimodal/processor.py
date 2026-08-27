import hashlib

import torch
try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor

from nanovllm.multimodal.inputs import MultiModalFeature, PlaceholderRange

# MiniMindVProcessor类似 tokenizer 前的文本规范化
# SiglipVisionModel类似文本 embedding + Transformer encoder
class MiniMindVProcessor:

    def __init__(
        self,
        tokenizer,
        vision_model_path: str,
        config,
        max_num_images: int = 1,
    ):
        self.tokenizer = tokenizer
        self.image_token_id = config.image_ids[0]
        self.image_token = config.image_special_token
        self.image_token_len = config.image_token_len
        self.max_num_images = max_num_images
        self.image_processor = SiglipImageProcessor.from_pretrained(
            vision_model_path,
            local_files_only=True,
        )

    @staticmethod
    def _normalize_images(images):
        if images is None:
            return []
        return images if isinstance(images, (list, tuple)) else [images]

    @staticmethod
    def _hash_pixels(pixel_values: torch.Tensor) -> str:
        pixels = pixel_values.detach().cpu().contiguous()
        h = hashlib.sha256()
        h.update(str(pixels.dtype).encode())
        h.update(str(tuple(pixels.shape)).encode())
        h.update(pixels.numpy().tobytes())
        return h.hexdigest()

    def _find_placeholders(self, token_ids: list[int]) -> list[PlaceholderRange]:
        ranges = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] != self.image_token_id:
                i += 1
                continue
            start = i
            while i < len(token_ids) and token_ids[i] == self.image_token_id:
                i += 1
            length = i - start
            if length % self.image_token_len:
                raise ValueError(
                    f"image placeholder run has {length} tokens; expected a "
                    f"multiple of {self.image_token_len}"
                )
            ranges.extend(
                PlaceholderRange(start + j, self.image_token_len)
                for j in range(0, length, self.image_token_len)
            )
        return ranges

    def process(self, prompt: str, images) -> tuple[list[int], list[MultiModalFeature]]:
        images = self._normalize_images(images)
        if len(images) > self.max_num_images:
            raise ValueError(
                f"received {len(images)} images; maximum is {self.max_num_images}"
            )
        marker_count = prompt.count("<image>")
        # image使用占位Token
        if marker_count:
            if marker_count != len(images):
                raise ValueError(
                    f"prompt contains {marker_count} <image> markers but received "
                    f"{len(images)} images"
                )
            prompt = prompt.replace(
                "<image>",
                self.image_token * self.image_token_len,
            )

        token_ids = self.tokenizer.encode(prompt)
        # 查找所有图片占位范围
        placeholders = self._find_placeholders(token_ids)
        if len(placeholders) != len(images):
            raise ValueError(
                f"prompt contains {len(placeholders)} image placeholder groups but "
                f"received {len(images)} images"
            )
        if not images:
            return token_ids, []

        # 图像预处理，无可训练参数
        processed = self.image_processor(images=list(images), return_tensors="pt")
        # SiglipImageProcessor 输出形状 [B, 3, 256, 256]
        pixel_values = processed["pixel_values"]
        features = [
            MultiModalFeature(
                # 图片hash: 基于处理后的像素
                identifier=self._hash_pixels(pixel_values[i]),
                pixel_values=pixel_values[i : i + 1],
                placeholder=placeholder,
            )
            for i, placeholder in enumerate(placeholders)
        ]
        return token_ids, features
