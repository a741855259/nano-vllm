from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class PlaceholderRange:
    # 表示一张图片在 prompt token 中占据的位置
    offset: int
    length: int

    # 判断一个 prefill chunk 是否与图片范围相交
    def overlaps(self, start: int, end: int) -> bool:
        return self.offset < end and self.offset + self.length > start


@dataclass
class MultiModalFeature:
    # 媒体内容 hash，用于 encoder cache 和 Prefix Cache
    identifier: str
    placeholder: PlaceholderRange
    data: dict[str, torch.Tensor]
    modality: str = "image"

    def __post_init__(self):
        if not self.data:
            raise ValueError("multimodal feature data cannot be empty")
