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
    # 图片内容 hash，用于 encoder cache 和 Prefix Cache
    identifier: str
    # SigLIP processor 处理后的 [1, 3, 256, 256]
    pixel_values: torch.Tensor
    # 该图片对应的 64 个 token 在 prompt 中的位置
    placeholder: PlaceholderRange
