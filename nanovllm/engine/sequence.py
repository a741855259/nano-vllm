from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        mm_features=None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.last_token = token_ids[-1]
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.mm_features = sorted(
            mm_features or [],
            # 按图片在 prompt 中的位置排序
            # 当前ModelRunner实现会按顺序(batch 中 sequence 的顺序, 每个 sequence 中图片出现的顺序)收集视觉 embedding
            key=lambda feature: feature.placeholder.offset,
        )

    def __len__(self):
        return len(self.token_ids)

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return len(self) - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (len(self) + self.block_size - 1) // self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size:(i+1)*self.block_size]

    # 检查第 i 个 KV block 是否与图片范围重叠
    def block_extra_keys(self, i):
        start = i * self.block_size
        end = min(start + self.block_size, len(self))
        return tuple(
            (feature.identifier, feature.placeholder.offset - start)
            for feature in self.mm_features
            if feature.placeholder.overlaps(start, end)
        )

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
