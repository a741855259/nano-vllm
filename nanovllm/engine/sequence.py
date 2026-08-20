from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """A single generation request and its logical-to-physical KV block table."""

    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.prompt_token_ids = list(token_ids)
        self.last_token = token_ids[-1]
        self.block_table: list[int] = []
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return len(self.token_ids)

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - len(self.prompt_token_ids)

    @property
    def completion_token_ids(self):
        return self.token_ids[len(self.prompt_token_ids):]
