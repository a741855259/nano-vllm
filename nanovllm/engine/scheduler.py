from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """Minimal KV-aware scheduler without prefix caching or chunked prefill."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running and not self.finished

    def add(self, seq: Sequence):
        if self._finished(seq):
            seq.status = SequenceStatus.FINISHED
            self.finished.append(seq)
        else:
            self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Prefer new prefills; otherwise schedule one decode step per request."""
        prefill = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill.append(seq)
        if prefill:
            return prefill, True

        decode = []
        while self.running and len(decode) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                decode.append(seq)
        self.running.extendleft(reversed(decode))
        return decode, False

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def _finished(self, seq: Sequence):
        if seq.num_completion_tokens >= seq.max_tokens:
            return True
        if len(seq) >= self.max_model_len:
            return True
        return not seq.ignore_eos and seq.last_token == self.eos

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        assert len(seqs) == len(token_ids)
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens = len(seq)
            seq.append_token(token_id)
            if self._finished(seq):
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)
                self.block_manager.deallocate(seq)
                self.finished.append(seq)

    def pop_finished(self) -> list[Sequence]:
        finished = list(self.finished)
        self.finished.clear()
        return finished
