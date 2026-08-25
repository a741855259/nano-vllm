from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """KV-aware scheduler with reference-style chunked prefill."""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Prefer prefills, bounded by a per-step token budget.

        As in nano-vllm, a request is only split when it is the first request in
        the prefill batch. This avoids fragmenting an otherwise short prompt
        merely to fill the tail of an existing batch.
        """
        prefill = []
        num_batched_tokens = 0
        while (
            self.waiting
            and len(prefill) < self.max_num_seqs
        ):
            seq = self.waiting[0]
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            if remaining_budget == 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = (
                    len(seq)
                    - num_cached_blocks * self.block_size
                )
            else:
                num_cached_blocks = 0
                num_tokens = len(seq) - seq.num_cached_tokens
            assert num_tokens > 0
            if remaining_budget < num_tokens and prefill:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining_budget)
            num_batched_tokens += seq.num_scheduled_tokens
            prefill.append(seq)

            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
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
                seq.num_scheduled_tokens = 1
                decode.append(seq)
        assert decode
        self.running.extendleft(reversed(decode))
        return decode, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ):
        for seq, token_id in zip(seqs, token_ids):
            # Publish only after the model has written the scheduled tokens'
            # K/V into their physical cache slots.
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < len(seq):
                continue

            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
