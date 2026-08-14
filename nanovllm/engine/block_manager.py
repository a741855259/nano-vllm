from collections import deque

from nanovllm.engine.sequence import Sequence


class BlockManager:
    """Physical KV-block allocator owned by the scheduler."""

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.total_blocks = num_blocks
        self.block_size = block_size
        self.free_block_ids = deque(range(num_blocks))

    def num_blocks(self, seq: Sequence):
        return (len(seq) + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence):
        return len(self.free_block_ids) >= self.num_blocks(seq)

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        assert self.can_allocate(seq)
        for _ in range(self.num_blocks(seq)):
            seq.block_table.append(self.free_block_ids.popleft())

    def can_append(self, seq: Sequence):
        needs_block = len(seq) % self.block_size == 1
        return not needs_block or bool(self.free_block_ids)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            assert self.free_block_ids
            seq.block_table.append(self.free_block_ids.popleft())

    def deallocate(self, seq: Sequence):
        self.free_block_ids.extend(seq.block_table)
        seq.block_table.clear()
        seq.num_cached_tokens = 0
