from collections import deque

from nanovllm.engine.sequence import Sequence


class BlockManager:
    """Minimal physical-block allocator for serial sequence execution."""

    def __init__(self, num_blocks: int):
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.free_block_ids = deque(range(num_blocks))

    def ensure_capacity(self, seq: Sequence, num_blocks: int):
        num_new_blocks = num_blocks - len(seq.block_table)
        if num_new_blocks <= 0:
            return
        if num_new_blocks > len(self.free_block_ids):
            raise RuntimeError(
                f"KV cache exhausted: need {num_new_blocks} more blocks, "
                f"only {len(self.free_block_ids)} free"
            )
        for _ in range(num_new_blocks):
            seq.block_table.append(self.free_block_ids.popleft())

    def deallocate(self, seq: Sequence):
        self.free_block_ids.extend(seq.block_table)
        seq.block_table.clear()
        seq.num_cached_tokens = 0
