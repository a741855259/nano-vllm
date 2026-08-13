from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    slot_mapping: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    context_len: int = 0
    block_size: int = 0


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, slot_mapping, block_table, context_len, block_size)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
