from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    block_size: int = 0


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        slot_mapping,
        block_tables,
        context_lens,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        block_size,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
