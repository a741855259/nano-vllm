from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    slot_mapping: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        slot_mapping,
        block_tables,
        context_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
