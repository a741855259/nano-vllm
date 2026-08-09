# Minimal Qwen3 Load + Inference — Design

**Date:** 2026-08-09
**Reference project:** `/root/Infer/nano-vllm` (GeeeekExplorer/nano-vllm)
**Target project:** `/root/Infer/my-nano-vllm`
**Model under test:** Qwen3-0.6B at `/mnt/c/Downloads/Qwen3-0.6B`

## Goal

Implement a minimal Qwen3 model-structure loader and inference engine in
`my-nano-vllm`, referencing `nano-vllm`. Load Qwen3-0.6B weights from
safetensors, run greedy/temperature-sampled generation to completion, and print
results. **No inference optimizations are required** — drop everything whose only
purpose is speed or batching.

## Out of scope (explicitly dropped from the reference)

Paged KV cache + `BlockManager`; prefix caching (xxhash); continuous-batch
`Scheduler`; chunked prefill; tensor parallelism (`torch.distributed` / nccl /
multiprocessing / shared memory); CUDA graphs; `torch.compile`; flash_attn;
triton; the triton KV-store kernel. Corresponding files `engine/block_manager.py`,
`engine/scheduler.py`, `utils/context.py` are not created.

## Decisions (confirmed with user)

1. **Runtime: GPU + plain torch.** Run on the 4 GB RTX 2050 with stock torch
   `scaled_dot_product_attention` (SDPA). No flash_attn, no triton. The 0.6B bf16
   model (~1.5 GB) fits with room for a per-sequence KV cache.
2. **Layout: mirror nano-vllm's `nanovllm/` subdir structure**, dropping only the
   optimization-only files. Lets the two codebases be read side by side.

## Architecture

### File tree

```
my-nano-vllm/
  main.py                      # entry: load model, run 3 chat-templated prompts
  pyproject.toml               # deps: torch(CUDA), transformers, safetensors, tqdm
  nanovllm/
    __init__.py                # exports LLM, SamplingParams
    config.py                  # Config dataclass — reads HF config.json via AutoConfig
    llm.py                     # LLM(LLMEngine)
    sampling_params.py         # SamplingParams (temperature, max_tokens, ignore_eos)
    engine/
      sequence.py              # trimmed Sequence (no block_table / caching fields)
      llm_engine.py            # LLMEngine: tokenizer, generate() loop (no mp/TP/scheduler)
      model_runner.py          # ModelRunner: load, prefill/decode, sample (no dist/cudagraph)
    layers/
      attention.py             # SDPA + append-and-attend KV cache (no flash_attn/triton/context)
      linear.py                # QKVParallelLinear etc., single-GPU (no dist)
      layernorm.py             # RMSNorm (no @torch.compile)
      rotary_embedding.py      # RoPE (no @torch.compile)
      activation.py            # SiluAndMul (no @torch.compile)
      embed_head.py            # VocabParallelEmbedding / ParallelLMHead (no dist/context)
      sampler.py               # Sampler (no @torch.compile)
    models/
      qwen3.py                 # Qwen3ForCausalLM — near-identical to reference
    utils/
      loader.py                # load_model from safetensors (no TP sharding)
```

### Model (`models/qwen3.py`) — faithful to the reference

`Qwen3Attention` / `Qwen3MLP` / `Qwen3DecoderLayer` / `Qwen3Model` /
`Qwen3ForCausalLM` keep the reference structure: GQA with `q_norm`/`k_norm`
(active because Qwen3-0.6B has `attention_bias=false`), SwiGLU MLP, RMSNorm with
fused residual add, RoPE, tied `lm_head`. `packed_modules_mapping` is retained so
the loader fuses q/k/v → `qkv_proj` and gate/up → `gate_up_proj`. Layer forward
signatures stay `(positions, hidden_states)` — `positions` feeds RoPE only.

### Attention (`layers/attention.py`) — the core simplification

Each `Attention` layer owns its own `k_cache`/`v_cache`. On each step the new k/v
is appended and SDPA runs over the whole cache with `is_causal=True`:

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads, self.head_dim, self.scale, self.num_kv_heads = ...
        self.k_cache = None   # None per-sequence; reset between sequences
        self.v_cache = None

    def forward(self, q, k, v):                 # q:[N,H,D] k,v:[N,kvH,D]
        if self.k_cache is None:
            self.k_cache, self.v_cache = k, v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=0)
            self.v_cache = torch.cat([self.v_cache, v], dim=0)
        # reshape to [1, H, N, D], repeat kv heads for GQA, SDPA causal
        o = F.scaled_dot_product_attention(q_, self.k_cache_, self.v_cache_,
                                           scale=self.scale, is_causal=True)
        return o.flatten(1, -1)
```

`is_causal=True` is correct in both prefill (q=[L], kv=[L]) and decode (q=[1],
kv=[cur_len]; the single last-position query attends to all prior tokens).

**No `utils/context.py`** — the global `Context` existed only to pass
flash_attn/paged-cache metadata, which is no longer used. `ParallelLMHead` drops
its context dependency: the runner slices hidden states to the last token before
calling `compute_logits`.

### Linear / Embedding (`layers/linear.py`, `layers/embed_head.py`)

Keep `QKVParallelLinear` / `MergedColumnParallelLinear` / `RowParallelLinear` /
`VocabParallelEmbedding` / `ParallelLMHead` so `qwen3.py` mirrors the reference,
but with `tp_size=1` (no `torch.distributed`): `weight_loader` copies the full
tensor, no sharding, no all-reduce. `packed_modules_mapping` still routes q/k/v
and gate/up into the merged weights.

### Engine (`engine/`)

- **`Sequence`** (trimmed): `seq_id`, `token_ids`, `prompt_token_ids`, `status`,
  `temperature`, `max_tokens`, `ignore_eos`, `append_token()`. Dropped:
  `block_table`, `num_cached_tokens`, `num_scheduled_tokens`, `is_prefill`,
  block/pickle machinery.
- **`ModelRunner`**: `load_model` (safetensors → params via `weight_loader`);
  `reset_kv_cache()` (walks `model.modules()` setting each `Attention`'s
  `k_cache`/`v_cache` to `None`); `prefill(seq)` (input_ids=prompt,
  positions=`0..L-1`, forward → last-token logits → sample → 1st token);
  `decode(seq)` (input_ids=`[last]`, positions=`[len-1]`, forward → sample → next
  token). All under `@torch.inference_mode()`, bf16, on `cuda`.
- **`LLMEngine`**: holds tokenizer + ModelRunner. `generate(prompts,
  sampling_params)` encodes each prompt (string or token-id list), runs
  prefill→decode loop per sequence until EOS or `max_tokens`, decodes token ids to
  text, returns `[{text, token_ids}, ...]` in input order.

### Config & sampling (`config.py`, `sampling_params.py`)

- **`Config`** reads `AutoConfig.from_pretrained(model_path)` — **no hardcoded
  magic numbers** (removes bugs in the existing `main.py`, e.g.
  `max_model_length=128 < max_tokens`). Fields: `model`, `max_model_len`,
  `eos`. Dropped: `tensor_parallel_size`, `kvcache_block_size`,
  `num_kvcache_blocks`, `enforce_eager`, `gpu_memory_utilization` (the latter
  only fed the paged-cache block-count calculation, which is gone; per-sequence
  KV cache grows as used and is bounded by `max_model_len`, no pre-allocation).
  `max_model_len` defaults to HF `max_position_embeddings` and is a **hard cap**:
  the RoPE `cos_sin_cache` is sized to it, so positions must stay `< max_model_len`.
  Generation stops at EOS **or** `max_tokens` completion tokens **or** when total
  length reaches `max_model_len` (whichever first).
- **`SamplingParams`**: `temperature`, `max_tokens`, `ignore_eos`.
  `temperature <= 0` → greedy argmax (useful for reproducible smoke tests);
  otherwise temperature sampling like the reference `Sampler`.

### Entry point (`main.py`)

Load Qwen3-0.6B from `/mnt/c/Downloads/Qwen3-0.6B`, build `LLM`, apply the Qwen
chat template to 3 prompts, `generate`, print completions. Replaces the current
broken `main.py` (which imports a non-existent `myvllm.*` and a hardcoded config
dict).

## Dependencies

`pyproject.toml` deps: `torch` (CUDA wheel), `transformers`, `safetensors`,
`tqdm`. Dropped: `flash-attn`, `triton`, `xxhash`, `numpy`. RTX 2050 is Ampere
(cc 8.6) — supports bf16 and SDPA's flash/mem-efficient backend. Install via the
available `uv` (or `pip`) and run a smoke test against the local model.

## Verification

1. **Smoke test** (`main.py`): load Qwen3-0.6B, run 3 chat-templated prompts,
   print completions. Greedy decode of "introduce yourself" produces coherent
   text and stops at EOS.
2. **Weight-loading correctness** (`check.py`): cross-check the next-token
   logits / argmax for one prompt against HF `AutoModelForCausalLM` (the current
   `check.py` already loads the HF model) to confirm safetensors weights load
   correctly into the custom `Qwen3ForCausalLM`.

## Risk / constraints

- 4 GB VRAM: model ~1.5 GB bf16 + per-sequence KV cache. Per-sequence cache grows
  with `max_model_len`; keep `max_tokens` modest (e.g. 256). No batching, so peak
  memory is single-sequence — fits comfortably.
- `transformers` is needed only for `AutoTokenizer`, `AutoConfig`, and the chat
  template (and the HF reference in `check.py`). The model itself is implemented
  from scratch — it does **not** use `AutoModelForCausalLM` for inference.
