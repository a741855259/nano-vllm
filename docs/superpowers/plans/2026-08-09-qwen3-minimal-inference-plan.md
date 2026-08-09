# Implementation Plan — Minimal Qwen3 Load + Inference

**Spec:** `docs/superpowers/specs/2026-08-09-qwen3-minimal-inference-design.md`
**Reference:** `/root/Infer/nano-vllm`
**Target:** `/root/Infer/my-nano-vllm`
**Model:** Qwen3-0.6B at `/mnt/c/Downloads/Qwen3-0.6B`

## Principles

- Mirror nano-vllm's `nanovllm/` structure; copy a file then strip the
  optimization machinery (dist / compile / flash_attn / triton / context / paging),
  keeping names and signatures so the two read side by side.
- Single GPU, `tp_size=1`. No `torch.distributed` calls anywhere.
- Implement the model from scratch — `transformers` is used **only** for
  `AutoTokenizer`, `AutoConfig`, and the chat template (and the HF reference in
  `check.py`).
- Verify incrementally: unit-test pure-math layers against torch references, then
  the HF cross-check as the integration test, then the end-to-end smoke test.

## Steps

### Step 1 — Dependencies & project metadata
- Rewrite `pyproject.toml`: name `my-nano-vllm`, deps `torch`, `transformers`,
  `safetensors`, `tqdm`. (No flash-attn/triton/xxhash/numpy.)
- Install into a venv with `uv` (CUDA torch wheel). Verify `torch.cuda.is_available()`
  and bf16/SDPA support on the RTX 2050.
- **Verify:** `python -c "import torch,transformers,safetensors,tqdm; print(torch.cuda.is_available())"`

### Step 2 — Pure-math layers (with unit tests)
Write under `nanovllm/layers/`. Each drops `@torch.compile`; no dist.
- `activation.py` — `SiluAndMul`: `x,y = x.chunk(2,-1); F.silu(x)*y`.
- `layernorm.py` — `RMSNorm`: `forward(x)` and `forward(x, residual)` (fused add),
  upcast to float for variance, return to orig dtype. No `@torch.compile`.
- `rotary_embedding.py` — `RotaryEmbedding` + `get_rope` (lru_cache). `inv_freq`
  from `base`; `cos_sin_cache` registered non-persistent. `apply_rotary_emb` splits
  in half (rotate-half). No `@torch.compile`.
- **Tests** (`tests/test_layers.py`): `SiluAndMul` vs manual silu*gate; `RMSNorm`
  vs `torch.nn.functional.rms_norm`; RoPE vs HF
  `Qwen3RotaryEmbedding`/`apply_rotary_emb` reference on a small random input.
- **Verify:** `pytest tests/test_layers.py -q` passes.

### Step 3 — Linear & embedding layers (single-GPU)
- `linear.py` — keep `LinearBase`, `ReplicatedLinear`, `ColumnParallelLinear`,
  `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear`. Remove
  all `dist.*` calls; `tp_size=1`, `tp_rank=0`. `weight_loader` copies the full
  tensor (no `narrow`/`chunk`). `RowParallelLinear.forward` drops the all_reduce.
- `embed_head.py` — `VocabParallelEmbedding` + `ParallelLMHead`. Remove dist and
  the `get_context()` dependency. `ParallelLMHead.forward(x)` just does
  `F.linear(x, self.weight)` (no last-token slicing here — the runner slices).
- **Verify:** import smoke — instantiate `QKVParallelLinear(1024,128,16,8)` and
  `MergedColumnParallelLinear(1024,[3072,3072])`, check output shapes.

### Step 4 — Attention (the core simplification)
- `attention.py` — `Attention(num_heads, head_dim, scale, num_kv_heads)`.
  Owns `k_cache`/`v_cache` (None until first use). `forward(q,k,v)` appends k/v
  and runs `F.scaled_dot_product_attention` with `is_causal=True`, repeating kv
  heads for GQA. Shapes: q/k/v in as `[N,H,D]`/`[N,kvH,D]`; SDPA wants
  `[1,H,N,D]` → repeat kv heads → `[1,H,N,D]`. Output flattened back to
  `[N, H*D]`.
- No `context.py`, no triton kernel, no flash_attn.
- **Test:** build a 1-layer toy attention, run prefill (q=[L]) then 3 decode steps
  (q=[1]), assert output length grows and causal masking holds (a later token's
  output doesn't depend on a not-yet-seen key — flip a future key, output
  unchanged).
- **Verify:** `pytest tests/test_attention.py -q` passes.

### Step 5 — Model
- `models/qwen3.py` — copy reference; remove `import torch.distributed as dist`.
  `Qwen3Attention.__init__` no longer calls `dist.get_world_size()` (tp=1). Keep
  q_norm/k_norm (Qwen3-0.6B has `attention_bias=false`). Keep
  `packed_modules_mapping`. Signatures `(positions, hidden_states)` unchanged.
- **Verify:** instantiate `Qwen3ForCausalLM(hf_config)` from the real
  `config.json`, forward random `input_ids`/`positions`, check logits shape
  `[L, vocab_size]`.

### Step 6 — Weight loader
- `utils/loader.py` — copy reference; `default_weight_loader` + `load_model`.
  Iterates `*.safetensors`, uses `packed_modules_mapping` to route q/k/v→qkv_proj
  and gate/up→gate_up_proj via each param's `weight_loader`. No TP sharding (the
  single-GPU weight_loaders already copy full tensors).
- **Verify:** load real weights into the model; assert a known weight
  (e.g. `model.model.embed_tokens.weight[0]`) matches the safetensors value.

### Step 7 — Config & sampling params
- `config.py` — `Config(model, max_model_len=4096, eos=-1)`. `__post_init__`:
  `AutoConfig.from_pretrained(model)`, `max_model_len =
  min(max_model_len, hf_config.max_position_embeddings)`. No assert on
  `kvcache_block_size`, no TP.
- `sampling_params.py` — `SamplingParams(temperature=1.0, max_tokens=64,
  ignore_eos=False)`. `temperature <= 0` allowed → greedy (drop the
  `assert temperature > 1e-10`).
- `sampler.py` — `Sampler`: greedy argmax if temperature≤0, else temperature
  sampling (reference's exponential-then-argmax trick). No `@torch.compile`.
- **Verify:** `Config("/mnt/c/Downloads/Qwen3-0.6B")` loads; `max_model_len` ≤ 40960.

### Step 8 — Engine
- `engine/sequence.py` — trimmed `Sequence`: `seq_id`, `token_ids`,
  `prompt_token_ids`, `status`, `temperature`, `max_tokens`, `ignore_eos`,
  `append_token()`, `is_finished`, `num_completion_tokens`. No block table /
  caching / pickle.
- `engine/model_runner.py` — `ModelRunner(config)`: sets dtype=bf16, device=cuda,
  builds `Qwen3ForCausalLM(hf_config)`, `load_model`. `reset_kv_cache()` walks
  `model.modules()` setting each `Attention`'s `k_cache=v_cache=None`.
  `prefill(seq)` → `run(seq, is_prefill=True)`; `decode(seq)` →
  `run(seq, is_prefill=False)`; both `@torch.inference_mode()`. `run`:
  `input_ids`/`positions` → `model(...)` → slice last-token hidden →
  `compute_logits` → `sampler(logits, temperature)` → token id.
- `engine/llm_engine.py` — `LLMEngine(model, **kwargs)`: build `Config`,
  `AutoTokenizer`, `ModelRunner`; `config.eos = tokenizer.eos_token_id`.
  `add_request`, `generate(prompts, sampling_params)` runs each seq prefill→decode
  until EOS or `max_tokens` or `len(seq) == max_model_len`; returns
  `[{text, token_ids}, ...]` in input order. Simple progress print (no tqdm
  throughput metrics required; tqdm optional for progress bar).
- `llm.py` — `LLM(LLMEngine): pass`. `__init__.py` — export `LLM, SamplingParams`.
- **Verify:** import `nanovllm`; construct `LLM(model_path)`; print param count.

### Step 9 — Entry points
- `main.py` — replace broken version. Load Qwen3-0.6B, build `LLM`, apply chat
  template to 3 prompts, `generate` (greedy, `max_tokens=256`), print
  `Prompt`/`Completion`.
- `check.py` — extend to cross-check: for one prompt, compare our model's greedy
  next-token argmax vs HF `AutoModelForCausalLM`'s argmax (top-1 match, or
  top-k overlap). Confirms weight loading correctness.

### Step 10 — End-to-end verification
- Run `python main.py`. Expect coherent Qwen3 completions stopping at EOS.
- Run `python check.py`. Expect our argmax to match HF's argmax on the prompt.
- Fix any divergences (typical: RoPE convention, q/k norm placement, weight
  name routing in loader, dtype).
- **Done when:** smoke test prints coherent text AND `check.py` matches HF.

## Files created/modified (summary)
- modify: `pyproject.toml`, `main.py`, `check.py`
- create: `nanovllm/__init__.py`, `nanovllm/llm.py`, `nanovllm/config.py`,
  `nanovllm/sampling_params.py`, `nanovllm/engine/{sequence,llm_engine,model_runner}.py`,
  `nanovllm/layers/{activation,layernorm,rotary_embedding,linear,embed_head,sampler,attention}.py`,
  `nanovllm/models/qwen3.py`, `nanovllm/utils/loader.py`,
  `tests/test_layers.py`, `tests/test_attention.py`

## Risks
- CUDA torch wheel install on WSL2 + RTX 2050 (cc 8.6). Mitigation: use the
  official `cu121`/`cu124` wheel via `uv pip`.
- RoPE rotate-half vs HF convention mismatch → silent quality degradation. The
  `check.py` HF cross-check catches this.
- 4 GB VRAM: keep `max_tokens` modest; single-sequence, no batching, so fits.
