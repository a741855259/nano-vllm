# my-nano-vllm

A minimal Qwen3 model-structure loader + inference engine, referencing
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

It loads Qwen3 weights from safetensors and runs greedy / temperature-sampled
generation on GPU — **without** any of nano-vllm's inference optimizations
(paged KV cache, prefix caching, continuous batching, tensor parallelism, CUDA
graphs, `torch.compile`, flash_attn, triton). The model itself is implemented
from scratch; `transformers` is used only for the tokenizer, the config, and the
chat template.

## Design

- **Attention:** `F.scaled_dot_product_attention`, **stateless — no KV cache**.
  The runner feeds the full token sequence on every step and recomputes
  attention over all of it (so `is_causal=True` always, `q_len == kv_len`).
  Simplest possible form: no paged cache, no carried state, no prefill/decode
  split. Trades speed (O(n²) per generation) for clarity.
- **Layout:** mirrors `nano-vllm/nanovllm/` (`engine/`, `layers/`, `models/`,
  `utils/`), dropping the optimization-only files (`block_manager`, `scheduler`,
  `context`).
- Single GPU, `tp_size == 1` (all `torch.distributed` machinery removed).

See `docs/superpowers/specs/` for the full design and `docs/superpowers/plans/`
for the implementation plan.

## Run

Dependencies are pre-installed in the shared uv env at `/root/Infer/.venv`
(torch 2.13+cu130, transformers, safetensors, tqdm).

```bash
# smoke test (greedy, 3 chat-templated prompts)
PYTHONPATH=. /root/Infer/.venv/bin/python main.py

# HF cross-check (our greedy argmax vs AutoModelForCausalLM)
PYTHONPATH=. /root/Infer/.venv/bin/python check.py

# unit tests
PYTHONPATH=. /root/Infer/.venv/bin/python tests/test_layers.py
PYTHONPATH=. /root/Infer/.venv/bin/python tests/test_attention.py
```

Point `model_path` in `main.py` / `check.py` at a Qwen3 checkpoint
(e.g. `/mnt/c/Downloads/Qwen3-0.6B`).

## Usage

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", max_model_len=4096)
sp = SamplingParams(temperature=0.0, max_tokens=256)   # temperature <= 0 -> greedy
outputs = llm.generate(["Hello."], sp)
print(outputs[0]["text"])
```
