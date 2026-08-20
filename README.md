# my-nano-vllm

A minimal Qwen3 model-structure loader + inference engine, referencing
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

It loads Qwen3 weights from safetensors and runs greedy / temperature-sampled
generation on GPU. It includes a minimal paged KV cache and request scheduler,
plus packed prefill, block-level prefix caching, and batched decode. It does
not include nano-vllm's other inference optimizations (tensor parallelism,
CUDA graphs, `torch.compile`, or Triton KV-cache writes).
The model itself is implemented from scratch; `transformers` is used only for
the tokenizer, the config, and the chat template.

## Design

- **Attention:** a preallocated, per-layer paged KV cache with FlashAttention.
  Full prefill uses packed Q/K/V with `flash_attn_varlen_func`; decode uses
  `flash_attn_with_kvcache` to consume block tables directly. KV writes still
  use PyTorch `index_copy_`. Prefill is chunked by a per-step token budget.
- **Layout:** mirrors `nano-vllm/nanovllm/` (`engine/`, `layers/`, `models/`,
  `utils/`). `block_manager` and `context` support paged caching. The minimal
  `scheduler` owns KV-block allocation, manages waiting/running requests,
  prioritizes prefill, supports reference-style chunked prefill, shares
  immutable full prefix blocks by chained xxHash64 token hashes, and preempts
  on block pressure.
- Single GPU, `tp_size == 1` (all `torch.distributed` machinery removed).

See `docs/superpowers/specs/` for the full design and `docs/superpowers/plans/`
for the implementation plan.

## Run

Install the locked dependencies with `uv sync`. FlashAttention requires either
a wheel matching the Python, PyTorch, CUDA, and C++ ABI versions or a local CUDA
toolkit with `nvcc` for a source build.

```bash
# smoke test (greedy, 3 chat-templated prompts)
PYTHONPATH=. /root/Infer/.venv/bin/python main.py

# HF cross-check (our greedy argmax vs AutoModelForCausalLM)
PYTHONPATH=. /root/Infer/.venv/bin/python check.py

# unit tests
PYTHONPATH=. /root/Infer/.venv/bin/python -m unittest discover -s tests -v
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
