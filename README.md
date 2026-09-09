# my-nano-vllm

A minimal Qwen3 model-structure loader + inference engine, referencing
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

It loads Qwen3 weights from safetensors and runs greedy / temperature-sampled
generation on GPU. It includes a minimal paged KV cache and request scheduler,
plus packed prefill, block-level prefix caching, batched decode, and Triton
KV-cache writes and CUDA Graph decode. It does not include nano-vllm's other
inference optimization for tensor parallelism. Core pointwise operators use
`torch.compile`. Transformer linear layers can optionally be quantized online
to symmetric group-wise INT4 weights while keeping activations in BF16.
The model itself is implemented from scratch; `transformers` is used only for
the tokenizer, the config, and the chat template.

## Design

- **Attention:** a preallocated, per-layer paged KV cache with FlashAttention.
  Full prefill uses packed Q/K/V with `flash_attn_varlen_func`; decode uses
  `flash_attn_with_kvcache` to consume block tables directly. KV writes use a
  Triton kernel. Prefill is chunked by a per-step token budget. At startup the
  model is warmed up and the number of KV-cache blocks is derived from measured
  peak memory and `gpu_memory_utilization`, matching nano-vllm.
- **Layout:** mirrors `nano-vllm/nanovllm/` (`engine/`, `layers/`, `models/`,
  `utils/`). `block_manager` and `context` support paged caching. The minimal
  `scheduler` owns KV-block allocation, manages waiting/running requests,
  prioritizes prefill, supports reference-style chunked prefill, shares
  immutable full prefix blocks by chained xxHash64 token hashes, and preempts
  on block pressure.
- Single GPU, `tp_size == 1` (all `torch.distributed` machinery removed).

See `docs/superpowers/specs/` for the full design and `docs/superpowers/plans/`
for the implementation plan.

## Installation

```bash
pip install .
```

For development with the locked dependency set, use `uv sync`.

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

llm = LLM(
    "/path/to/Qwen3-0.6B",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)
sp = SamplingParams(temperature=0.0, max_tokens=256)   # temperature <= 0 -> greedy
outputs = llm.generate(["Hello."], sp)
print(outputs[0]["text"])
```

## INT4 weight-only quantization

Pass `quantization="int4"` to quantize the Qwen3 attention and MLP linear
weights while loading an ordinary BF16/FP16 safetensors checkpoint. Q/K/V and
gate/up shards remain fused. Embeddings, the LM head, normalization, attention,
activations, and the KV cache retain their original dtype.

```python
llm = LLM(
    "/path/to/Qwen3-0.6B",
    quantization="int4",
    quantization_group_size=128,
)
```

The format stores two signed INT4 values per byte and one FP16 scale per output
row and input group. The selected group size must evenly divide every
Transformer linear layer's input dimension. CUDA inference uses a fused Triton
dequantization/GEMM kernel; the CPU implementation is a correctness reference,
not a performance path. Quantization currently reduces weight memory but does
not quantize the KV cache.

See `example.py` for a complete chat-template example and `bench.py` for the
reference-compatible benchmark.

## MiniMind-V

MiniMind-V accepts vLLM-style multimodal request dictionaries. The vision
tower is loaded separately because it is not included in the MiniMind-V
language-model checkpoint.

```python
from PIL import Image
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/mnt/c/Downloads/minimind-3v",
    vision_model="/mnt/c/Downloads/siglip2-base-p32-256-ve",
)
prompt = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "<image>\nDescribe this image."}],
    tokenize=False,
    add_generation_prompt=True,
    open_thinking=False,
)
outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": Image.open("image.jpg").convert("RGB")},
    }],
    SamplingParams(temperature=0.0, max_tokens=256),
)
```

The input processor expands each `<image>` marker into 64 image placeholder
tokens. Vision encoder outputs are cached by image content, sliced correctly
across chunked prefill boundaries, and included in prefix-cache block hashes.
See `minimind_v_example.py` for a runnable local example.

MiniMind-V MoE checkpoints using the same architecture are supported without
configuration changes. For BF16/FP16, expert weights are packed as vLLM-style
`w13` and `w2` tensors. Prefill groups tokens by selected expert, while Top-1
decode uses a Triton routed-GEMM path that only reads the selected expert and is
compatible with CUDA Graph capture. Online INT4 uses the existing per-expert
linear fallback; the router remains in the model dtype.

```python
llm = LLM(
    "/mnt/c/Downloads/minimind-3v-moe",
    vision_model="/mnt/c/Downloads/siglip2-base-p32-256-ve",
)
```

## MiniMind-O Thinker

The first-stage MiniMind-O integration supports text, image, and audio inputs
with text generation. The language-model checkpoint contains the trainable
Thinker, projectors, and Talker weights, while the frozen SigLIP2 and
SenseVoice models are loaded from separate local directories. Talker inference
and Mimi waveform generation are intentionally left for the second stage.

Install the optional audio dependencies:

```bash
uv sync --extra omni
```

Expected local model layout:

```text
/mnt/c/Downloads/
├── minimind-3o/
├── siglip2-base-p32-256-ve/
└── SenseVoiceSmall/
```

Create the engine with both external encoders:

```python
llm = LLM(
    "/mnt/c/Downloads/minimind-3o",
    vision_model="/mnt/c/Downloads/siglip2-base-p32-256-ve",
    audio_model="/mnt/c/Downloads/SenseVoiceSmall",
    enforce_eager=True,
)
```

Multimodal prompts use one `<image>` marker per image and one `<audio>` marker
per audio item. The processor expands image markers to 64 SigLIP2 tokens and
audio markers to the number of valid SenseVoice feature frames. Both modality
outputs use the same encoder cache, chunked-prefill slicing, and prefix-cache
media identity mechanism as MiniMind-V.

See `minimind_o_example.py` for a mixed audio-and-image request. This stage
returns only `text` and `token_ids`; it does not return Mimi audio codes or a
waveform.

## License

MIT. See `LICENSE`.
