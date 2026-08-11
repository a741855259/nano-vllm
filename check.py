"""Cross-check: compare our Qwen3ForCausalLM against HF AutoModelForCausalLM.

For a fixed prompt, both models run a greedy prefill and we compare the
next-token argmax. A match confirms safetensors weights are loaded correctly
into the custom model (RoPE convention, q/k norm placement, fused projections).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanovllm import LLM


def main():
    model_path = "/mnt/c/Downloads/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "list all prime numbers within 100"}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    # --- HF reference ---
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    with torch.inference_mode():
        hf_logits = hf_model(input_ids).logits[0, -1]
    hf_token = hf_logits.argmax(dim=-1).item()
    hf_top = torch.topk(hf_logits.float(), 5).indices.tolist()

    # --- our engine (greedy, single forward over the prompt) ---
    llm = LLM(model_path, max_model_len=4096)
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    seq = Sequence(input_ids[0].tolist(), SamplingParams(temperature=0.0, max_tokens=1))
    our_token = llm.model_runner.run(seq)

    print(f"prompt tokens : {input_ids.numel()}")
    print(f"HF  argmax id : {hf_token}  ({tokenizer.decode([hf_token])!r})")
    print(f"ours argmax id: {our_token}  ({tokenizer.decode([our_token])!r})")
    print(f"HF  top-5     : {hf_top}")
    print("MATCH" if our_token == hf_token else "MISMATCH")


if __name__ == "__main__":
    main()
