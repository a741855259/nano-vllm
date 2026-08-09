import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    model_path = os.path.expanduser("/mnt/c/Downloads/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, max_model_len=4096)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "give me your opinion on the impact of artificial intelligence on society",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
