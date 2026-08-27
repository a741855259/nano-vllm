from PIL import Image

from nanovllm import LLM, SamplingParams


def main():
    model_path = "/mnt/c/Downloads/minimind-3v"
    vision_model_path = "/mnt/c/Downloads/siglip2-base-p32-256-ve"
    image_path = "/root/Train/minimind-v/dataset/eval_images/image-01-golden-dog-balloons.jpg"

    llm = LLM(
        model_path,
        vision_model=vision_model_path,
        max_num_seqs=1,
        max_model_len=4096,
    )
    prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "<image>\n请描述这张图片。"}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "image": Image.open(image_path).convert("RGB"),
        },
    }]
    outputs = llm.generate(
        inputs,
        SamplingParams(temperature=0.0, max_tokens=128),
    )
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
