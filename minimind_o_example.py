from PIL import Image

from nanovllm import LLM, SamplingParams


def main():
    model_path = "/mnt/c/Downloads/minimind-3o"
    vision_model_path = "/mnt/c/Downloads/siglip2-base-p32-256-ve"
    audio_model_path = "/mnt/c/Downloads/SenseVoiceSmall"
    image_path = "/root/Train/minimind-o/dataset/eval_omni/image-01-orange-cat-moon-desk.jpg"
    audio_path = "/root/Train/minimind-o/dataset/eval_omni/img-01_图中是什么东西.mp3"

    llm = LLM(
        model_path,
        vision_model=vision_model_path,
        audio_model=audio_model_path,
        max_num_seqs=1,
        max_model_len=4096,
        enforce_eager=True,
    )
    prompt = llm.tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": "<audio>\n<image>\n请结合语音问题回答。",
        }],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    outputs = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_path,
                "image": Image.open(image_path).convert("RGB"),
            },
        }],
        SamplingParams(temperature=0.0, max_tokens=128),
    )
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
