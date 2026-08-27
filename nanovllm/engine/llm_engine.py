from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler


class LLMEngine:

    def __init__(self, model: str, **kwargs):
        config_fields = {k for k in Config.__dataclass_fields__} - {"hf_config", "eos"}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        Sequence.block_size = self.config.kvcache_block_size
        self.model_runner = ModelRunner(self.config)
        if getattr(self.config.hf_config, "model_type", None) == "minimind-v":
            from nanovllm.multimodal.processor import MiniMindVProcessor

            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.config.model,
                local_files_only=True,
            )
            self.multimodal_processor = MiniMindVProcessor(
                self.tokenizer,
                self.config.vision_model,
                self.config.hf_config,
                self.config.max_num_images_per_prompt,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                use_fast=True,
            )
            self.multimodal_processor = None
        self.config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(self.config)

    def add_request(self, prompt, sampling_params: SamplingParams):
        mm_features = []
        if isinstance(prompt, dict):
            multi_modal_data = prompt.get("multi_modal_data") or {}
            images = multi_modal_data.get("image")
            prompt = prompt.get("prompt")
            if not isinstance(prompt, str):
                raise TypeError("multimodal input requires a string 'prompt'")
            if self.multimodal_processor is None:
                raise ValueError("this model does not support multimodal inputs")
            prompt, mm_features = self.multimodal_processor.process(prompt, images)
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add(Sequence(prompt, sampling_params, mm_features))

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
