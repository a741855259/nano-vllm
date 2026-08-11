from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model: str, **kwargs):
        config_fields = {k for k in Config.__dataclass_fields__} - {"hf_config", "eos"}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)
        self.model_runner = ModelRunner(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id

    def _finished(self, seq: Sequence) -> bool:
        if seq.num_completion_tokens >= seq.max_tokens:
            return True
        if len(seq) >= self.config.max_model_len:
            return True
        if not seq.ignore_eos and seq.last_token == self.config.eos:
            return True
        return False

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        seqs = []
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)
            seqs.append(Sequence(prompt, sp))

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        for seq in seqs:
            # Requests are still serial, but decode inputs only the newest token
            # and reuses the sequence's per-layer KV cache.
            first_step = True
            while not self._finished(seq):
                t = perf_counter()
                token_id = self.model_runner.run(seq)
                seq.append_token(token_id)
                num_tokens = len(seq.prompt_token_ids) if first_step else -1
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                first_step = False
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            outputs[seq.seq_id] = seq.completion_token_ids
            self.model_runner.reset_cache()
            pbar.update(1)
        pbar.close()

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
