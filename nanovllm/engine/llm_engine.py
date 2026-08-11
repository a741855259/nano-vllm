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
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        seqs = []
        for prompt, sp in zip(prompts, sampling_params):
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)
            seqs.append(Sequence(prompt, sp))

        results: dict[int, list[int]] = {}
        for seq in seqs:
            # Stateless generation: recompute over the full sequence each step
            # (no KV cache), sampling one token at a time until finished.
            while not self._finished(seq):
                token_id = self.model_runner.run(seq)
                seq.append_token(token_id)
            results[seq.seq_id] = seq.completion_token_ids

        ordered = [results[seq.seq_id] for seq in seqs]
        return [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in ordered]
