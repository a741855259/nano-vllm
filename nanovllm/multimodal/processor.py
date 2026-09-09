from nanovllm.multimodal.minimind_base import MiniMindProcessorBase


class MiniMindVProcessor(MiniMindProcessorBase):
    modality_arguments = {"image": "images"}

    def process(self, prompt: str, images=None):
        prompt, images = self._prepare_image_prompt(prompt, images)
        token_ids = self.tokenizer.encode(prompt)
        return token_ids, self._image_features(token_ids, images)
