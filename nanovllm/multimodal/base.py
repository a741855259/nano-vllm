from collections.abc import Mapping


class MultiModalProcessor:
    """Adapt request modality keys to a processor's local process arguments."""

    modality_arguments: dict[str, str] = {}

    def process_request(self, prompt: str, multi_modal_data: Mapping):
        if not isinstance(multi_modal_data, Mapping):
            raise TypeError("multi_modal_data must be a mapping")
        unsupported = set(multi_modal_data) - self.modality_arguments.keys()
        if unsupported:
            raise ValueError(f"unsupported input modalities: {sorted(unsupported)}")
        kwargs = {
            argument: multi_modal_data.get(modality)
            for modality, argument in self.modality_arguments.items()
        }
        return self.process(prompt, **kwargs)

    def process(self, prompt: str, **kwargs):
        raise NotImplementedError
