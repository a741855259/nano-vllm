from transformers import AutoConfig, PretrainedConfig


class MiniMindVConfig(PretrainedConfig):
    """Local MiniMind-V config used without executing checkpoint code."""

    model_type = "minimind-v"

    def __init__(
        self,
        image_special_token: str = "<|image_pad|>",
        image_ids: list[int] | None = None,
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_special_token = image_special_token
        self.image_ids = [12] if image_ids is None else image_ids
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len


def register_model_configs() -> None:
    """Register nano-vLLM-owned configs with Transformers auto classes."""

    AutoConfig.register(
        MiniMindVConfig.model_type,
        MiniMindVConfig,
        exist_ok=True,
    )
