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


class MiniMindOConfig(MiniMindVConfig):
    """Local MiniMind-O config used without executing checkpoint code."""

    model_type = "minimind-o"

    def __init__(
        self,
        audio_special_token: str = "<|audio_pad|>",
        audio_ids: list[int] | None = None,
        audio_hidden_size: int = 512,
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        num_talker_hidden_layers: int = 4,
        talker_hidden_size: int = 768,
        spk_emb_size: int = 192,
        think_end_ids: list[int] | None = None,
        bridge_layer: int | None = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tie_word_embeddings = tie_word_embeddings
        self.audio_special_token = audio_special_token
        self.audio_ids = [16] if audio_ids is None else audio_ids
        self.audio_hidden_size = audio_hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.audio_stop_token = audio_stop_token
        self.audio_spk_token = audio_spk_token
        self.num_talker_hidden_layers = num_talker_hidden_layers
        self.talker_hidden_size = talker_hidden_size
        self.spk_emb_size = spk_emb_size
        self.think_end_ids = [26, 234, 234] if think_end_ids is None else think_end_ids
        self.bridge_layer = (
            getattr(self, "num_hidden_layers", 8) // 2 - 1
            if bridge_layer is None
            else bridge_layer
        )


def register_model_configs() -> None:
    """Register nano-vLLM-owned configs with Transformers auto classes."""

    AutoConfig.register(
        MiniMindVConfig.model_type,
        MiniMindVConfig,
        exist_ok=True,
    )
    AutoConfig.register(
        MiniMindOConfig.model_type,
        MiniMindOConfig,
        exist_ok=True,
    )
