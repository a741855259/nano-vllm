from importlib import import_module

from torch import nn


_MODELS = {
    "MiniMindOmni": (
        "nanovllm.models.minimind_o",
        "MiniMindOmniThinker",
    ),
    "MiniMindVLM": (
        "nanovllm.models.minimind_v",
        "MiniMindVLM",
    ),
    "Qwen3ForCausalLM": (
        "nanovllm.models.qwen3",
        "Qwen3ForCausalLM",
    ),
}


def resolve_model_class(hf_config) -> type[nn.Module]:
    architectures = getattr(hf_config, "architectures", None) or []
    for architecture in architectures:
        target = _MODELS.get(architecture)
        if target is None:
            continue
        module_name, class_name = target
        module = import_module(module_name)
        return getattr(module, class_name)

    raise ValueError(
        f"Unsupported model architectures: {architectures}. "
        f"Supported architectures: {sorted(_MODELS)}"
    )
