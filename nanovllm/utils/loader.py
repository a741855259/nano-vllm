from glob import glob
import os
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def iter_weights(path: str):
    safetensor_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if safetensor_files:
        for file in safetensor_files:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    yield weight_name, f.get_tensor(weight_name)
        return

    bin_files = sorted(glob(os.path.join(path, "pytorch_model*.bin")))
    if not bin_files:
        raise FileNotFoundError(
            f"no safetensors or pytorch_model*.bin weights found in {path}"
        )
    for file in bin_files:
        state_dict = torch.load(
            file,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        yield from state_dict.items()


def load_model(model: nn.Module, path: str):
    """Load checkpoint weights into the model, fusing q/k/v -> qkv_proj and
    gate/up -> gate_up_proj via packed_modules_mapping (same scheme as nano-vllm)."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    ignored_weight_prefixes = getattr(model, "ignored_weight_prefixes", ())
    for weight_name, loaded_weight in iter_weights(path):
        if weight_name.startswith(ignored_weight_prefixes):
            continue
        parts = weight_name.split(".")
        if len(parts) >= 5 and parts[-4] == "experts":
            try:
                expert_id = int(parts[-3])
            except ValueError:
                expert_id = -1
            projection = parts[-2]
            experts_name = ".".join(parts[:-3])
            if expert_id >= 0 and projection in {
                "gate_proj",
                "up_proj",
                "down_proj",
            }:
                experts = model.get_submodule(experts_name)
                expert_loader = getattr(experts, "load_expert_weight", None)
                # 按专家ID逐步加载到同一个packed专家权重矩阵
                if expert_loader is not None:
                    expert_loader(expert_id, projection, loaded_weight)
                    continue
        for source_name, (target_name, shard_id) in packed_modules_mapping.items():
            if source_name in weight_name:
                param_name = weight_name.replace(source_name, target_name)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
