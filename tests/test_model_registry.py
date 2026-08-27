import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.models.registry import resolve_model_class


class ModelRegistryTest(unittest.TestCase):

    def test_resolves_registered_architecture_lazily(self):
        fake_module = types.ModuleType("nanovllm.models.qwen3")
        fake_model_class = object()
        fake_module.Qwen3ForCausalLM = fake_model_class

        with patch.dict(sys.modules, {"nanovllm.models.qwen3": fake_module}):
            model_class = resolve_model_class(
                SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
            )

        self.assertIs(model_class, fake_model_class)

    def test_uses_first_supported_architecture(self):
        fake_module = types.ModuleType("nanovllm.models.qwen3")
        fake_model_class = object()
        fake_module.Qwen3ForCausalLM = fake_model_class

        with patch.dict(sys.modules, {"nanovllm.models.qwen3": fake_module}):
            model_class = resolve_model_class(
                SimpleNamespace(
                    architectures=["UnknownModel", "Qwen3ForCausalLM"],
                ),
            )

        self.assertIs(model_class, fake_model_class)

    def test_rejects_unsupported_architecture(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model architectures"):
            resolve_model_class(
                SimpleNamespace(architectures=["UnknownModel"]),
            )


if __name__ == "__main__":
    unittest.main()
