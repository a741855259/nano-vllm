import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.quantization import (
    Int4QuantizationConfig,
    dequantize_int4,
    quantize_int4,
    unpack_int4,
)
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from transformers import Qwen3Config


class Int4QuantizationTest(unittest.TestCase):

    def test_packs_signed_nibbles_in_input_order(self):
        values = torch.tensor(
            [[-7.0, -6.0, -1.0, 0.0, 1.0, 2.0, 5.0, 6.0]],
        )
        packed, scales = quantize_int4(values, group_size=8)

        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.tolist(), [[0xA9, 0x0F, 0x21, 0x65]])
        self.assertEqual(unpack_int4(packed).tolist(), values.int().tolist())
        self.assertEqual(scales.shape, (1, 1))

        all_nibbles = torch.tensor(
            [[0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76]],
            dtype=torch.uint8,
        )
        self.assertEqual(
            unpack_int4(all_nibbles).tolist(),
            [list(range(-8, 8))],
        )

    def test_groupwise_error_is_bounded_and_zero_group_is_exact(self):
        torch.manual_seed(0)
        weight = torch.randn(5, 16)
        weight[0, :8] = 0
        packed, scales = quantize_int4(weight, group_size=8)
        restored = dequantize_int4(packed, scales, group_size=8)

        error = (restored - weight).abs().reshape(5, 2, 8).amax(dim=-1)
        bound = scales.float() / 2 + 2e-3
        self.assertTrue(torch.all(error <= bound))
        self.assertTrue(torch.equal(restored[0, :8], torch.zeros(8)))

    def test_quantized_linear_uses_packed_storage_and_cpu_reference(self):
        quant_config = Int4QuantizationConfig(group_size=8)
        layer = RowParallelLinear(
            16,
            6,
            bias=True,
            quant_config=quant_config,
            prefix="proj",
        )
        weight = torch.randn(6, 16)
        bias = torch.randn(6)
        layer.weight_loader(layer.weight, weight)
        layer.weight_loader(layer.bias, bias)

        self.assertEqual(layer.weight.shape, (6, 8))
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertEqual(layer.weight_scale.shape, (6, 2))
        self.assertFalse(layer.weight.requires_grad)

        x = torch.randn(3, 16)
        restored = dequantize_int4(layer.weight, layer.weight_scale, 8)
        self.assertTrue(torch.equal(layer(x), F.linear(x, restored, bias)))

    def test_loader_quantizes_fused_qkv_shards_into_their_row_ranges(self):
        class TinyModel(torch.nn.Module):
            packed_modules_mapping = {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
                "v_proj": ("qkv_proj", "v"),
            }

            def __init__(self):
                super().__init__()
                quant_config = Int4QuantizationConfig(group_size=4)
                self.qkv_proj = QKVParallelLinear(
                    hidden_size=8,
                    head_size=2,
                    num_heads=2,
                    num_kv_heads=1,
                    quant_config=quant_config,
                    prefix="qkv_proj",
                )

        q = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 10
        k = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 7
        v = -torch.arange(16, dtype=torch.float32).reshape(2, 8) / 5
        with tempfile.TemporaryDirectory() as model_dir:
            save_file(
                {
                    "q_proj.weight": q,
                    "k_proj.weight": k,
                    "v_proj.weight": v,
                },
                os.path.join(model_dir, "model.safetensors"),
            )
            model = TinyModel()
            load_model(model, model_dir)

        restored = dequantize_int4(
            model.qkv_proj.weight,
            model.qkv_proj.weight_scale,
            group_size=4,
        )
        expected_parts = []
        for shard in (q, k, v):
            packed, scales = quantize_int4(shard, group_size=4)
            expected_parts.append(dequantize_int4(packed, scales, group_size=4))
        self.assertTrue(torch.equal(restored, torch.cat(expected_parts)))

    def test_rejects_incompatible_group_size(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            RowParallelLinear(
                12,
                4,
                quant_config=Int4QuantizationConfig(group_size=8),
            )

    def test_qwen_threads_quantization_only_to_transformer_linears(self):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
        )
        model = Qwen3ForCausalLM(
            config,
            quant_config=Int4QuantizationConfig(group_size=8),
        )

        layer = model.model.layers[0]
        for linear in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        ):
            self.assertEqual(linear.weight.dtype, torch.uint8)
            self.assertTrue(hasattr(linear, "weight_scale"))
        self.assertEqual(
            layer.self_attn.qkv_proj.prefix,
            "model.layers.0.self_attn.qkv_proj",
        )
        self.assertEqual(
            layer.mlp.down_proj.prefix,
            "model.layers.0.mlp.down_proj",
        )
        self.assertIsNot(
            layer.self_attn.qkv_proj.quant_method,
            layer.self_attn.o_proj.quant_method,
        )
        self.assertEqual(model.model.embed_tokens.weight.dtype, torch.float32)
        self.assertEqual(
            model.lm_head.weight.data_ptr(),
            model.model.embed_tokens.weight.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
