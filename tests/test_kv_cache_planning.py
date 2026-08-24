import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.attention import Attention


class KVCachePlanningTest(unittest.TestCase):

    def test_warmup_uses_reference_batch_shape(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(
            max_num_batched_tokens=128,
            max_model_len=64,
            max_num_seqs=4,
        )

        with (
            patch("torch.cuda.empty_cache") as empty_cache,
            patch("torch.cuda.reset_peak_memory_stats") as reset_peak,
            patch.object(runner, "run", return_value=[0, 0]) as run,
        ):
            runner.warmup_model()

        seqs, is_prefill = run.call_args.args
        self.assertTrue(is_prefill)
        self.assertEqual(len(seqs), 2)
        self.assertTrue(all(len(seq) == 64 for seq in seqs))
        self.assertTrue(all(seq.num_scheduled_tokens == 64 for seq in seqs))
        self.assertEqual(empty_cache.call_count, 2)
        reset_peak.assert_called_once_with()

    def test_allocates_blocks_from_profiled_gpu_memory(self):
        dtype = torch.bfloat16
        config = SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=2,
                num_key_value_heads=2,
                num_attention_heads=4,
                hidden_size=20,
                dtype=dtype,
            ),
            gpu_memory_utilization=0.9,
            num_kvcache_blocks=-1,
        )
        attentions = [Attention(4, 5, 5**-0.5, 2) for _ in range(2)]
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = config
        runner.block_size = 10
        runner.model = SimpleNamespace(modules=lambda: iter(attentions))

        real_empty = torch.empty
        with (
            patch("torch.cuda.mem_get_info", return_value=(40_000, 100_000)),
            patch(
                "torch.cuda.memory_stats",
                return_value={
                    "allocated_bytes.all.peak": 25_000,
                    "allocated_bytes.all.current": 10_000,
                },
            ),
            patch("torch.empty", side_effect=real_empty),
        ):
            runner.allocate_kv_cache()

        block_bytes = 2 * 2 * 10 * 2 * 5 * dtype.itemsize
        expected = int(100_000 * 0.9 - 60_000 - 25_000 + 10_000) // block_bytes
        self.assertEqual(config.num_kvcache_blocks, expected)
        self.assertEqual(runner.kv_cache.shape, (2, 2, expected, 10, 2, 5))
        self.assertEqual(attentions[0].k_cache.data_ptr(), runner.kv_cache[0, 0].data_ptr())
        self.assertEqual(attentions[1].v_cache.data_ptr(), runner.kv_cache[1, 1].data_ptr())


if __name__ == "__main__":
    unittest.main()
