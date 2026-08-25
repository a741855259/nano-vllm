import unittest
from unittest.mock import patch

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


class FakeTensor:

    def __init__(self, data):
        self.data = data
        self.non_blocking = None

    def cuda(self, non_blocking=False):
        self.non_blocking = non_blocking
        return self


class AsyncBatchPreparationTest(unittest.TestCase):

    def setUp(self):
        self.runner = ModelRunner.__new__(ModelRunner)
        self.runner.block_size = 4
        self.created = []

    def tearDown(self):
        reset_context()

    def fake_tensor(self, data, **kwargs):
        tensor = FakeTensor(data)
        tensor.kwargs = kwargs
        self.created.append(tensor)
        return tensor

    def assert_pinned_async(self):
        self.assertTrue(self.created)
        for tensor in self.created:
            self.assertEqual(tensor.kwargs["device"], "cpu")
            self.assertTrue(tensor.kwargs["pin_memory"])
            self.assertTrue(tensor.non_blocking)

    def test_prefill_metadata_uses_pinned_non_blocking_transfers(self):
        seq = Sequence([11, 12, 13])
        seq.num_scheduled_tokens = 3
        seq.block_table = [2]

        with patch("torch.tensor", side_effect=self.fake_tensor):
            input_ids, positions, _ = self.runner.prepare_prefill([seq])

        self.assertEqual(input_ids.data, [11, 12, 13])
        self.assertEqual(positions.data, [0, 1, 2])
        self.assertEqual(get_context().slot_mapping.data, [8, 9, 10])
        self.assert_pinned_async()

    def test_decode_sample_and_block_tables_use_async_transfers(self):
        seq = Sequence([11, 12, 13, 14, 15])
        seq.num_cached_tokens = 4
        seq.num_scheduled_tokens = 1
        seq.block_table = [3, 7]

        with patch("torch.tensor", side_effect=self.fake_tensor):
            input_ids, positions = self.runner.prepare_decode([seq])
            temperatures = self.runner.prepare_sample([seq])

        self.assertEqual(input_ids.data, [15])
        self.assertEqual(positions.data, [4])
        self.assertEqual(get_context().slot_mapping.data, [28])
        self.assertEqual(get_context().block_tables.data, [[3, 7]])
        self.assertEqual(temperatures.data, [seq.temperature])
        self.assert_pinned_async()


if __name__ == "__main__":
    unittest.main()
