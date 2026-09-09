import unittest

from sglang.srt.layers.attention.dsa.dual_stream import (
    can_use_dsa_indexer_dual_stream,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSAIndexerDualStream(CustomTestCase):
    def test_cuda_keeps_existing_token_range(self):
        expected = {0: False, 1: True, 1024: True, 1025: False}
        for num_tokens, enabled in expected.items():
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(
                    can_use_dsa_indexer_dual_stream(
                        num_tokens, is_cuda=True, is_hip=False
                    ),
                    enabled,
                )

    def test_hip_accepts_every_nonempty_batch(self):
        expected = {0: False, 1: True, 1024: True, 1025: True}
        for num_tokens, enabled in expected.items():
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(
                    can_use_dsa_indexer_dual_stream(
                        num_tokens, is_cuda=False, is_hip=True
                    ),
                    enabled,
                )

    def test_other_platforms_remain_disabled(self):
        self.assertFalse(
            can_use_dsa_indexer_dual_stream(1, is_cuda=False, is_hip=False)
        )


if __name__ == "__main__":
    unittest.main()
