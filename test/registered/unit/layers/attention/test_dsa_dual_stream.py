import unittest

from sglang.srt.layers.attention.dsa.dual_stream import (
    can_use_dsa_indexer_dual_stream,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSAIndexerDualStream(CustomTestCase):
    def test_cuda_keeps_existing_token_range(self):
        expected = {-1: False, 0: False, 1: True, 1024: True, 1025: False}
        for num_tokens, enabled in expected.items():
            with self.subTest(num_tokens=num_tokens):
                self.assertEqual(
                    can_use_dsa_indexer_dual_stream(num_tokens, is_cuda=True),
                    enabled,
                )

    def test_non_cuda_is_explicitly_disabled(self):
        for num_tokens in (-1, 0, 1, 1024, 1025):
            with self.subTest(num_tokens=num_tokens):
                self.assertFalse(
                    can_use_dsa_indexer_dual_stream(num_tokens, is_cuda=False)
                )


if __name__ == "__main__":
    unittest.main()
