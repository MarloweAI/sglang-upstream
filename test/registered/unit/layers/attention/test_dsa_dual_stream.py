import ast
import unittest
from pathlib import Path

from sglang.srt.layers.attention.dsa.dual_stream import (
    can_use_dsa_indexer_dual_stream,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_DSA_DIR = (
    Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/attention/dsa"
)
_INDEXERS: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    (
        "dsa_indexer.py",
        "Indexer",
        "forward_cuda",
        {"is_cuda": "_is_cuda", "is_hip": "_is_hip"},
    ),
    (
        "dsa_indexer_kpool.py",
        "IndexerKPool",
        "_forward_cuda_impl",
        {"is_cuda": "is_cuda()", "is_hip": "is_hip()"},
    ),
)


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _dual_stream_assignment(method: ast.FunctionDef) -> ast.Assign:
    return next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "enable_dual_stream"
    )


class TestDSAIndexerDualStream(CustomTestCase):
    def test_cuda_keeps_existing_token_range(self):
        expected = {-1: False, 0: False, 1: True, 1024: True, 1025: False}
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

    def test_both_indexers_wire_platform_gate(self):
        for filename, class_name, method_name, expected_keywords in _INDEXERS:
            with self.subTest(filename=filename):
                assignment = _dual_stream_assignment(
                    _method(_DSA_DIR / filename, class_name, method_name)
                )
                self.assertIsInstance(assignment.value, ast.BoolOp)
                self.assertIsInstance(assignment.value.op, ast.And)

                conjuncts = {ast.unparse(value) for value in assignment.value.values}
                self.assertIn("self.alt_stream is not None", conjuncts)
                self.assertIn("get_is_capture_mode()", conjuncts)

                calls = [
                    value
                    for value in assignment.value.values
                    if isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "can_use_dsa_indexer_dual_stream"
                ]
                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertEqual(ast.unparse(call.args[0]), "q_lora.shape[0]")
                self.assertEqual(
                    {kw.arg: ast.unparse(kw.value) for kw in call.keywords},
                    expected_keywords,
                )

    def test_kpool_keeps_breakable_graph_single_stream_guard(self):
        assignment = _dual_stream_assignment(
            _method(
                _DSA_DIR / "dsa_indexer_kpool.py",
                "IndexerKPool",
                "_forward_cuda_impl",
            )
        )
        conjuncts = {ast.unparse(value) for value in assignment.value.values}
        self.assertIn(
            "not (is_in_breakable_cuda_graph() and "
            "forward_batch.forward_mode.is_extend_without_speculative())",
            conjuncts,
        )

    def test_both_indexers_initialize_sm_budget_on_every_platform(self):
        for filename, class_name, _, _ in _INDEXERS:
            with self.subTest(filename=filename):
                init = _method(_DSA_DIR / filename, class_name, "__init__")
                values: list[ast.expr | None] = []
                for statement in init.body:
                    if isinstance(statement, ast.AnnAssign):
                        targets = [statement.target]
                        value = statement.value
                    elif isinstance(statement, ast.Assign):
                        targets = statement.targets
                        value = statement.value
                    else:
                        continue
                    if any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "half_device_sm_count"
                        for target in targets
                    ):
                        values.append(value)

                self.assertEqual(len(values), 1)
                self.assertIsInstance(values[0], ast.Constant)
                self.assertIsNone(values[0].value)


if __name__ == "__main__":
    unittest.main()
