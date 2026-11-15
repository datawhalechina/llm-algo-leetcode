"""Check the GEMM notebook's CPU reference; CUDA kernel execution is separate."""

import ast
import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


NOTEBOOK = Path(__file__).resolve().parents[2] / '03_Triton_Kernels/05_Triton_GEMM_Tutorial.ipynb'


@unittest.skipIf(torch is None, 'requires CPU PyTorch for the reference calculation')
class GemmEpilogueReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load actual reference and wrappers without importing or starting Triton."""
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.reference = None
        reference_definitions = []
        cls.wrappers = []
        for cell in cls.notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in ('gemm_reference', 'triton_gemm'):
                    namespace = {'torch': torch}
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
                    if node.name == 'gemm_reference':
                        reference_definitions.append(ast.dump(node))
                        cls.reference = staticmethod(namespace[node.name])
                    else:
                        cls.wrappers.append(namespace[node.name])
        if (cls.reference is None or len(cls.wrappers) != 2
                or len(reference_definitions) != 2
                or reference_definitions[0] != reference_definitions[1]):
            raise AssertionError('Exercise and answer must provide matching reference definitions and wrappers')

    def test_four_modes_broadcast_columns_without_mutating_inputs(self):
        a = torch.tensor([[1, 2], [-1, 1]], dtype=torch.float16)
        b = torch.tensor([[2, 0, -2], [1, -1, 1]], dtype=torch.float16)
        bias = torch.tensor([-1, 0, 2], dtype=torch.float16)
        snapshots = [x.clone() for x in (a, b, bias)]
        base = torch.tensor([[4, -2, 0], [-1, -1, 3]], dtype=torch.float16)
        for use_bias in (False, True):
            for activation in ('none', 'silu'):
                expected = base + bias if use_bias else base
                if activation == 'silu':
                    expected = torch.nn.functional.silu(expected.float()).half()
                actual = self.reference(a, b, bias if use_bias else None, activation)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(actual.dtype, torch.float16)
        for value, original in zip((a, b, bias), snapshots):
            torch.testing.assert_close(value, original, atol=0, rtol=0)

    def test_matmul_rounding_is_observable_before_bias_cancellation(self):
        a = torch.ones((1, 2), dtype=torch.float16)
        b = torch.tensor([[1024], [0.5]], dtype=torch.float16)
        bias = torch.tensor([-1024], dtype=torch.float16)
        self.assertEqual(self.reference(a, b, bias).item(), 0)
        omitted_rounding = a.float() @ b.float() + bias.float()
        self.assertEqual(omitted_rounding.item(), 0.5)

    def test_bias_rounding_is_observable_before_activation(self):
        a = b = torch.ones((1, 1), dtype=torch.float16)
        bias = torch.tensor([2 ** -11], dtype=torch.float16)
        actual = self.reference(a, b, bias, 'silu')
        omitted_rounding = torch.nn.functional.silu((a @ b).float() + bias.float()).half()
        self.assertEqual(actual.item(), 0.73095703125)
        self.assertEqual(omitted_rounding.item(), 0.7314453125)

    def test_activation_follows_the_whole_k_reduction(self):
        a = torch.ones((1, 2), dtype=torch.float16)
        b = torch.tensor([[2], [-3]], dtype=torch.float16)
        actual = self.reference(a, b, activation='silu')
        expected = torch.nn.functional.silu(torch.tensor([[-1.0]])).half()
        wrong = torch.nn.functional.silu(torch.tensor([2.0, -3.0])).sum().half()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertGreater(abs(actual.item() - wrong.item()), 1)

    def test_both_wrappers_reject_cpu_tensors_before_launch(self):
        a = b = torch.ones((2, 2), dtype=torch.float16)
        for wrapper in self.wrappers:
            with self.subTest(wrapper=wrapper), self.assertRaisesRegex(ValueError, 'CUDA'):
                wrapper(a, b)


if __name__ == '__main__':
    unittest.main()
