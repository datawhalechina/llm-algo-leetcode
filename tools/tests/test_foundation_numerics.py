"""Check Part01 numerical examples against independent values and counterexamples."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]


def execute_numerical_notebook(prefix):
    """Execute the actual CPU teaching cells without changing saved notebook outputs."""
    path = next((ROOT / '01_Hardware_Math_and_Systems').glob(f'{prefix}_*.ipynb'))
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': __name__}
    with torch.random.fork_rng(devices=[]), redirect_stdout(StringIO()):
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), str(path), 'exec'), namespace)
    return namespace


class FoundationNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.precision = execute_numerical_notebook('01')
        cls.scale = execute_numerical_notebook('02')
        cls.quantization = execute_numerical_notebook('21')
        cls.moe = execute_numerical_notebook('22')

    def test_overflow_is_not_hidden_by_finite_only_error(self):
        """Distinguish finite rounding error from complete FP16 overflow."""
        inspect = self.precision['cast_error_report']
        row = inspect([1.0, 70_000.0], 'fp16')
        self.assertEqual((row['finite_count'], row['overflow_count']), (1, 1))
        self.assertEqual(row['max_abs_error'], 0.0)
        overflow = inspect([70_000.0, -70_000.0], 'fp16')
        self.assertEqual((overflow['finite_count'], overflow['overflow_count']), (0, 2))
        self.assertIsNone(overflow['max_abs_error'])
        self.assertEqual(inspect([], 'fp16')['total_count'], 0)
        for value in (float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspect([value], 'fp16')

    def test_nonzero_overflow_is_not_a_usable_gradient(self):
        """An infinite gradient is nonzero but cannot be counted as a finite survivor."""
        row = self.precision['gradient_survival_report']([1.0, 70_000.0, 0.0], 'fp16')
        self.assertEqual(row['nonzero_count'], 2)
        self.assertEqual(row['finite_count'], 2)
        self.assertEqual(row['finite_nonzero_count'], 1)

    def test_fp8_scaling_uses_supplied_not_historical_amax(self):
        """Changing the supplied magnitude changes scale by its reciprocal ratio."""
        policy = self.precision['fp8_variant_policy']
        row = policy(forward_amax=2, backward_amax=2)
        self.assertEqual(row['scaling_source'], 'provided_amax')
        self.assertEqual(row['forward_scale'], 224)
        self.assertEqual(row['backward_scale'], 28_672)
        self.assertEqual(policy(forward_amax=4)['forward_scale'], 112)

    def test_master_weight_storage_is_counted_once(self):
        """Two FP32 moments cost 8 bytes per parameter; master weights add 4."""
        ledger = self.precision['training_state_ledger']
        self.assertEqual(ledger(7, keep_master_weights=True)['total_gb'], 112)
        self.assertEqual(ledger(7, keep_master_weights=False)['total_gb'], 84)

    def test_parameter_formula_counts_norms_and_ffn_matrix_count(self):
        """Manually count embedding, MHA, FFN and two norm vectors for a tiny block."""
        params = self.scale['calculate_transformer_params']
        self.assertEqual(params(10, 6, 1, ffn_type='dense'), 120 + 144 + 288 + 12)
        self.assertEqual(params(10, 6, 1, intermediate_size=16), 564)
        self.assertEqual(params(10, 6, 1, intermediate_size=24), 120 + 144 + 432 + 12)
        self.assertEqual(params(10, 6, 1, tie_embeddings=True), 564 - 60)
        self.assertEqual(params(32_000, 4096, 32, intermediate_size=11_008), 6_738_411_520)

    def test_training_time_uses_all_devices_and_given_efficiency(self):
        """A 7B, 1T-token run uses the declared 6PT model, not a measured duration."""
        estimate = self.scale['estimate_training_time']
        self.assertAlmostEqual(estimate(7, 1e12, 312, 1000) / 24, 4.451566951566951)
        self.assertAlmostEqual(estimate(7, 1e12, 312, 8), estimate(7, 1e12, 312, 1) / 8)

    def test_mfu_matches_the_scope_of_model_work_and_peak(self):
        """Scale model work and peak together when moving from one GPU to eight."""
        mfu = self.scale['calculate_mfu']
        self.assertAlmostEqual(mfu(1e15, 10, 312), 1 / 3.12)
        self.assertEqual(mfu(8e15, 10, 8 * 312), mfu(1e15, 10, 312))

    def test_time_fractions_do_not_invent_waiting(self):
        """No waits yield no dominant stall; equal waits remain tied."""
        inspect = self.scale['compute_time_fraction']
        self.assertEqual(inspect(100, 0, 0)['dominant_stall'], 'none')
        self.assertEqual(inspect(100, 20, 20)['dominant_stall'], 'mixed')
        self.assertEqual(inspect(100, 30, 20)['compute_time_fraction'], 0.667)
        with self.assertRaises(ValueError):
            inspect(100, -1, 2)

    def test_quantization_metadata_and_partial_groups_cost_bytes(self):
        """Count packed nibbles and ceil-divided groups independently."""
        storage = self.quantization['quantized_weight_bytes']
        self.assertEqual(storage(1024)['total_bytes'], 512 + 8 * 2)
        self.assertEqual(storage(1025)['total_bytes'], 513 + 9 * 2)
        self.assertEqual(storage(1024, zero_point_bytes=1)['total_bytes'], 512 + 8 * 3)
        self.assertEqual(storage(0)['total_bytes'], 0)

    def test_quantization_rejects_unrepresentable_bitwidth(self):
        """The INT8 carrier must not silently wrap a 16-bit quantized integer."""
        x = torch.ones(2, 2)
        with self.assertRaises(ValueError):
            self.quantization['quantize_per_tensor'](x, num_bits=16)
        with self.assertRaises(ValueError):
            self.quantization['per_channel_scales'](x, num_bits=16)

    def test_quantization_zero_half_and_per_channel_scales(self):
        """FP16 zero inputs retain a nonzero FP32 scale; row magnitudes stay separate."""
        quantize = self.quantization['quantize_per_tensor']
        q, scale = quantize(torch.zeros(4, dtype=torch.float16), 4)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(scale.item(), 1)
        self.assertTrue(torch.all(q == 0))
        scales = self.quantization['per_channel_scales'](torch.tensor([[1., -1.], [2., -2.]]))
        torch.testing.assert_close(scales, torch.tensor([[1 / 127], [2 / 127]]))

    def test_qat_budget_cannot_rescue_failed_ptq_quality(self):
        """An explicit trial cost controls budget checks, not a fixed ten-hour rule."""
        assess = self.quantization['qat_recommendation']
        failed = assess(0.6, retrain_budget_hours=8, estimated_qat_hours=10)
        self.assertFalse(failed['ptq_quality_met'])
        self.assertEqual(failed['recommendation'], 'adjust_quantization_or_budget')
        self.assertEqual(assess(0.6, retrain_budget_hours=8, estimated_qat_hours=2)['recommendation'], 'QAT_trial')
        self.assertEqual(assess(0.6)['recommendation'], 'estimate_QAT_cost')
        self.assertEqual(assess(0.5)['recommendation'], 'PTQ_candidate')

    def test_moe_capacity_counts_topk_assignments(self):
        """Four tokens create eight assignments; overflowing assignments are not unique tokens."""
        capacity = self.moe['expert_capacity'](4, 4, top_k=2)
        self.assertEqual(capacity, 2)
        stats = self.moe['load_balance_stats']([4, 3, 1, 0], capacity)
        self.assertEqual(stats['total_assignments'], 8)
        self.assertEqual(stats['overflow_assignments'], 3)
        self.assertEqual(stats['overflow_assignment_rate'], 3 / 8)

    def test_moe_communication_follows_assignments_not_expert_count(self):
        """Each remote activation copy is counted once on dispatch and once on return."""
        traffic = self.moe['estimate_expert_parallel_traffic']
        self.assertEqual(traffic(1, 4, 32, 8, 4, top_k=2), 768)
        self.assertEqual(traffic(1, 4, 32, 16, 4, top_k=2), 768)
        self.assertEqual(traffic(1, 4, 32, 8, 1, top_k=2), 0)
        self.assertEqual(traffic(1, 4, 32, 8, 4, top_k=2, remote_fraction=0), 0)
        self.assertEqual(traffic(1, 4, 32, 8, 4, top_k=2, remote_fraction=1), 1024)
        with self.assertRaises(ValueError):
            traffic(1, 4, 32, 7, 4, top_k=2)


if __name__ == '__main__':
    unittest.main()
