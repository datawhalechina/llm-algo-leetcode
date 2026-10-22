"""Check declaration-based hardware screening and simplified cost models."""

import unittest

if __package__:
    from .test_foundation_performance_evidence import notebook_functions
else:
    from test_foundation_performance_evidence import notebook_functions


class FoundationHardwareConstraintsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = "01_Hardware_Math_and_Systems/"
        cls.devices = notebook_functions(prefix + "10_Domestic_AI_Chips_Overview.ipynb")
        cls.isolation = notebook_functions(prefix + "31_GPU_Virtualization_and_MIG.ipynb")
        cls.cost = notebook_functions(prefix + "33_TCO_and_Cost_Model.ipynb")
        cls.runtime = notebook_functions(prefix + "34_Heterogeneous_Accelerators_and_Runtime_Constraints.ipynb")

    def cpu_profile(self):
        return dict(memory_gb=2, precisions={'fp32'}, ops={'tokenize'},
                    runtimes={'python'}, collective=False)

    def prepare_plan(self):
        return dict(estimated_peak_gb=0.5, precision='fp32', required_ops={'tokenize'},
                    runtime='python', requires_collective=False)

    def test_communication_is_required_only_when_requested(self):
        check = self.devices['stack_readiness']
        self.assertTrue(check(True, True, True, True, False)['declared_requirements_met'])
        self.assertEqual(check(True, True, True, True, False,
                               requires_communication=True)['missing'], ['communication'])
        with self.assertRaises(ValueError):
            check(True, True, True, True, 'available')

    def test_invalid_memory_and_risk_values_are_rejected(self):
        with self.assertRaises(ValueError):
            self.devices['accelerator_feasibility'](
                float('nan'), 16e9, {'matmul'}, {'matmul'}, 'bf16', {'bf16'})
        with self.assertRaises(ValueError):
            self.devices['admit_accelerator_candidate'](True, True, 1, 8, 2, 0.5, 0.5, 1)

    def test_partition_model_requires_integer_nonempty_slices(self):
        check = self.isolation['partition_resources']
        self.assertEqual(check(80, 108, 7, 2)['unused_compute_units'], 3)
        for units, slices in ((4, 5), (4, 1.5), (True, 1)):
            with self.subTest(units=units, slices=slices), self.assertRaises(ValueError):
                check(80, units, slices)

    def test_percentiles_and_nonfinite_metrics_cannot_pass(self):
        shared = dict(throughput_rps=120, p50_ms=30, p99_ms=150)
        for change in ({'p99_ms': 20}, {'p99_ms': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.isolation['isolation_delta'](shared, {**shared, **change})
        check = self.isolation['select_isolation_plan']
        plans = {'too_slow': dict(throughput_rps=100, p99_ms=200, jitter_ms=20)}
        self.assertIsNone(check(plans, 100, 90, 30)['selected'])
        with self.assertRaises(ValueError):
            check({'bad': dict(throughput_rps=100, p99_ms=float('nan'), jitter_ms=2)}, 100, 90, 30)

    def test_idle_power_cost_remains_when_no_work_is_done(self):
        report = self.cost['tco_ledger'](100, 1, 1, 10, 20, 1, 0, idle_power_kw=0.1)
        self.assertEqual(report['powered_hours'], 720)
        self.assertEqual(report['busy_hours'], 0)
        self.assertEqual(report['energy_kwh'], 72)
        self.assertEqual(report['total_cost'], 202)

    def test_successful_request_denominator_includes_both_rates(self):
        check = self.cost['cost_per_million_successful_requests']
        baseline = check(1000, 10, 1, 1)
        self.assertEqual(check(1000, 10, 1, 1, utilization=0.5), 2 * baseline)
        self.assertEqual(check(1000, 10, 1, 1, success_rate=0.5), 2 * baseline)
        self.assertEqual(check(1000, 10, 0.5, 1), 2 * baseline)
        with self.assertRaises(ValueError):
            check(1000, 10, 1, 1, success_rate=0)

    def test_nonfinite_cost_and_invalid_power_are_rejected(self):
        with self.assertRaises(ValueError):
            self.cost['tco_ledger'](100, 1, 1, 0, 0, 1, 0.5, idle_power_kw=2)
        with self.assertRaises(ValueError):
            self.cost['migration_cost'](1, 100, 1, 1, 1000, float('nan'))
        with self.assertRaises(ValueError):
            self.cost['effective_cost_per_used_hour'](100, 10, 10, 12, 24, float('nan'))

    def test_prepare_stage_does_not_inherit_model_memory_or_precision(self):
        plan = dict(estimated_peak_gb=40, precision='bf16', runtime='vllm',
                    stages={'prepare': self.prepare_plan()})
        report = self.runtime['assign_stages']({'small_cpu': self.cpu_profile()}, plan)
        self.assertEqual(report['prepare']['candidate_devices'], ['small_cpu'])

    def test_stage_runtime_is_explicit_not_inferred_from_name(self):
        report = self.runtime['assign_stages'](
            {'cpu': self.cpu_profile()}, {'stages': {'renamed_stage': self.prepare_plan()}})
        self.assertEqual(report['renamed_stage']['candidate_devices'], ['cpu'])
        with self.assertRaises(ValueError):
            self.runtime['assign_stages']({'cpu': self.cpu_profile()}, {'stages': {'prepare': {}}})

    def test_stage_declarations_are_strict_and_communication_is_conditional(self):
        check = self.runtime['check_device_feasibility']
        self.assertTrue(check(self.cpu_profile(), self.prepare_plan())['declared_requirements_met'])
        self.assertFalse(check(self.cpu_profile(), {
            **self.prepare_plan(), 'requires_collective': True})['collective_ok'])
        for change in ({'estimated_peak_gb': float('nan')}, {'requires_collective': 'yes'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check(self.cpu_profile(), {**self.prepare_plan(), **change})

    def test_admission_rejects_missing_or_unmatched_stages(self):
        check = self.runtime['benchmark_admission']
        stages = {'prepare': {'candidate_devices': ['cpu']}}
        self.assertEqual(check({}, ['prepare'], True, True)['status'], 'blocked')
        self.assertEqual(check(stages, ['prepare', 'generate'], True, True)['status'], 'blocked')
        self.assertEqual(check({'prepare': {'candidate_devices': []}},
                               ['prepare'], True, True)['status'], 'blocked')
        with self.assertRaises(ValueError):
            check(stages, [], True, True)

    def test_smoke_test_precedes_benchmark_but_no_benchmark_is_required(self):
        check = self.runtime['benchmark_admission']
        stages = {'prepare': {'candidate_devices': ['cpu']}}
        self.assertEqual(check(stages, ['prepare'], True, False)['status'], 'needs_runtime_check')
        self.assertEqual(check(stages, ['prepare'], False, True)['status'], 'blocked')
        self.assertEqual(check(stages, ['prepare'], True, True)['status'], 'benchmark_ready')
        with self.assertRaises(ValueError):
            check(stages, ['prepare'], 'passed', True)


if __name__ == '__main__':
    unittest.main()
