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

    def test_accelerator_screen_numeric_fields_reject_booleans(self):
        """Vary one numeric field at a time; declaration flags remain booleans."""
        capacity = dict(required_peak_bytes=8e9, device_memory_bytes=16e9,
                        required_ops={'matmul'}, supported_ops={'matmul'},
                        required_dtype='bf16', supported_dtypes={'bf16'})
        for field in ('required_peak_bytes', 'device_memory_bytes'):
            for value in (True, False, '16', float('inf')):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.devices['accelerator_feasibility'](**(capacity | {field: value}))
        components = dict(driver=True, compiler=True, runtime=True,
                          kernel=True, communication=False)
        for value in (True, False, '1', float('nan')):
            with self.subTest(coverage=value), self.assertRaises(ValueError):
                self.devices['stack_readiness'](**components, kernel_coverage=value)
        limits = dict(feasible=True, stack_ready=True, migration_weeks=2,
                      max_migration_weeks=8, supply_risk=0.2, max_supply_risk=0.5,
                      power_kw=0.7, power_cap_kw=1)
        for field in set(limits) - {'feasible', 'stack_ready'}:
            for value in (True, False):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.devices['admit_accelerator_candidate'](**(limits | {field: value}))

    def test_capacity_operator_and_precision_checks_fail_independently(self):
        """A single unsupported property produces its corresponding reason only."""
        inputs = dict(required_peak_bytes=8e9, device_memory_bytes=16e9,
                      required_ops={'matmul', 'softmax'}, supported_ops={'matmul', 'softmax'},
                      required_dtype='bf16', supported_dtypes={'bf16'})
        check = self.devices['accelerator_feasibility']
        self.assertTrue(check(**inputs)['feasible'])
        for change, reason in (({'required_peak_bytes': 17e9}, 'insufficient_memory'),
                               ({'supported_ops': {'matmul'}}, 'missing_ops'),
                               ({'supported_dtypes': {'fp16'}}, 'unsupported_dtype')):
            with self.subTest(change=change):
                report = check(**(inputs | change))
                self.assertFalse(report['feasible'])
                self.assertEqual(report['reasons'], [reason])

    def test_candidate_rejection_reasons_are_independently_observable(self):
        """Each threshold has a passing control and its own single-change failure."""
        inputs = dict(feasible=True, stack_ready=True, migration_weeks=2,
                      max_migration_weeks=8, supply_risk=0.2, max_supply_risk=0.5,
                      power_kw=0.7, power_cap_kw=1)
        check = self.devices['admit_accelerator_candidate']
        self.assertEqual(check(**inputs), {'admitted': True, 'reasons': []})
        cases = [('feasible', False, 'device_constraints'),
                 ('stack_ready', False, 'software_stack'),
                 ('migration_weeks', 9, 'migration_schedule'),
                 ('supply_risk', 0.6, 'supply_risk'),
                 ('power_kw', 1.1, 'power_cap')]
        for field, value, reason in cases:
            with self.subTest(field=field):
                self.assertEqual(check(**(inputs | {field: value})),
                                 {'admitted': False, 'reasons': [reason]})
        self.assertTrue(check(**(inputs | {'migration_weeks': 8,
                                          'supply_risk': 0.5, 'power_kw': 1}))['admitted'])

    def test_capacity_match_does_not_supply_missing_target_artifacts(self):
        """Keep hardware declarations fixed while changing target software readiness."""
        hardware = self.devices['accelerator_feasibility'](
            8e9, 16e9, {'matmul', 'softmax'}, {'matmul', 'softmax'}, 'bf16', {'bf16'})
        self.assertTrue(hardware['feasible'])
        components = dict(driver=True, compiler=True, runtime=True,
                          kernel=True, communication=False)
        self.assertTrue(self.devices['stack_readiness'](**components)['declared_requirements_met'])
        missing = self.devices['stack_readiness'](
            **(components | {'compiler': False, 'kernel': False}))
        self.assertEqual(missing['missing'], ['compiler', 'kernel'])
        self.assertFalse(missing['declared_requirements_met'])
        candidate = self.devices['admit_accelerator_candidate'](
            feasible=hardware['feasible'], stack_ready=missing['declared_requirements_met'],
            migration_weeks=6, max_migration_weeks=8, supply_risk=0.4,
            max_supply_risk=0.5, power_kw=0.5, power_cap_kw=1)
        self.assertEqual(candidate['reasons'], ['software_stack'])

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

    def test_partition_and_latency_numeric_fields_reject_booleans(self):
        """Reject flags in capacity and measurement fields before doing arithmetic."""
        inputs = dict(total_memory_gb=12, total_compute_units=8,
                      slices=2, reserved_memory_gb=0)
        for field in inputs:
            for value in (True, False):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.isolation['partition_resources'](**(inputs | {field: value}))
        metrics = dict(throughput_rps=120, p50_ms=30, p99_ms=150)
        for field in metrics:
            for value in (True, False):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.isolation['isolation_delta'](metrics, metrics | {field: value})
        thresholds = dict(max_p99_ms=100, min_throughput_rps=90, max_jitter_ms=20)
        candidate = dict(throughput_rps=90, p99_ms=100, jitter_ms=20)
        check = self.isolation['select_isolation_plan']
        for field in thresholds:
            for value in (True, False):
                with self.subTest(threshold=field, value=value), self.assertRaises(ValueError):
                    check({'A': candidate}, **(thresholds | {field: value}))
        for field in candidate:
            for value in (True, False):
                with self.subTest(metric=field, value=value), self.assertRaises(ValueError):
                    check({'A': candidate | {field: value}}, **thresholds)

    def test_isolation_thresholds_reject_each_failure_independently(self):
        """Avoid masking a broken gate with a second failing metric."""
        metrics = dict(throughput_rps=90, p99_ms=100, jitter_ms=20)
        check = self.isolation['select_isolation_plan']
        for field, value, reason in (('p99_ms', 101, 'p99_slo'),
                                     ('throughput_rps', 89, 'throughput'),
                                     ('jitter_ms', 21, 'jitter')):
            with self.subTest(field=field):
                result = check({'A': metrics | {field: value}}, 100, 90, 20)
                self.assertIsNone(result['selected'])
                self.assertEqual(result['rejected'], {'A': [reason]})

    def test_partition_fit_distinguishes_total_capacity_and_per_task_capacity(self):
        """Idle capacity in another fixed partition does not satisfy an oversized task."""
        check = self.isolation['partition_fit']
        tasks = {'A': 8, 'B': 2}
        equal = check(tasks, {'A': 6, 'B': 6})
        self.assertEqual(equal['A'], {'fits': False, 'free_gb': 0, 'shortfall_gb': 2})
        self.assertEqual(equal['B']['free_gb'], 4)
        self.assertTrue(all(row['fits'] for row in check(tasks, {'A': 8, 'B': 4}).values()))
        with self.assertRaises(ValueError):
            check(tasks, {'A': 12})
        for value in (True, -1, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check({'A': value}, {'A': 6})

    def test_isolation_ratios_and_exact_service_boundaries(self):
        """Check exact metric changes and inclusive limits, rather than only their signs."""
        delta = self.isolation['isolation_delta'](
            {'throughput_rps': 120, 'p50_ms': 30, 'p99_ms': 150},
            {'throughput_rps': 108, 'p50_ms': 32, 'p99_ms': 80})
        self.assertAlmostEqual(delta['throughput_change_ratio'], -0.1)
        self.assertAlmostEqual(delta['p99_reduction_ratio'], 7 / 15)
        self.assertEqual(delta['isolated_tail_amplification'], 2.5)
        metric = {'throughput_rps': 90, 'p99_ms': 100, 'jitter_ms': 20}
        check = self.isolation['select_isolation_plan']
        self.assertEqual(check({'equal': metric}, 100, 90, 20)['selected'], 'equal')
        self.assertIsNone(check({'slow': metric | {'throughput_rps': 89}}, 100, 90, 20)['selected'])

    def test_handoff_preserves_token_count_but_changes_storage_and_copy_payload(self):
        """The widened input and original IDs coexist on host before the assumed transfer."""
        check = self.runtime['handoff_ledger']
        report = check(4096, 'int32', 'int64', 16)
        self.assertEqual(report['source_bytes'], 16384)
        self.assertEqual(report['target_bytes'], 32768)
        self.assertEqual(report['host_live_bytes'], 49152)
        self.assertAlmostEqual(report['transfer_ms'], 0.002048)
        self.assertEqual(check(4096, 'int64', 'int64', 16)['host_live_bytes'], 32768)
        self.assertAlmostEqual(check(8192, 'int32', 'int64', 16)['transfer_ms'],
                               2 * report['transfer_ms'])
        self.assertAlmostEqual(check(4096, 'int32', 'int64', 32)['transfer_ms'],
                               report['transfer_ms'] / 2)

    def test_handoff_rejects_unsupported_conversion_and_invalid_bandwidth(self):
        check = self.runtime['handoff_ledger']
        for dtype in ('int32', 'fp32'):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                check(4096, 'int64', dtype, 16)
        for bandwidth in (0, True, float('nan')):
            with self.subTest(bandwidth=bandwidth), self.assertRaises(ValueError):
                check(4096, 'int32', 'int64', bandwidth)

    def test_joint_residency_includes_new_input_and_checks_exact_capacity(self):
        check = self.runtime['placement_memory_check']
        self.assertEqual(check(40_000_000_000, 32768, 40_000_000_000),
                         {'peak_bytes': 40_000_032_768, 'fits': False, 'shortfall_bytes': 32768})
        self.assertTrue(check(40_000_000_000, 32768, 40_000_032_768)['fits'])
        self.assertTrue(check(0, 0, 0)['fits'])
        with self.assertRaises(ValueError):
            check(True, 32768, 48_000_000_000)

    def test_idle_power_cost_remains_when_no_work_is_done(self):
        report = self.cost['tco_ledger'](100, 1, 1, 10, 20, 1, 0, idle_power_kw=0.1)
        self.assertEqual(report['powered_hours'], 720)
        self.assertEqual(report['busy_hours'], 0)
        self.assertEqual(report['energy_kwh'], 72)
        self.assertEqual(report['total_cost'], 202)

    def test_cost_and_request_models_share_available_busy_time(self):
        ledger = self.cost['tco_ledger'](
            purchase_cost=100, power_kw=1, electricity_per_kwh=1,
            ops_cost_per_month=10, migration_cost=20, months=1,
            utilization=0.5, idle_power_kw=0.1, availability=0.5)
        self.assertEqual(ledger['available_hours'], 360)
        self.assertEqual(ledger['busy_hours'], 180)
        self.assertEqual(ledger['idle_hours'], 540)
        self.assertEqual(ledger['energy_kwh'], 234)
        self.assertEqual(ledger['total_cost'], 364)
        cost = self.cost['cost_per_million_successful_requests'](
            total_cost=ledger['total_cost'], requests_per_second=2,
            availability=0.5, months=1, utilization=0.5, success_rate=0.5)
        self.assertAlmostEqual(cost, 364 / 648000 * 1000000)

    def test_migration_labor_and_conditional_risk_feed_total_cost(self):
        inputs = dict(engineering_hours=120, hourly_rate=100, validation_hours=40,
                      downtime_hours=2, downtime_cost_per_hour=5000)
        low = self.cost['migration_cost'](**inputs, rollback_probability=0.05)
        high = self.cost['migration_cost'](**inputs, rollback_probability=0.5)
        self.assertEqual(low, {'labor_cost': 16000, 'expected_rollback_cost': 500,
                               'total_cost': 16500})
        self.assertEqual(high['total_cost'], 21000)
        self.assertEqual(self.cost['migration_cost'](**inputs, rollback_probability=0)
                         ['total_cost'], 16000)
        ledger = self.cost['tco_ledger'](
            purchase_cost=100, power_kw=1, electricity_per_kwh=1,
            ops_cost_per_month=10, migration_cost=low['total_cost'], months=1,
            utilization=0.5, idle_power_kw=0.1, availability=0.5)
        self.assertEqual(ledger['components']['migration_cost'], low['total_cost'])
        self.assertEqual(ledger['total_cost'], 16844)

    def test_depreciation_changes_numerator_not_busy_time_or_purchase_count(self):
        cost = self.cost['effective_cost_per_used_hour']
        inputs = dict(purchase_cost=300, operations_cost=30, migration_cost=20,
                      horizon_months=12, utilization=0.5, availability=0.8)
        full = cost(**inputs, depreciation_months=12)
        allocated = cost(**inputs, depreciation_months=36)
        self.assertEqual(allocated['allocated_purchase'], 100)
        self.assertEqual(full['used_hours'], allocated['used_hours'])
        self.assertAlmostEqual(allocated['used_hours'], 3456)
        self.assertAlmostEqual(allocated['cost_per_used_hour'], 150 / 3456)
        self.assertAlmostEqual(full['cost_per_used_hour'], 350 / 3456)
        extended = cost(**(inputs | {'horizon_months': 48}), depreciation_months=36)
        self.assertEqual(extended['allocated_purchase'], 300)

    def test_unavailable_powered_device_still_costs_but_has_no_successful_requests(self):
        ledger = self.cost['tco_ledger'](
            purchase_cost=0, power_kw=1, electricity_per_kwh=1,
            ops_cost_per_month=0, migration_cost=0, months=1,
            utilization=1, idle_power_kw=0.1, availability=0)
        self.assertEqual(ledger['busy_hours'], 0)
        self.assertEqual(ledger['energy_kwh'], 72)
        with self.assertRaises(ValueError):
            self.cost['cost_per_million_successful_requests'](72, 10, 0, 1)

    def test_cost_rates_reject_invalid_values_without_notebook_error_demos(self):
        for availability in (True, 1.1, float('nan')):
            with self.subTest(availability=availability), self.assertRaises(ValueError):
                self.cost['tco_ledger'](100, 1, 1, 10, 20, 1, 0.5,
                                        availability=availability)
        with self.assertRaises(ValueError):
            self.cost['migration_cost'](120, 100, 40, 2, 5000, 1.1)
        with self.assertRaises(ValueError):
            self.cost['effective_cost_per_used_hour'](100, 10, 20, 1, 12, 0.5,
                                                     availability=0)

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

    def test_stage_declaration_checks_fail_independently(self):
        """Keep all other device requirements satisfied while varying one demand."""
        check = self.runtime['check_device_feasibility']
        profile, plan = self.cpu_profile(), self.prepare_plan()
        flags = ('memory_ok', 'precision_ok', 'runtime_ok', 'collective_ok')
        cases = [({'estimated_peak_gb': 3}, 'memory_ok'),
                 ({'precision': 'bf16'}, 'precision_ok'),
                 ({'runtime': 'vllm'}, 'runtime_ok'),
                 ({'requires_collective': True}, 'collective_ok')]
        for change, failed in cases:
            with self.subTest(change=change):
                report = check(profile, plan | change)
                self.assertFalse(report['declared_requirements_met'])
                self.assertFalse(report[failed])
                self.assertTrue(all(report[field] for field in flags if field != failed))
                self.assertEqual(report['missing_ops'], [])
        missing_op = check(profile, plan | {'required_ops': {'tokenize', 'decode'}})
        self.assertFalse(missing_op['declared_requirements_met'])
        self.assertEqual(missing_op['missing_ops'], ['decode'])
        self.assertTrue(all(missing_op[field] for field in flags))

    def test_device_capacity_and_stage_budget_reject_boolean_numbers(self):
        """A flag is not a capacity estimate, even though Python treats bool as int."""
        check = self.runtime['check_device_feasibility']
        for value in (True, False, '2', float('inf')):
            with self.subTest(profile_memory=value), self.assertRaises(ValueError):
                check(self.cpu_profile() | {'memory_gb': value}, self.prepare_plan())
            with self.subTest(stage_memory=value), self.assertRaises(ValueError):
                check(self.cpu_profile(), self.prepare_plan() | {'estimated_peak_gb': value})
        self.assertTrue(check(self.cpu_profile(), self.prepare_plan() | {
            'estimated_peak_gb': 2})['memory_ok'])
        self.assertTrue(check(self.cpu_profile(), self.prepare_plan() | {
            'estimated_peak_gb': 0})['memory_ok'])

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
