"""Check the actual Part01 state, topology and collective teaching calculators."""

import unittest

if __package__:
    from .test_foundation_performance_evidence import notebook_functions
    from .test_foundation_numerics import execute_numerical_notebook
else:
    from test_foundation_performance_evidence import notebook_functions
    from test_foundation_numerics import execute_numerical_notebook


class FoundationCommunicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = '01_Hardware_Math_and_Systems/'
        cls.precision = execute_numerical_notebook('01')
        cls.memory = notebook_functions(prefix + '06_VRAM_Calculation_and_ZeRO.ipynb')
        cls.topology = notebook_functions(prefix + '05_Communication_Topologies.ipynb')
        cls.collective = notebook_functions(prefix + '20_NCCL_and_AllReduce_Basics.ipynb')
        cls.parallel = notebook_functions(prefix + '26_Parallel_Strategy_Decision_Framework.ipynb')
        cls.scheduling = notebook_functions(prefix + '27_Communication_Scheduling_Optimization.ipynb')
        cls.checkpoint = notebook_functions(prefix + '28_Fault_Tolerance_and_Checkpointing.ipynb')

    def test_adam_components_match_the_precision_notebook(self):
        """The two pages count moments and the optional master copy exactly once."""
        for dtype in ('fp16', 'bf16', 'fp32'):
            for master in (False, True):
                with self.subTest(dtype=dtype, master=master):
                    reference = self.precision['training_state_ledger'](
                        7, parameter_dtype=dtype, gradient_dtype=dtype,
                        moment_dtype='fp32', keep_master_weights=master,
                    )
                    actual = self.memory['training_state_breakdown'](
                        7, model_dtype=dtype, keep_master_weights=master,
                    )
                    for name, amount in reference['components'].items():
                        self.assertEqual(actual[name], amount)
                    self.assertEqual(actual['training_state_gb'], reference['total_gb'])
                    self.assertEqual(actual['optimizer_state_gb'], 56)

    def test_default_master_copy_depends_on_parameter_storage(self):
        """FP32 storage has no redundant master by default; the choice remains explicit."""
        ledger = self.memory['training_state_breakdown']
        self.assertEqual(ledger(7, model_dtype='bf16')['master_weights_gb'], 28)
        self.assertEqual(ledger(7, model_dtype='fp32')['master_weights_gb'], 0)
        self.assertEqual(ledger(7, model_dtype='fp32', keep_master_weights=True)['training_state_gb'], 140)
        self.assertEqual(ledger(7, optimizer='sgd')['optimizer_state_gb'], 0)
        self.assertEqual(ledger(7, optimizer='sgd', keep_master_weights=False)['training_state_gb'], 28)

    def test_zero_sharding_reuses_one_component_ledger(self):
        """Use P=14, G=14 and update state=84 GB; divide only sharded groups."""
        calculate = self.memory['calculate_zero_memory']
        for stage, expected in [('ddp', 112), (0, 112), (1, 38.5), (2, 26.25), (3, 14)]:
            with self.subTest(stage=stage):
                self.assertEqual(calculate(7, stage, 8), expected)
                self.assertEqual(calculate(7, stage, 1), 112)
        self.assertEqual(calculate(7, 1, 8, keep_master_weights=False), 35)
        self.assertEqual(calculate(7, 2, 8, keep_master_weights=False), 22.75)
        self.assertEqual(calculate(7, 3, 8, keep_master_weights=False), 10.5)

    def test_state_budget_limit_is_reversible_at_capacity_equality(self):
        """With 64 GB reserved for states, inverse and forward calculations agree."""
        inverse = self.memory['state_capacity_limit_params']
        forward = self.memory['calculate_zero_memory']
        for stage in ('ddp', 1, 2, 3):
            with self.subTest(stage=stage):
                params_b = inverse(80, 8, stage)
                self.assertAlmostEqual(forward(params_b, stage, 8), 64)
        self.assertEqual(inverse(80, 8, 'ddp'), 4)
        self.assertEqual(inverse(80, 8, 3), 32)
        self.assertEqual(inverse(80, 8, 3, overhead_ratio=0), 40)

    def test_state_table_does_not_reapply_master_weight_bytes(self):
        """Disabling the master copy changes the shared baseline and all stage rows."""
        rows = self.memory['compare_zero_strategies'](80, 8, keep_master_weights=False)
        self.assertEqual(rows[0]['per_gpu_state_gb_per_1b'], 12)
        self.assertEqual(rows[-1]['per_gpu_state_gb_per_1b'], 1.5)
        self.assertAlmostEqual(rows[-1]['static_capacity_limit_params_b'], 42.667)
        self.assertEqual(rows[-1]['memory_saving_ratio'], 0.875)

    def test_collective_shape_always_means_actual_input(self):
        """Reduce-scatter consumes the large tensor, unlike all-gather's small input."""
        shape = self.topology['collective_shape']
        for kind, input_shape, expected in (
            ('allreduce', (8, 1024), (8, 1024)),
            ('allgather', (8, 1024), (32, 1024)),
            ('reducescatter', (32, 1024), (8, 1024)),
            ('reducescatter', (8, 1024), (2, 1024)),
        ):
            with self.subTest(kind=kind, input_shape=input_shape):
                result = shape(kind, 4, input_shape)
                self.assertEqual(result['input_per_rank'], input_shape)
                self.assertEqual(result['output_per_rank'], expected)
        with self.assertRaises(ValueError):
            shape('reducescatter', 4, (10, 1024))

    def test_topology_time_uses_mib_payload_and_gib_per_second(self):
        """The slowest pipelined hop limits long-message throughput in this model."""
        estimate = self.topology['path_transfer_lower_bound']
        fast = [{'name': 'fast', 'bandwidth_gib_s': 100}]
        route = fast + [{'name': 'network', 'bandwidth_gib_s': 12.5}]
        self.assertEqual(estimate(256, fast)['lower_bound_ms'], 2.5)
        self.assertEqual(estimate(256, route)['lower_bound_ms'], 20)
        self.assertEqual(estimate(256, route)['bottleneck_link'], 'network')
        self.assertEqual(estimate(0, route)['lower_bound_ms'], 0)

    def test_topology_domains_are_explicit_permissions_not_speed_ranks(self):
        """A cross-node-only rule cannot silently accept an intra-node placement."""
        check = self.topology['check_group_placement']
        groups = [dict(name='group', allowed_domains=['cross_node'], actual_domain='intra_node')]
        self.assertFalse(check(groups)[0]['placement_allowed'])
        groups[0]['actual_domain'] = 'cross_node'
        self.assertTrue(check(groups)[0]['placement_allowed'])
        groups[0]['allowed_domains'] = ['cross_node', 'intra_node']
        self.assertTrue(check(groups)[0]['placement_allowed'])

    def test_ring_send_receive_and_time_do_not_double_count_full_duplex(self):
        """Eight ranks send 448 MiB each; adding receives does not double transfer time."""
        cost = self.collective['ring_cost'](256 * 2**20, 8, 0.01, 100)
        self.assertEqual(cost['rounds'], 14)
        self.assertEqual(cost['send_bytes_per_rank'], 448 * 2**20)
        self.assertEqual(cost['receive_bytes_per_rank'], 448 * 2**20)
        self.assertEqual(cost['send_receive_bytes_per_rank'], 896 * 2**20)
        self.assertAlmostEqual(cost['transfer_ms'], 4.375)
        self.assertAlmostEqual(cost['latency_ms'], 0.14)
        self.assertAlmostEqual(cost['total_ms'], 4.515)
        self.assertAlmostEqual(self.collective['ring_cost'](256 * 2**20, 8, 0.01, 12.5)['total_ms'], 35.14)

    def test_ring_model_rejects_fractional_participants_or_unequal_byte_chunks(self):
        """The equal-chunk teaching formula has explicit rank and divisibility assumptions."""
        cost = self.collective['ring_cost']
        for payload, ranks in ((256, 3), (256, 1), (256, 2.5), (True, 2)):
            with self.subTest(payload=payload, ranks=ranks), self.assertRaises(ValueError):
                cost(payload, ranks, 0.01, 100)

    def test_ring_state_transitions_match_independent_reduction(self):
        """Exercise several ring sizes, negative partial sums and exact chunk owners."""
        simulate = self.collective['ring_allreduce']
        for ranks in (2, 3, 4, 7):
            chunks = [[rank * 10 - chunk for chunk in range(ranks)] for rank in range(ranks)]
            expected = [10 * ranks * (ranks - 1) // 2 - ranks * chunk for chunk in range(ranks)]
            result = simulate(chunks)
            self.assertEqual(result['outputs'], [expected] * ranks)
            self.assertEqual(result['owned_after_reduce_scatter'], dict(enumerate(expected)))
            self.assertEqual(len(result['trace']), 2 * (ranks - 1))


    def candidate(self):
        """Return one feasible candidate so each test changes exactly one condition."""
        return dict(name='candidate', memory_per_rank_gib=24, modes={'training'},
                    model_kinds={'dense'}, allowed_link_scopes={'cross_node'})

    def environment(self):
        """Keep capacity equality and a cross-node rule as the passing control."""
        return dict(memory_per_device_gib=24, mode='training',
                    model_kind='dense', link_scope='cross_node')

    def test_parallel_candidate_checks_fail_independently(self):
        """Memory, mode, structure and placement each have their own rejection reason."""
        check = self.parallel['evaluate_candidate']
        self.assertTrue(check(self.candidate(), self.environment())['conditions_met'])
        cases = [
            ('memory_per_rank_gib', 25, 'per-rank memory exceeds device memory'),
            ('modes', {'serving'}, 'workload mode is unsupported'),
            ('model_kinds', {'moe'}, 'model structure is unsupported'),
            ('allowed_link_scopes', {'intra_node'}, 'placement domain is not allowed'),
        ]
        for field, value, reason in cases:
            with self.subTest(field=field):
                report = check(self.candidate() | {field: value}, self.environment())
                self.assertFalse(report['conditions_met'])
                self.assertEqual(report['reasons'], [reason])
        self.assertFalse(check(self.candidate(), self.environment() | {'link_scope': 'intra_node'})['conditions_met'])

    def test_parallel_communication_compares_equal_total_send_bytes(self):
        """Changing message count leaves the transfer term unchanged at fixed total bytes."""
        estimate = self.parallel['communication_ledger']
        large = estimate(256, 1, 100, 5)
        small = estimate(32, 8, 100, 5)
        self.assertEqual(large['total_payload_mib'], small['total_payload_mib'])
        self.assertEqual(large['transfer_ms'], small['transfer_ms'])
        self.assertEqual((large['lower_bound_ms'], small['lower_bound_ms']), (2.505, 2.54))

    def test_parallel_product_requires_independent_axes(self):
        """A shared EP/DP group must not become an invented product of rank counts."""
        check = self.parallel['validate_parallel_composition']
        result = check({'DP': 2, 'TP': 4, 'PP': 2}, 16, 'dense', independent_axes=True)
        self.assertTrue(result['valid'])
        self.assertEqual(result['required_ranks'], 16)
        shared = check({'DP': 4, 'EP': 2}, 4, 'moe', independent_axes=False)
        self.assertFalse(shared['valid'])
        self.assertIsNone(shared['required_ranks'])
        self.assertEqual(shared['issues'], ['需要实际进程组映射'])

    def test_parallel_count_structure_and_role_checks_are_separate(self):
        """Each composition counterexample has one failure, not several masking each other."""
        check = self.parallel['validate_parallel_composition']
        shortage = check({'TP': 4}, 2, 'dense', independent_axes=True)
        structure = check({'EP': 2}, 2, 'dense', independent_axes=True)
        roles = check({'DP': 2, 'Replica': 2}, 4, 'dense', independent_axes=True)
        self.assertEqual(shortage['issues'], ['requires 4 ranks but only 2 are available'])
        self.assertEqual(structure['issues'], ['EP degree greater than 1 requires an MoE model'])
        self.assertEqual(roles['issues'], ['Replica and training DP must use distinct execution roles'])
        with self.assertRaises(ValueError):
            check({'TP': True}, 2, 'dense', independent_axes=True)

    def layers(self, ready_together=False):
        """Use the same 128 MiB of gradients in both readiness scenarios."""
        return [dict(name=f'L{i}', ready_ms=0 if ready_together else ready, size_mib=32)
                for i, ready in enumerate((2, 5, 9, 14))]

    def test_bucket_waiting_can_outweigh_lower_launch_cost(self):
        """Count Ring send bytes before comparing tails for staggered-ready gradients."""
        schedule = self.scheduling['schedule_gradient_buckets']
        for cap, count, tail in ((32, 4, 0.9455), (64, 2, 1.883), (128, 1, 3.758)):
            with self.subTest(cap=cap):
                result = schedule(self.layers(), cap)
                self.assertEqual(result['bucket_count'], count)
                self.assertEqual(result['total_send_mib_per_rank'], 192)
                self.assertEqual(result['exposed_after_backward_ms'], tail)
                self.assertEqual(result['communication_end_ms'], round(14 + tail, 4))

    def test_bucket_merging_saves_launches_when_all_inputs_are_ready(self):
        """Remove readiness delay without changing gradient bytes or link conditions."""
        schedule = self.scheduling['schedule_gradient_buckets']
        small = schedule(self.layers(ready_together=True), 32)
        large = schedule(self.layers(ready_together=True), 128)
        self.assertEqual(small['communication_end_ms'], 3.782)
        self.assertEqual(large['communication_end_ms'], 3.758)
        self.assertEqual(small['total_send_mib_per_rank'], large['total_send_mib_per_rank'])

    def test_buckets_respect_fixed_layer_order_and_capacity(self):
        """No sorting by local readiness may silently change collective ordering."""
        schedule = self.scheduling['schedule_gradient_buckets']
        layers = [dict(name='first', ready_ms=10, size_mib=20),
                  dict(name='second', ready_ms=1, size_mib=20)]
        result = schedule(layers, 32)
        self.assertEqual([row['layers'] for row in result['timeline']], [['first'], ['second']])
        self.assertGreaterEqual(result['timeline'][1]['start_ms'], result['timeline'][0]['end_ms'])
        with self.assertRaises(ValueError):
            schedule(layers, 16)

    def test_scheduling_send_factor_matches_the_ring_cost_not_receive_sum(self):
        """The 27 and 20 pages must use the same single-direction accounting."""
        send_mib = self.scheduling['ring_allreduce_payload_mib'](256, 8)
        ring = self.collective['ring_cost'](256 * 2**20, 8, 0.01, 100)
        self.assertEqual(send_mib, ring['send_bytes_per_rank'] / 2**20)
        phase = self.scheduling['communication_time_ms'](send_mib, 100, 140)
        self.assertAlmostEqual(phase['communication_ms'], ring['total_ms'])

    def test_alltoall_zero_remote_load_has_no_imbalance_ratio(self):
        """Both an empty route and an all-local route have zero network payload."""
        ledger = self.scheduling['alltoall_route_ledger']
        for matrix in ([[0, 0], [0, 0]], [[100, 0], [0, 1]]):
            with self.subTest(matrix=matrix):
                result = ledger(matrix, 4096)
                self.assertEqual(result['remote_send_mib'], 0)
                self.assertEqual(result['remote_receive_assignments_per_rank'], [0, 0])
                self.assertIsNone(result['max_to_mean_remote_receive_ratio'])

    def test_alltoall_separates_remote_receive_and_local_work(self):
        """The diagonal contributes expert work but not network transfers."""
        result = self.scheduling['alltoall_route_ledger']([[100, 3], [7, 200]], 4)
        self.assertEqual(result['remote_send_assignments_per_rank'], [3, 7])
        self.assertEqual(result['remote_receive_assignments_per_rank'], [7, 3])
        self.assertEqual(result['received_assignments_including_local'], [107, 203])
        self.assertEqual(result['remote_send_mib'], 40 / 2**20)
        self.assertEqual(result['max_to_mean_remote_receive_ratio'], 1.4)

    def test_interval_union_counts_duplicate_coverage_once(self):
        """The ideal interval model neither double-counts overlap nor invents a zero-time ratio."""
        inspect = self.scheduling['communication_exposure']
        result = inspect((0, 12), [(1, 4), (3, 8), (1, 4)])
        self.assertEqual((result['hidden_ms'], result['exposed_ms']), (7, 5))
        self.assertIsNone(inspect((2, 2), [(0, 4)])['overlap_ratio'])

    def checkpoint_entries(self):
        """Return metadata for the teaching configuration, not actual files on disk."""
        return [dict(role=role, path=f'{role}.bin', size_mib=1)
                for role in ('model', 'optimizer', 'scheduler', 'rng', 'progress')]

    def checkpoint_manifest(self):
        """Use one declared rank package per saved rank and a complete role summary."""
        return dict(world_size=4, checkpoint_id='step-1200', committed=True,
                    entries=self.checkpoint_entries())

    def checkpoint_shards(self):
        """Use a complete unique rank set as the passing control."""
        return [dict(rank=rank, checkpoint_id='step-1200', size_mib=900) for rank in range(4)]

    def test_manifest_check_is_explicitly_metadata_only(self):
        """Nonexistent illustrative paths can satisfy metadata, never prove successful loading."""
        inspect = self.checkpoint['inspect_checkpoint_manifest']
        result = inspect(self.checkpoint_entries(), committed=True)
        self.assertTrue(result['manifest_checks_passed'])
        self.assertNotIn('recoverable', result)
        self.assertFalse(inspect(self.checkpoint_entries(), committed=False)['manifest_checks_passed'])
        missing = inspect(self.checkpoint_entries()[:1], committed=True)
        self.assertEqual(missing['missing_roles'], ['optimizer', 'progress', 'rng', 'scheduler'])

    def test_manifest_configuration_adds_required_state_roles(self):
        """A run using a scaler must explicitly include that state in the required set."""
        inspect = self.checkpoint['inspect_checkpoint_manifest']
        required = {'model', 'optimizer', 'scheduler', 'rng', 'progress', 'scaler'}
        result = inspect(self.checkpoint_entries(), True, required_roles=required)
        self.assertEqual(result['missing_roles'], ['scaler'])
        duplicate_path = self.checkpoint_entries()
        duplicate_path[1]['path'] = duplicate_path[0]['path']
        self.assertEqual(inspect(duplicate_path, True)['duplicate_paths'], ['model.bin'])

    def test_rollback_mean_accounts_for_partial_and_short_windows(self):
        """Integrate time since the most recent checkpoint over each full and tail interval."""
        mean = self.checkpoint['mean_rollback_minutes']
        self.assertEqual(mean(120, 60), 30)
        self.assertEqual(mean(90, 60), 25)
        self.assertEqual(mean(30, 60), 15)
        self.assertEqual(mean(0, 60), 0)
        self.assertEqual(self.checkpoint['expected_failure_overhead'](72, 0.02, 60, 8)
                         ['expected_failure_overhead_minutes'], 54.72)

    def test_checkpoint_cost_counts_only_pre_end_recovery_saves(self):
        """Final artifact delivery is excluded consistently from the declared save model."""
        cost = self.checkpoint['checkpoint_interval_cost']
        result = cost(1440, 120, 2, 0.02, 8)
        self.assertEqual(result['save_count'], 11)
        self.assertEqual(result['total_overhead_minutes'], 54.64)
        self.assertEqual(cost(60, 60, 2, 0, 8)['save_count'], 0)
        self.assertEqual(cost(61, 60, 2, 0, 8)['save_count'], 1)
        self.assertEqual(cost(0, 60, 2, 0.02, 8)['total_overhead_minutes'], 0)

    def test_shard_count_uniqueness_coverage_id_and_size_fail_separately(self):
        """A complete rank set plus an extra duplicate must fail as well as a missing rank."""
        check = self.checkpoint['validate_sharded_checkpoint']
        manifest, shards = self.checkpoint_manifest(), self.checkpoint_shards()
        self.assertTrue(check(manifest, shards)['metadata_checks_passed'])
        self.assertIn('shard count differs from saved world size', check(manifest, shards + [shards[0]])['issues'])
        duplicated = [shards[0], shards[0], *shards[2:]]
        self.assertIn('rank IDs are duplicated', check(manifest, duplicated)['issues'])
        self.assertIn('rank coverage differs from saved world size', check(manifest, shards[:-1])['issues'])
        changed_id = [shards[0] | {'checkpoint_id': 'other'}, *shards[1:]]
        self.assertEqual(check(manifest, changed_id)['issues'], ['checkpoint IDs are inconsistent'])
        empty = [shards[0] | {'size_mib': 0}, *shards[1:]]
        self.assertEqual(check(manifest, empty)['issues'], ['one or more shard sizes are invalid'])

    def test_shards_also_require_the_state_role_manifest(self):
        """Complete rank coverage does not compensate for a missing optimizer role."""
        manifest = self.checkpoint_manifest()
        manifest['entries'] = manifest['entries'][:1]
        result = self.checkpoint['validate_sharded_checkpoint'](manifest, self.checkpoint_shards())
        self.assertFalse(result['metadata_checks_passed'])
        self.assertEqual(result['issues'], ['state manifest checks failed'])

    def test_reshard_requirement_is_separate_from_saved_metadata_consistency(self):
        """A different target rank count reports extra restore work, not a success claim."""
        check = self.checkpoint['validate_sharded_checkpoint']
        result = check(self.checkpoint_manifest(), self.checkpoint_shards(), target_world_size=2)
        self.assertTrue(result['metadata_checks_passed'])
        self.assertTrue(result['requires_reshard'])
        self.assertNotIn('recoverable', result)
        with self.assertRaises(ValueError):
            check(self.checkpoint_manifest() | {'world_size': 0}, [])


if __name__ == '__main__':
    unittest.main()
