"""Test pure decision helpers in Part 00/01 without executing profiler cells.

The canonical notebook check still executes every code cell in order. These
tests add boundary cases against the actual definitions, not copied answers.
"""

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import unittest
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]


def notebook_functions(relative_path):
    path = ROOT / relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            functions.extend(
                node for node in ast.parse("".join(cell["source"])).body
                if isinstance(node, ast.FunctionDef)
                or (isinstance(node, ast.ClassDef) and node.name == 'TaskStage')
            )
    namespace = {
        "__name__": __name__, "math": math, "statistics": statistics,
        "dataclass": dataclass, "List": List, "Dict": Dict, "Tuple": Tuple,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class FoundationPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.debug = notebook_functions("00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb")
        cls.timing = notebook_functions("00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb")
        cls.memory = notebook_functions("00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb")
        cls.design = notebook_functions("00_Prerequisites/20_Performance_Evidence_and_Experiment_Design.ipynb")
        cls.resource = notebook_functions("01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb")
        cls.scale = notebook_functions("01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb")
        cls.sram = notebook_functions("01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb")
        cls.stream = notebook_functions("01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb")
        cls.schedule = notebook_functions("01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.ipynb")
        cls.placement = notebook_functions("01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb")
        cls.launch = notebook_functions("01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb")
        cls.shared = notebook_functions("01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb")
        cls.mapping = notebook_functions("01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb")
        cls.graph = notebook_functions("01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb")
        cls.fusion = notebook_functions("01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb")
        cls.shapes = notebook_functions("01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.ipynb")
        cls.compiler = notebook_functions("01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.ipynb")

    def runs(self, values, quality=0.9):
        return [{"ok": True, "latency_ms": value, "quality": quality} for value in values]

    def validity(self):
        return self.debug["build_validity_summary"]([])

    def bundle(self):
        workload = {"model": "TinyLM", "scope": "training_step"}
        return {
            "validity": self.validity(),
            "time": {
                "evidence_type": "time", "status": "constructed_example",
                "workload": workload, "hotspot": {"name": "backward", "share_pct": 50},
            },
            "memory": {
                "evidence_type": "memory", "status": "constructed_example", "unit": "MiB",
                "workload": dict(workload), "dominant_object": "activation",
                "reserved_mib": 95, "capacity_mib": 100,
            },
        }

    def resource_record(self):
        return {
            "flops": 4e10, "dram_bytes": 8e9, "kernel_ms": 20,
            "transfer_ms": 3, "sync_ms": 2, "launch_ms": 1,
            "time_model": "nonoverlapping_exposed_intervals",
        }

    def test_correctness_record_rejects_contradictions(self):
        for change in ({"issues": ["numerical"]}, {"run_valid": "True"}, {"first_failure": "loss"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.design["validate_validity_report"]({**self.validity(), **change})

    def test_invalid_run_stops_without_performance_fields(self):
        invalid = self.debug["build_validity_summary"](["numerical"], first_failure="loss")
        result = self.design["synthesize_evidence"]({"validity": invalid})
        self.assertEqual(result["status"], "blocked_by_correctness")

    def test_constructed_sources_remain_explicit(self):
        result = self.design["synthesize_evidence"](self.bundle())
        self.assertEqual(result["observation_sources"], {
            "time": "constructed_example", "memory": "constructed_example"
        })

    def test_different_workloads_cannot_be_combined(self):
        bundle = self.bundle()
        bundle["memory"]["workload"] = {"model": "OtherModel"}
        with self.assertRaises(ValueError):
            self.design["synthesize_evidence"](bundle)

    def test_timeouts_are_in_the_failure_denominator(self):
        candidate = self.runs([80] * 5) + [{"ok": False, "failure": "timeout"}] * 50
        result = self.design["evaluate_regression"](self.runs([100] * 5), candidate)
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["candidate"]["failure_rate"], 50 / 55)

    def test_clean_candidate_can_pass(self):
        result = self.design["evaluate_regression"](self.runs([100] * 5), self.runs([80] * 5))
        self.assertEqual(result["decision"], "accept")

    def test_tail_and_quality_guards(self):
        for candidate, failed_check in (
            (self.runs([80, 80, 80, 80, 200]), "p95_ok"),
            (self.runs([80] * 5, quality=0.5), "quality_ok"),
        ):
            with self.subTest(check=failed_check):
                result = self.design["evaluate_regression"](self.runs([100] * 5), candidate)
                self.assertEqual(result["decision"], "reject")
                self.assertIn(failed_check, result["failed_checks"])

    def test_empty_and_short_runs_need_more_measurements(self):
        for candidate in ([], self.runs([80] * 2)):
            result = self.design["evaluate_regression"](self.runs([100] * 5), candidate)
            self.assertEqual(result["decision"], "rerun")

    def test_even_count_median_and_nearest_rank_p95(self):
        summary = self.design["summarize_repeated_runs"](self.runs([1, 2, 3, 100]))
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertEqual(summary["p95_ms"], self.timing["nearest_rank"]([1, 2, 3, 100], 0.95))

    def test_nonfinite_runs_are_rejected(self):
        for value in (float("nan"), float("inf"), -1, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.design["summarize_repeated_runs"](self.runs([value]))

    def test_memory_keeps_multiple_signals_and_signed_difference(self):
        result = self.memory["reconcile_memory_record"](20, 23, 30, 31)
        self.assertEqual(len(result["signals"]), 3)
        self.assertEqual(self.memory["reconcile_memory_record"](20, 18, 19, 30)["estimate_difference_mib"], -2)

    def test_low_efficiency_is_not_confirmed_bandwidth_saturation(self):
        result = self.resource["attribute_bottleneck"](self.resource_record(), 120, 3000)
        self.assertEqual(result["roofline_bound"], "memory")
        self.assertEqual(result["primary_hypothesis"], "kernel_underutilized")

    def test_inconsistent_reference_requires_recheck(self):
        result = self.resource["attribute_bottleneck"](self.resource_record(), 1, 100)
        self.assertEqual(result["primary_hypothesis"], "recheck_reference")

    def test_overlapping_intervals_cannot_be_summed(self):
        with self.assertRaises(ValueError):
            self.resource["derive_resource_metrics"]({**self.resource_record(), "time_model": "overlap"})

    def test_experiment_selects_one_concrete_variable(self):
        attribution = {"primary_hypothesis": "kernel_memory"}
        self.assertEqual(self.resource["build_discrimination_experiment"](attribution, "tile_size")["single_change"], "tile_size")
        with self.assertRaises(ValueError):
            self.resource["build_discrimination_experiment"](attribution, "tile_or_layout")

    def test_scale_checks_propagate_wrong_results_and_runtime_errors(self):
        for function_name, test_name in (
            ("calculate_transformer_params", "test_calculate_transformer_params"),
            ("calculate_forward_flops", "test_calculate_flops"),
        ):
            original = self.scale[function_name]
            try:
                self.scale[function_name] = lambda *args, **kwargs: -1
                with self.subTest(function=function_name), self.assertRaises(AssertionError):
                    self.scale[test_name]()

                def broken(*args, **kwargs):
                    raise RuntimeError("deliberate regression probe")

                self.scale[function_name] = broken
                with self.subTest(function=function_name), self.assertRaises(RuntimeError):
                    self.scale[test_name]()
            finally:
                self.scale[function_name] = original

    def test_reuse_traffic_counts_bytes_separately_from_sync(self):
        report = self.sram["reuse_traffic_ledger"](1024, 4, sync_points=3)
        self.assertEqual(report["global_read_bytes_saved"], 3072)
        self.assertEqual(report["shared_write_bytes"], 1024)
        self.assertEqual(report["shared_read_bytes"], 4096)
        self.assertEqual(report["sync_points"], 3)
        self.assertEqual(self.sram["reuse_traffic_ledger"](1024, 1)["global_read_bytes_saved"], 0)

    def test_buffering_and_padding_consume_shared_memory(self):
        ledger = self.sram["tile_resource_ledger"]
        self.assertEqual(ledger(32, 32, 32)["shared_bytes_per_block"], 8192)
        self.assertEqual(ledger(32, 32, 32, buffers=2)["resident_blocks_upper_bound"], 4)
        self.assertEqual(ledger(32, 32, 32, padding=1)["shared_bytes_per_block"], 8320)
        self.assertEqual(ledger(32, 32, 32, padding=1)["resident_blocks_upper_bound"], 7)

    def test_residency_uses_the_tightest_resource(self):
        ledger = self.sram["tile_resource_ledger"]
        self.assertEqual(ledger(1, 1, 1, max_blocks_per_sm=3)["resident_blocks_upper_bound"], 3)
        self.assertEqual(ledger(1, 1, 1, threads_per_block=1024)["resident_blocks_upper_bound"], 2)
        self.assertEqual(ledger(128, 128, 128)["resident_blocks_upper_bound"], 0)

    def test_register_excess_is_not_reported_as_actual_spill(self):
        report = self.sram["register_resource_ledger"](80)
        self.assertEqual(report["registers_per_block"], 20480)
        self.assertEqual(report["excess_registers_per_thread"], 16)
        self.assertEqual(report["resident_blocks_upper_bound"], 3)
        self.assertNotIn("spill", report)
        self.assertNotIn("net_gain_proxy", report)

    def test_resource_ledgers_reject_invalid_counts(self):
        for name, args, kwargs in (
            ("reuse_traffic_ledger", (1024, 0), {}),
            ("reuse_traffic_ledger", (1024, 2), {"sync_points": -1}),
            ("tile_resource_ledger", (32, 32, 32), {"buffers": 0}),
            ("tile_resource_ledger", (32, 32, 32), {"padding": -1}),
            ("tile_resource_ledger", (32.5, 32, 32), {}),
            ("register_resource_ledger", (True,), {}),
            ("register_resource_ledger", (32,), {"registers_per_sm": 0}),
        ):
            with self.subTest(function=name, args=args, kwargs=kwargs), self.assertRaises(ValueError):
                self.sram[name](*args, **kwargs)

    def test_dependent_copy_and_compute_cannot_overlap(self):
        self.assertEqual(self.stream["stream_timeline"](10, 20, 4)["hidden_ms"], 10)
        dependent = self.stream["stream_timeline"](10, 20, 4, independent=False)
        self.assertEqual(dependent["estimated_elapsed_ms"], 34)
        self.assertEqual(dependent["hidden_ms"], 0)

    def test_each_overlap_condition_is_required(self):
        for condition in ("pinned_memory", "independent", "concurrent_resources"):
            result = self.stream["overlap_candidate"](10, 20, 1, **{condition: False})
            self.assertEqual(result["ideal_hidden_ms"], 0)
            self.assertEqual(result["missing_conditions"], [condition])
        self.assertEqual(self.stream["overlap_candidate"](1, 3, 3)["estimated_net_saved_ms"], -2)

    def test_explicit_dependency_chain_is_serial(self):
        stage = self.schedule["TaskStage"]
        stages = [stage("in", 12, "copy"), stage("kernel", 40, "compute", ("in",)),
                  stage("out", 10, "copy", ("kernel",)), stage("post", 8, "host", ("out",))]
        self.assertEqual(self.schedule["pipelined_time_us"](stages), 70)
        self.assertEqual(self.schedule["schedule_stages"](stages)["out"]["start_us"], 52)

    def test_stream_order_and_independent_queues(self):
        stage = self.schedule["TaskStage"]
        stages = [stage("a", 10, "one"), stage("b", 20, "one"), stage("c", 25, "two")]
        timeline = self.schedule["schedule_stages"](stages)
        self.assertEqual(timeline["b"]["start_us"], 10)
        self.assertEqual(timeline["c"]["start_us"], 0)
        self.assertEqual(self.schedule["pipelined_time_us"](stages), 30)

    def test_forward_reference_on_another_queue_is_resolved(self):
        stage = self.schedule["TaskStage"]
        stages = [stage("consumer", 5, "one", ("producer",)), stage("producer", 7, "two")]
        self.assertEqual(self.schedule["pipelined_time_us"](stages), 12)

    def test_invalid_dependency_graphs_are_rejected(self):
        stage = self.schedule["TaskStage"]
        cases = [
            [stage("a", 1, "one"), stage("a", 2, "two")],
            [stage("a", 1, "one", ("missing",))],
            [stage("a", 1, "one", ("a",))],
            [stage("a", 1, "one", ("b",)), stage("b", 1, "two", ("a",))],
            [stage("a", 1, "one", ("b",)), stage("b", 1, "one")],
            [stage("a", float("nan"), "one")],
            [stage("a", -1, "one")],
        ]
        for stages in cases:
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                self.schedule["schedule_stages"](stages)

    def test_two_batches_overlap_but_overhead_can_erase_savings(self):
        stage = self.schedule["TaskStage"]
        stages = []
        for batch in range(2):
            stages.extend([stage(f"in{batch}", 12, "in"),
                           stage(f"k{batch}", 40, "compute", (f"in{batch}",)),
                           stage(f"out{batch}", 10, "out", (f"k{batch}",))])
        result = self.schedule["pipeline_comparison"](stages, 25)
        self.assertEqual(result["serial_us"], 124)
        self.assertEqual(result["ideal_pipeline_us"], 102)
        self.assertEqual(result["net_saved_us"], -3)

    def test_empty_schedule_and_invalid_timing_values(self):
        self.assertEqual(self.schedule["pipelined_time_us"]([]), 0)
        self.assertEqual(self.schedule["overlap_ratio"](0, 0), 0)
        for values in ((-1, 0), (1, 2), (float("inf"), 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.schedule["overlap_ratio"](*values)
        with self.assertRaises(ValueError):
            self.stream["classify_timeline"](float("nan"), 0, 0, 0)

    def test_transfer_bandwidth_is_decimal_bytes_per_second(self):
        result = self.placement["stage_placement"](100, 1, 64e6, 64e6, 16, 0.1)
        self.assertAlmostEqual(result["transfer_ms"], 8.1)
        ledger = self.placement["offload_ledger"](4, 32, overlap_ratio=0.75)
        self.assertEqual(ledger["transfer_gb_per_step"], 8)
        self.assertEqual(ledger["transfer_ms"], 250)
        self.assertEqual(ledger["estimated_exposed_ms"], 62.5)
        self.assertNotIn("saved_vram_gb", ledger)

    def test_copy_dependency_prefix_reduces_overlap(self):
        for prefix, expected in ((0, 8), (5, 3), (8, 0)):
            self.assertEqual(self.placement["overlap_timeline"](8, 20, prefix)["hidden_ms"], expected)

    def test_time_coverage_is_not_clamped_hardware_utilization(self):
        result = self.placement["parallelism_evidence"](30, 32, 18, 8)
        self.assertEqual(result["elapsed_saved_ms"], -2)
        self.assertNotIn("gpu_utilization", result)
        with self.assertRaises(ValueError):
            self.placement["parallelism_evidence"](30, 20, 18, 8)
        with self.assertRaises(ValueError):
            self.placement["stage_placement"](30, 2, 1, 1, float("nan"))

    def test_warp_packing_is_separate_from_input_tail(self):
        result = self.launch["launch_geometry"](66, 33)
        self.assertEqual(result["inactive_tail_threads"], 0)
        self.assertEqual(result["unassigned_warp_slots"], 62)
        self.assertEqual(self.shared["execution_coverage"](33)["warps"], 2)

    def test_launch_gate_rejects_zero_threads_and_per_thread_overflow(self):
        limits = dict(max_threads_per_block=1024, max_shared_bytes_per_block=65536,
                      max_registers_per_thread=128, register_file_bytes_per_block=262144)
        with self.assertRaises(ValueError):
            self.launch["block_resource_gate"](0, 0, 0, limits)
        result = self.launch["block_resource_gate"](32, 0, 129, limits)
        self.assertEqual(result["reasons"], ["registers_per_thread"])
        with self.assertRaises(ValueError):
            self.launch["block_resource_gate"](32, 0, 1, {})

    def test_correct_shared_memory_can_have_no_read_reduction(self):
        result = self.shared["shared_memory_candidate"](1024, 1, 8, 65536, True)
        self.assertTrue(result["capacity_and_sync_conditions_met"])
        self.assertFalse(result["has_logical_read_reduction"])
        with self.assertRaises(ValueError):
            self.shared["shared_memory_candidate"](1024, 2, 8, 65536, "yes")
        self.assertEqual(self.launch["block_reuse_traffic"](8, 4096, 2, 4)["global_read_bytes_saved"],
                         self.shared["shared_reuse_gain"](8192, 4, 8)["saved_global_read_bytes"])

    def test_minimum_program_grid_covers_input_once(self):
        for size in (1, 32, 64, 100):
            grid = self.mapping["program_coverage"](size, 32)
            offsets = [offset for pid in range(grid["programs"])
                       for offset in self.mapping["program_offsets"](pid, 32, size)["valid_offsets"]]
            self.assertEqual(offsets, list(range(size)))
        with self.assertRaises(ValueError):
            self.mapping["program_offsets"](4, 32, 100)
        self.assertEqual(self.mapping["tile_layout"](100, 130, 32, 32)["padding_elements"], 7480)

    def test_geometry_rejects_noninteger_or_nonpositive_inputs(self):
        for namespace, name, args in (
            (self.launch, "launch_geometry", (10.5, 32)),
            (self.launch, "launch_geometry", (0, 32)),
            (self.shared, "execution_coverage", (True,)),
            (self.mapping, "program_coverage", (100, 0)),
            (self.mapping, "tile_layout", (100, 100, 32.5, 32)),
            (self.mapping, "program_offsets", (-1, 32, 100)),
        ):
            with self.subTest(function=name, args=args), self.assertRaises(ValueError):
                namespace[name](*args)

    def test_layout_cost_is_separate_from_fusion_conditions(self):
        result = self.fusion["fusion_decision"](True, True, True, True, 1024)
        self.assertTrue(result["stated_conditions_met"])
        self.assertEqual(result["extra_reorder_bytes"], 1024)
        candidate = self.graph["fusion_candidate"](1024, 2048, True, 32, 64)
        self.assertTrue(candidate["stated_conditions_met"])
        self.assertEqual(candidate["net_logical_traffic_saved_bytes"], -1024)
        self.assertFalse(candidate["has_net_traffic_saving"])
        with self.assertRaises(ValueError):
            self.fusion["fusion_decision"]("yes", True, True, True)
        with self.assertRaises(ValueError):
            self.fusion["fusion_decision"](True, True, True, True, -1)

    def test_linear_chain_traffic_agrees_between_lessons(self):
        graph = self.graph["graph_cost"](4, 2, 100, 2)
        fusion = self.fusion["fusion_traffic"](4, 200, 2)
        self.assertEqual(graph["intermediate_traffic_bytes"], fusion["intermediate_bytes"])
        self.assertEqual(fusion["intermediate_bytes"], 400)

    def test_no_backend_meeting_constraints_returns_none(self):
        requirements = {'ops': {'matmul'}, 'dtype': 'bf16',
                        'dynamic_shape': True, 'workspace_limit_mib': 512}
        candidate = {'ops': {'matmul'}, 'dtypes': {'bf16'}, 'dynamic_shape': False,
                     'workspace_mib': 64, 'estimated_latency_ms': 1}
        result = self.graph["choose_backend"](requirements, {'static': candidate})
        self.assertIsNone(result["selected"])
        self.assertEqual(result["rejected"]["static"], ['dynamic_shape'])
        with self.assertRaises(ValueError):
            self.graph["choose_backend"](requirements, {'bad': {**candidate, 'estimated_latency_ms': float('nan')}})

    def test_shape_keys_do_not_claim_actual_compilation(self):
        result = self.shapes["shape_reuse_report"]([(8, 128), (8, 512), (8, 128)])
        self.assertEqual(result["distinct_shape_keys"], 2)
        self.assertEqual(result["assumed_exact_shape_hits"], 1)
        self.assertNotIn('compiled_paths', result)
        for shapes in ([], [()], [(8, 0)], [(True, 128)], [(8, 1.5)]):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                self.shapes["shape_reuse_report"](shapes)

    def test_coarse_buckets_can_increase_padding(self):
        baseline = self.shapes["padding_overhead"]([65, 66])
        bucketed = self.shapes["padding_overhead"]([65, 66], [128])
        self.assertEqual(baseline["wasted_tokens"], 1)
        self.assertEqual(bucketed["wasted_tokens"], 125)
        with self.assertRaises(ValueError):
            self.shapes["padding_overhead"]([65, 66], [64])

    def test_symbol_binding_rejects_unresolved_and_invalid_dimensions(self):
        source = {'op': 'matmul', 'dtype': 'fp16', 'shape': ('B', 4096, 4096)}
        for bindings in ({}, {'B': 0}, {'B': -1}, {'B': 8.5}, {'B': True}):
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                self.compiler["specialize_shape"](source, bindings)
        specialized = self.compiler["specialize_shape"](source)
        self.assertEqual(source['shape'][0], 'B')
        self.assertEqual(specialized['shape'][0], 8)

    def test_plan_requires_legalization_and_all_tile_dimensions(self):
        source = {'op': 'matmul', 'dtype': 'fp16', 'shape': ('B', 4096, 4096)}
        shape = self.compiler["specialize_shape"](source)
        with self.assertRaises(ValueError):
            self.compiler["select_schedule"](shape)
        with self.assertRaises(ValueError):
            self.compiler["legalize_ops"]({**shape, 'dtype': 'bf16'})
        legal = self.compiler["legalize_ops"](shape)
        with self.assertRaises(ValueError):
            self.compiler["select_schedule"]({**legal, 'shape': (7, 4096, 4096)})
        scheduled = self.compiler["select_schedule"](legal)
        self.assertEqual(self.compiler["build_backend_plan"](scheduled)['representation'], 'python_plan')
        with self.assertRaises(ValueError):
            self.compiler["build_backend_plan"]({**scheduled, 'shape': (7, 4096, 4096)})

    def test_tuning_handles_empty_feasible_set_and_filters_fastest(self):
        rows = [dict(tile=64, warps=4, shared_memory_bytes=32768, latency_us=80),
                dict(tile=256, warps=8, shared_memory_bytes=131072, latency_us=40)]
        select = self.compiler['select_tuning_candidate']
        self.assertEqual(select(rows, 4096, 99000)['selected']['tile'], 64)
        self.assertIsNone(select(rows, 4096, 1024)['selected'])
        self.assertIsNone(select([], 4096, 99000)['selected'])
        with self.assertRaises(ValueError):
            select([{**rows[0], 'latency_us': float('inf')}], 4096, 99000)

    def test_pipeline_rejects_incomplete_unknown_duplicate_and_bad_order(self):
        check = self.compiler['validate_pipeline']
        for pipeline in ([], ['shape_specialize'], ['unknown'],
                         ['shape_specialize', 'shape_specialize'],
                         ['shape_specialize', 'scheduling']):
            with self.subTest(pipeline=pipeline):
                self.assertFalse(check(pipeline)[0])
        self.assertTrue(check(['shape_specialize'], require_codegen=False)[0])
        self.assertTrue(check(['shape_specialize', 'legalize_ops', 'layout_transform',
                               'scheduling', 'codegen'])[0])


if __name__ == "__main__":
    unittest.main()
