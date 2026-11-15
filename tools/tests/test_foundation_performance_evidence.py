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
        cls.precision = notebook_functions("01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb")
        cls.attention = notebook_functions("01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb")
        cls.matrixpath = notebook_functions("01_Hardware_Math_and_Systems/23_TensorCore_Deep_Dive.ipynb")
        cls.hardware = notebook_functions("01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb")
        cls.scale = notebook_functions("01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb")
        cls.sram = notebook_functions("01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb")
        cls.sparse = notebook_functions("01_Hardware_Math_and_Systems/25_Sparse_Computation_and_Sparse_Attention.ipynb")
        cls.stream = notebook_functions("01_Hardware_Math_and_Systems/17_CUDA_Stream_and_Asynchrony.ipynb")
        cls.schedule = notebook_functions("01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.ipynb")
        cls.placement = notebook_functions("01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb")
        cls.programming = notebook_functions("01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.ipynb")
        cls.launch = notebook_functions("01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.ipynb")
        cls.shared = notebook_functions("01_Hardware_Math_and_Systems/16_Warp_Block_SharedMemory_Basics.ipynb")
        cls.mapping = notebook_functions("01_Hardware_Math_and_Systems/18_Triton_Block_Model.ipynb")
        cls.graph = notebook_functions("01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.ipynb")
        cls.fusion = notebook_functions("01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.ipynb")
        cls.shapes = notebook_functions("01_Hardware_Math_and_Systems/30_Dynamic_Shape_Handling.ipynb")
        cls.compiler = notebook_functions("01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.ipynb")

    def test_latent_content_scores_keep_projection_order_equivalent(self):
        """Compare both notebook paths against manually expanded keys and scores."""
        functions = notebook_functions("01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb")
        score = functions['latent_content_scores']
        projection = [[1, 0], [0, 1], [1, 1], [2, -1]]
        row = score([1, -1, 2, 0], [2, 3], projection)
        self.assertEqual(row['restored_key'], [2, 3, 5, 1])
        self.assertEqual(row['transformed_query'], [3, 1])
        self.assertEqual(row['restore_then_score'], 9)
        self.assertEqual(row['transform_then_score'], 9)
        changed = score([2, 0, -1, 1], [-1, 4], projection)
        self.assertEqual(changed['restore_then_score'], -11)
        self.assertEqual(changed['transform_then_score'], -11)
        zero = score([0, 0, 0, 0], [2, 3], projection)
        self.assertEqual(zero['transform_then_score'], 0)
        for query, latent, matrix in (([], [1], []), ([1], [], [[]]),
                                       ([1, 2], [1], [[1]]), ([1], [1, 2], [[1]])):
            with self.subTest(query=query, latent=latent), self.assertRaises(ValueError):
                score(query, latent, matrix)

    def test_compact_cache_counts_position_key_and_page_capacity(self):
        """Check stored elements, position-key width, tail slots and independent reuse count."""
        functions = notebook_functions("01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb")
        cache = functions['mla_cache_bytes']
        self.assertEqual(cache(5, 2, 4), 80)
        self.assertEqual(cache(5, 2, 4, rope_dim=2), 120)
        self.assertEqual(cache(5, 2, 4, rope_dim=2, dtype_bytes=4), 240)
        self.assertEqual(cache(5, 2, 4, rope_dim=2, batch_size=2), 240)
        compare = functions['compare_cache_strategies']
        case = dict(page_size=4, base_kv_heads=2, head_dim=4, latent_dim=4,
                    rope_dim=2, num_layers=2)
        row = compare(seq_len=5, **case)
        self.assertEqual(row['management'], {'pages': 2, 'waste_tokens': 3})
        self.assertEqual(row['representation'], {
            'base_kv_bytes': 320, 'compact_kv_bytes': 120,
            'paged_compact_kv_bytes': 192, 'representation_saving_ratio': 0.625})
        shared = compare(seq_len=5, prefix_len=5, **case)
        self.assertEqual(shared['representation'], row['representation'])
        full = compare(seq_len=8, **case)
        self.assertEqual(full['management']['waste_tokens'], 0)
        self.assertEqual(full['representation']['compact_kv_bytes'], 192)
        self.assertEqual(full['representation']['paged_compact_kv_bytes'], 192)
        for width in (-1, 1.5, True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                cache(5, 2, 4, rope_dim=width)
        with self.assertRaises(ValueError):
            compare(seq_len=5, prefix_len=6, **case)
        with self.assertRaises(ValueError):
            compare(seq_len=5, **{**case, 'page_size': 0})

    def test_matrix_work_is_independent_of_instruction_packing(self):
        for function in (self.precision["gemm_flops"], self.matrixpath["mma_flops"]):
            self.assertEqual(function(16, 16, 16), 8192)
            self.assertEqual(function(16, 16, 32), 16384)
            self.assertEqual(function(2, 3, 4), 48)
            with self.assertRaises(ValueError):
                function(16, 0, 16)

    def test_precision_ledger_uses_each_objects_storage_width(self):
        ledger = self.precision["precision_impact_ledger"]
        elements = dict(params=1024**2, activation_elements=2 * 1024**2, kv_elements=1024**2)
        for widths, expected_mib in (((4, 4, 4), 16), ((4, 2, 2), 10), ((2, 2, 2), 8)):
            row = ledger(**elements, weights_bytes=widths[0], activation_bytes=widths[1], kv_bytes=widths[2])
            self.assertEqual(row["total_bytes"], expected_mib * 1024**2)
        self.assertEqual(self.precision["tensor_storage_bytes"](0, 4), 0)
        self.assertEqual(self.precision["tensor_storage_bytes"](4096**2, 2), 32 * 1024**2)

    def test_precision_triplet_keeps_tf32_storage_and_accumulator_separate(self):
        inspect = self.matrixpath["tensorcore_dtype_path"]
        paths = {("FP32", "TF32", "FP32"), ("BF16", "BF16", "FP32")}
        tf32 = inspect("FP32", "TF32", "FP32", 1024, paths)
        self.assertTrue(tf32["declared_path_supported"])
        self.assertEqual(tf32["input_storage_bytes"], 4096)
        self.assertEqual(inspect("BF16", "BF16", "FP32", 1024, paths)["input_storage_bytes"], 2048)
        self.assertFalse(inspect("BF16", "BF16", "BF16", 1024, paths)["declared_path_supported"])

    def test_matrix_configuration_does_not_turn_occupancy_into_instruction_gate(self):
        inspect = self.matrixpath["inspect_matrix_configuration"]
        tail = inspect(130, 128, 128, occupancy=0.3, layout_contiguous=False)
        self.assertEqual(tail["padded_shape_mnk"], (144, 128, 128))
        self.assertEqual(tail["extra_output_slots"], 1792)
        self.assertTrue(tail["layout_adaptation_needed"])
        self.assertEqual(tail["given_occupancy"], 0.3)
        self.assertNotIn("path_candidate", tail)
        self.assertTrue(inspect(128, 128, 128, register_pressure=64)["register_budget_met"])
        self.assertFalse(inspect(128, 128, 128, register_pressure=65)["register_budget_met"])

    def test_tensor_instruction_observation_is_separate_from_speed(self):
        summarize = self.matrixpath["summarize_tensorcore_observation"]
        slower = summarize(2, 3, 64)
        self.assertTrue(slower["tensor_instructions_observed"])
        self.assertEqual(slower["saved_ms"], -1)
        self.assertEqual(summarize(2, 1.6, 64)["time_ratio"], 1.25)
        self.assertIsNone(summarize(2, 1.6, None)["tensor_instructions_observed"])
        self.assertFalse(summarize(2, 1.6, 0)["tensor_instructions_observed"])
        self.assertEqual(summarize(2, 2, 64)["saved_ms"], 0)
        for baseline, candidate, count in ((0, 1, 0), (1, math.nan, 0), (1, 2, -1)):
            with self.subTest(values=(baseline, candidate, count)), self.assertRaises(ValueError):
                summarize(baseline, candidate, count)

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

    def test_hardware_bandwidth_and_matmul_units(self):
        """Keep hypothetical bandwidth scenarios separate from device capabilities."""
        times = self.hardware["estimate_transfer_times"]({"a": 1000, "b": 2000})
        self.assertAlmostEqual(times["a"]["estimated_transfer_ms"], 0.001048576)
        self.assertEqual(times["b"]["estimated_transfer_ms"], times["a"]["estimated_transfer_ms"] / 2)
        bf16 = self.hardware["estimate_matmul_workload"](m=2, n=3, k=4, dtype="bf16")
        fp32 = self.hardware["estimate_matmul_workload"](m=2, n=3, k=4, dtype="fp32")
        self.assertEqual(bf16["flops"], 48)
        self.assertEqual(bf16["minimum_io_bytes"], 52)
        self.assertEqual(fp32["minimum_io_bytes"], 104)
        self.assertNotIn("tensor_core_candidate", fp32)

    def test_attention_storage_is_not_memory_traffic(self):
        """Count both score-tile axes without interpreting storage as arithmetic intensity."""
        storage = self.hardware["calculate_attention_vram"](seq_len=4, num_heads=2, head_dim=3)
        self.assertEqual(storage["total"], 256 / 1e9)
        self.assertEqual(storage["attention_matrix"], 64 / 1e9)
        self.assertEqual(storage["qk_pv_flops"], 384)
        self.assertNotIn("score_intensity", storage)
        tiles = self.hardware["compare_attention_working_set"](seq_len=100, block_size=32)
        self.assertEqual(tiles["tile_grid"], (4, 4))
        self.assertEqual(tiles["score_tile_count"], 16)
        self.assertEqual(tiles["score_tile_kib"], 2)

    def test_link_units_and_roofline_intersection(self):
        """GB/s means bytes per second; roofline equality is a shared bound."""
        link = self.hardware["pcie_vs_nvlink"](payload_mb=64, pcie_gb_s=16, nvlink_gb_s=64,
                                               communication_rounds=4, sync_overhead_ms=0.05)
        self.assertEqual(link["pcie_once_ms"], 4)
        self.assertAlmostEqual(link["pcie_total_ms"], 16.2)
        self.assertAlmostEqual(link["nvlink_total_ms"], 4.2)
        roof = self.hardware["roofline_estimate"](flops=2e10, bytes_moved=1e8,
                                                  peak_compute_tflops=300, memory_bandwidth_gb_s=1500)
        self.assertEqual(roof["limiting_roof"], "balance")
        self.assertEqual(roof["roofline_upper_tflops"], 300)
        self.assertNotIn("bottleneck", roof)

    def test_resource_metrics_exact_units_and_bad_records(self):
        """Use disjoint windows and finite real fields for record-derived metrics."""
        record = self.resource_record()
        metrics = self.resource["derive_resource_metrics"](record)
        self.assertEqual(metrics["achieved_tflops"], 2)
        self.assertEqual(metrics["achieved_dram_gb_s"], 400)
        self.assertEqual(metrics["total_ms"], 26)
        for change in ({"flops": True}, {"dram_bytes": float("inf")},
                       {"kernel_ms": 0}, {"sync_ms": -1}, {"launch_ms": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.resource["derive_resource_metrics"](record | change)

    def test_kernel_speedup_can_be_offset_by_conversion_transfer(self):
        """Separate kernel progress from a stage whose added transfer cancels it."""
        baseline = self.resource_record()
        candidate = baseline | {"kernel_ms": 10.0, "transfer_ms": 13.0}
        derive = self.resource["derive_resource_metrics"]
        self.assertEqual(baseline["kernel_ms"] / candidate["kernel_ms"], 2.0)
        self.assertEqual(derive(candidate)["achieved_tflops"], 4.0)
        self.assertEqual(derive(baseline)["total_ms"], 26.0)
        self.assertEqual(derive(candidate)["total_ms"], 26.0)
        plan = self.resource["build_discrimination_experiment"](
            attribution={"primary_hypothesis": "kernel_compute"},
            single_change="compute_kernel",
        )
        self.assertEqual(plan["target_metrics"], [
            "achieved_tflops", "kernel_ms", "stage_ms", "max_abs_error",
        ])
        self.assertEqual(plan["revisit_when"], "kernel_faster_but_stage_ms_unchanged")

    def test_reference_efficiency_and_tolerance_boundaries(self):
        """Example screens are explicit inputs rather than universal hardware thresholds."""
        record = self.resource_record() | {"flops": 8e10, "kernel_ms": 10}
        inputs = dict(record=record, reference_tflops=10, reference_dram_gb_s=1000)
        at_screen = self.resource["attribute_bottleneck"](**inputs, efficiency_screen=0.8)
        self.assertEqual(at_screen["reference_efficiency"], 0.8)
        self.assertEqual(at_screen["primary_hypothesis"], "kernel_balance")
        self.assertEqual(self.resource["attribute_bottleneck"](**inputs, efficiency_screen=0.81)
                         ["primary_hypothesis"], "kernel_underutilized")
        over = self.resource_record() | {"flops": 105e9, "dram_bytes": 1e9, "kernel_ms": 1}
        inputs = dict(record=over, reference_tflops=100, reference_dram_gb_s=1000)
        self.assertEqual(self.resource["attribute_bottleneck"](**inputs, reference_tolerance=0.05)
                         ["reference_consistency"], "within_tolerance")
        self.assertEqual(self.resource["attribute_bottleneck"](**inputs, reference_tolerance=0)
                         ["primary_hypothesis"], "recheck_reference")

    def test_equal_roofs_allow_one_concrete_follow_up(self):
        """Retain both resource explanations at the intersection without changing two variables."""
        plan = self.resource["build_discrimination_experiment"](
            attribution={"primary_hypothesis": "kernel_balance"}, single_change="tile_size",
        )
        self.assertEqual(plan["single_change"], "tile_size")
        self.assertIn("stage_ms", plan["target_metrics"])
        self.assertIn("max_abs_error", plan["target_metrics"])
        with self.assertRaises(ValueError):
            self.resource["build_discrimination_experiment"](
                attribution={"primary_hypothesis": "recheck_reference"}, single_change="tile_size",
            )

    def test_lesson_error_examples_remain_covered_outside_prose(self):
        """Exercise removed error demonstrations against actual notebook functions."""
        cases = []

        def add(namespace, name, base, field, values):
            cases.extend((namespace[name], base | {field: value}) for value in values)

        add(self.placement, "overlap_timeline", dict(copy_ms=8, compute_ms=20),
            "dependency_ms", (-1, 9, True, float("nan")))
        add(self.programming, "memory_traffic", dict(tensor_bytes=8), "num_ops", (0, 1.5, True))
        add(self.programming, "block_launch_stats", {}, "num_threads", (0, -1, 1.5, True))
        add(self.graph, "graph_cost", dict(num_ops=4, fused_groups=1, intermediate_elements=10,
                                          bytes_per_element=2),
            "launch_overhead_us", (True, -1, float("nan"), float("inf")))
        for value in (True, -1, float("nan"), float("inf")):
            cases.append((self.graph["choose_backend"], {
                "requirements": dict(ops={"matmul"}, dtype="bf16", dynamic_shape=True,
                                     workspace_limit_mib=value), "candidates": {},
            }))
        add(self.launch, "launch_geometry", dict(threads_per_block=32), "num_elements", (0, 1.5, True))
        add(self.shared, "execution_coverage", {}, "num_threads", (0, 1.5, True))
        add(self.shared, "shared_memory_candidate", dict(tile_bytes=32768, reuse_count=4, blocks=1,
                                                        synchronization_valid=True),
            "shared_limit_bytes", (0, True))
        cases.append((self.stream["classify_timeline"], dict(kernel_ms=0, exposed_copy_ms=0,
                                                           exposed_sync_ms=0, idle_ms=0)))
        add(self.mapping, "program_coverage", dict(num_elements=100), "block_size", (0, 1.5, True))
        add(self.mapping, "program_offsets", dict(block_size=32, num_elements=100),
            "program_id", (-1, 4, True))
        add(self.fusion, "fusion_traffic", dict(num_ops=3, tensor_bytes=4096), "fused_groups", (0, 4, True))
        add(self.fusion, "intermediate_traffic", dict(consumers={"activation"},
                                                       fused_consumers={"activation"}),
            "tensor_bytes", (0, -1, True))
        add(self.fusion, "fusion_decision", dict(sample_matches=True, shape_matches=True,
                                               u_slice_fits=True, backend_declared=True),
            "reorder_bytes", (-1, True))
        add(self.schedule, "graph_suitability", dict(is_repeated_path=True, has_many_branches=False),
            "is_fixed_shape", (1,))
        add(self.shapes, "shape_reuse_report", {}, "shapes", ([], [()], [(8, 0)], [(True, 128)]))
        add(self.shapes, "padding_overhead", dict(lengths=[65, 66]), "bucket_boundaries", ([], [64], [True, 128]))
        add(self.shapes, "path_stability", dict(lengths=[128]), "boundaries", ((256, 256), (1024, 256), (True, 1024)))
        for value in (True, 0, float("nan"), float("inf")):
            cases.append((self.compiler["select_tuning_candidate"], dict(
                candidates=[dict(tiles_mnk=(8, 128, 64), warps=4,
                                 required_capabilities=set(), latency_us=value)],
                problem_shape=(128, 256, 64), shared_memory_limit=65536,
            )))
        add(self.hardware, "estimate_matmul_workload", dict(n=3, k=4), "m", (0, 1.5, True))
        for function, inputs in cases:
            with self.subTest(function=function.__name__, inputs=inputs), self.assertRaises(ValueError):
                function(**inputs)

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

    def test_score_storage_scales_quadratically_and_counts_all_heads(self):
        storage = self.attention["attention_score_bytes"]
        self.assertEqual(storage(2048), 16 * 1024**2)
        self.assertEqual(storage(4096), 64 * 1024**2)
        self.assertEqual(storage(4096, batch_size=2, num_heads=4), 512 * 1024**2)
        for value in (0, -1, True, 2.5):
            with self.subTest(length=value), self.assertRaises(ValueError):
                storage(value)

    def test_online_attention_matches_full_normalization_across_blocks(self):
        online = self.attention["online_attention_scalar"]
        for scores, values in (([0, 0, math.log(2)], [1, 3, 5]),
                               ([1000, 999, 1001, 998, 1002], [-2, 1, 4, 8, 3]),
                               ([-1000], [7])):
            weights = [math.exp(score - max(scores)) for score in scores]
            reference = sum(w * value for w, value in zip(weights, values)) / sum(weights)
            for block_size in (1, 2, len(scores), len(scores) + 1):
                with self.subTest(scores=scores, block_size=block_size):
                    self.assertAlmostEqual(online(scores, values, block_size), reference, places=12)

    def test_online_attention_rejects_missing_or_nonfinite_values(self):
        online = self.attention["online_attention_scalar"]
        for args in (([], [], 1), ([0], [], 1), ([0], [1], 0),
                     ([math.inf], [1], 1), ([0], [math.nan], 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                online(*args)

    def test_attention_tile_payload_includes_accumulators_and_statistics(self):
        working_set = self.attention["flashattention_working_set_bytes"]
        self.assertEqual(working_set(64, 128), 98816)
        self.assertEqual(working_set(128, 128), 230400)
        self.assertEqual(working_set(256, 128), 591872)
        self.assertEqual(working_set(128, 128, dtype_bytes=4) - working_set(128, 128),
                         3 * 128 * 128 * 2)
        self.assertEqual(self.attention["num_1d_tiles"](130, 64), 3)
        self.assertEqual(self.attention["num_score_tiles"](130, 64), 9)
        self.assertEqual(self.attention["score_materialization_ratio"](4096, 128), 1024)
        with self.assertRaises(ValueError):
            working_set(0, 128)

    def test_bank_conflict_counts_distinct_words_not_broadcast_lanes(self):
        inspect = self.sram["bank_access_report"]
        self.assertEqual(inspect(32)["max_conflict_degree"], 32)
        self.assertEqual(inspect(32, padding_words=1)["max_conflict_degree"], 1)
        self.assertEqual(inspect(2, threads=16)["max_conflict_degree"], 1)
        broadcast = inspect(32, access_mode="broadcast")
        self.assertEqual(broadcast["max_lanes_per_bank"], 32)
        self.assertEqual(broadcast["max_conflict_degree"], 1)
        self.assertFalse(broadcast["serialized_risk"])
        for kwargs in ({"threads": 33}, {"stride": True}, {"padding_words": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inspect(**({"stride": 32} | kwargs))

    def test_shared_memory_capacity_equality_is_feasible(self):
        ledger = self.sram["tile_resource_ledger"]
        self.assertEqual(ledger(32, 32, 32, smem_per_sm_bytes=8192)
                         ["resident_blocks_upper_bound"], 1)
        self.assertEqual(ledger(32, 32, 32, smem_per_sm_bytes=8191)
                         ["resident_blocks_upper_bound"], 0)

    def test_sparse_density_does_not_describe_position_pattern(self):
        report = self.sparse["retained_position_report"]
        for keep in ([True, False, True, False], [True, True, False, False]):
            self.assertEqual(report(keep)["density"], 0.5)
            self.assertEqual(report(keep)["retained_positions"], 2)
        self.assertEqual(report([True] * 4)["removed_ratio"], 0)
        self.assertEqual(report([False] * 4)["density"], 0)

    def test_sparse_local_groups_can_fail_at_the_same_global_density(self):
        inspect = self.sparse["group_sparsity_report"]
        self.assertTrue(inspect([1, 0, 2, 0, 3, 0, 4, 0])["pattern_matches"])
        self.assertEqual(inspect([1, 2, 3, 0, 0, 0, 0, 4])["violating_groups"], [0])
        self.assertTrue(inspect([0, 0, 0, 0])["pattern_matches"])
        self.assertNotIn("hardware_path_available", inspect([1, 0, 2, 0]))

    def test_sparse_coo_index_storage_can_exceed_dense_payload(self):
        ledger = self.sparse["sparse_matmul_ledger"]
        row = ledger(m=2, n=4, k=4, retained_weights=4)
        self.assertEqual(row["dense_weight_bytes"], 32)
        self.assertEqual(row["coo_value_bytes"], 8)
        self.assertEqual(row["coo_index_bytes"], 32)
        self.assertEqual(row["coo_total_bytes"], 40)
        self.assertEqual(row["dense_flops"], 64)
        self.assertEqual(row["ideal_sparse_flops"], 16)
        self.assertEqual(ledger(2, 4, 4, 0)["coo_total_bytes"], 0)
        dense_pattern = ledger(2, 4, 4, 16)
        self.assertEqual(dense_pattern["ideal_sparse_flops"], dense_pattern["dense_flops"])

    def test_sparse_setup_reuse_is_separate_from_kernel_time(self):
        cost = self.sparse["sparse_call_cost"]
        self.assertAlmostEqual(cost(1, 0.6, 2, 1)["average_sparse_ms"], 2.6)
        self.assertAlmostEqual(cost(1, 0.6, 2, 10)["average_sparse_ms"], 0.8)
        self.assertEqual(cost(1, 0.5, 2, 4)["saved_ms_per_call"], 0)
        self.assertAlmostEqual(cost(1, 0.6, 0, 1)["saved_ms_per_call"], 0.4)

    def test_sparse_calculators_reject_invalid_ranges(self):
        cases = (
            ("retained_position_report", ([0, 1],)),
            ("retained_position_report", ([],)),
            ("group_sparsity_report", ([1, 0, 0],)),
            ("group_sparsity_report", ([1, 0, 0, 0], 4, 5)),
            ("sparse_matmul_ledger", (2, 4, 4, 17)),
            ("sparse_matmul_ledger", (2, 4, 4, True)),
            ("sparse_call_cost", (1, 0.6, math.inf, 1)),
            ("sparse_call_cost", (1, 0.6, 2, 0)),
        )
        for name, args in cases:
            with self.subTest(name=name, args=args), self.assertRaises(ValueError):
                self.sparse[name](*args)

    def test_local_attention_mask_counts_self_and_boundary_connections(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch CPU is required for the attention reference")
        self.sparse["torch"] = torch
        build = self.sparse["build_local_attention_mask"]
        self.assertEqual(build(8, 1).sum().item(), 22)
        self.assertEqual(build(8, 0).sum().item(), 8)
        self.assertEqual(build(8, 8).sum().item(), 64)
        self.assertTrue(build(8, 1)[0, 0, 0, 1].item())
        self.assertEqual(build(1, 0).sum().item(), 1)
        with self.assertRaises(ValueError):
            build(8, -1)

    def test_masked_attention_matches_sdpa_and_rejects_fully_removed_rows(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch CPU is required for the attention reference")
        self.sparse["torch"] = torch
        generator = torch.Generator(device="cpu").manual_seed(25)
        q, k, v = [torch.randn(1, 2, 5, 4, generator=generator) for _ in range(3)]
        apply = self.sparse["masked_attention_cpu"]
        for window in (0, 1, 4):
            mask = self.sparse["build_local_attention_mask"](5, window)
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
            torch.testing.assert_close(apply(q, k, v, mask), expected, atol=1e-6, rtol=1e-5)
        mask = self.sparse["build_local_attention_mask"](5, 1)
        mask[..., 0, :] = False
        with self.assertRaises(ValueError):
            apply(q, k, v, mask)

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
        result = self.placement["stage_placement"](100, 1, 64_000_000, 64_000_000, 16, 0.1)
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

    def test_stage_cost_rejects_boolean_times_and_fractional_bytes(self):
        """Use integer byte counts and finite real durations for placement estimates."""
        base = dict(cpu_ms=8.5, gpu_ms=0.4, input_bytes=64_000_000,
                    output_bytes=64_000_000, bandwidth_gb_per_s=16, fixed_transfer_ms=0.1)
        self.assertEqual(self.placement["stage_placement"](**base)["shorter_estimated_path"], "equal")
        for change in ({"cpu_ms": True}, {"input_bytes": 1.5}, {"output_bytes": True},
                       {"bandwidth_gb_per_s": 0}, {"fixed_transfer_ms": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.placement["stage_placement"](**(base | change))

    def test_offload_zero_full_overlap_and_round_trip_units(self):
        """Check endpoints without treating offloaded state as measured peak savings."""
        base = dict(state_gb=4, bandwidth_gb_per_s=32)
        estimate = self.placement["offload_ledger"]
        self.assertEqual(estimate(**base, overlap_ratio=0)["estimated_exposed_ms"], 250)
        self.assertEqual(estimate(**base, overlap_ratio=1)["estimated_exposed_ms"], 0)
        self.assertEqual(estimate(**base, round_trips_per_step=0)["transfer_ms"], 0)
        self.assertEqual(estimate(**base, round_trips_per_step=2)["transfer_gb_per_step"], 16)
        for change in ({"state_gb": True}, {"overlap_ratio": True}, {"round_trips_per_step": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                estimate(**(base | change))

    def test_stream_empty_window_and_exposed_overhead_boundaries(self):
        """Empty windows have no time ratio; exposed overhead can erase all savings."""
        self.assertIsNone(self.stream["stream_timeline"](0, 0, 0)["speedup"])
        self.assertEqual(self.stream["overlap_candidate"](10, 20, 10)["estimated_net_saved_ms"], 0)
        for name, args in (("stream_timeline", (True, 20, 4)),
                           ("overlap_candidate", (10, 20, True)),
                           ("classify_timeline", (1, 1, True, 0))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.stream[name](*args)
        shares = self.stream["classify_timeline"](1, 1, 1, 0)["fractions"]
        self.assertEqual(shares["kernel"], 1 / 3)
        self.assertAlmostEqual(sum(shares.values()), 1)

    def test_graph_candidate_conditions_and_duration_types_are_explicit(self):
        """Given graph conditions and stage durations must not accept truthy substitutes."""
        graph = self.schedule["graph_suitability"]
        self.assertTrue(graph(is_fixed_shape=True, is_repeated_path=True, has_many_branches=False))
        for change in ({"is_fixed_shape": False}, {"is_repeated_path": False},
                       {"has_many_branches": True}):
            base = dict(is_fixed_shape=True, is_repeated_path=True, has_many_branches=False)
            self.assertFalse(graph(**(base | change)))
        with self.assertRaises(ValueError):
            graph(is_fixed_shape=1, is_repeated_path=True, has_many_branches=False)
        with self.assertRaises(ValueError):
            self.schedule["schedule_stages"]([
                self.schedule["TaskStage"](name="invalid", duration_us=True, stream="one"),
            ])

    def test_pipeline_zero_net_saving_and_copy_queue_order(self):
        """A shared ordered copy queue can serialize this particular submission list."""
        stage = self.schedule["TaskStage"]
        stages = []
        for batch in range(2):
            stages.extend([
                stage(name=f"in{batch}", duration_us=12, stream="in"),
                stage(name=f"k{batch}", duration_us=40, stream="compute", depends_on=(f"in{batch}",)),
                stage(name=f"out{batch}", duration_us=10, stream="out", depends_on=(f"k{batch}",)),
            ])
        report = self.schedule["pipeline_comparison"](stages=stages, exposed_overhead_us=22)
        self.assertEqual(report["estimated_elapsed_us"], 124)
        self.assertEqual(report["net_saved_us"], 0)
        merged = [stage(name=item.name, duration_us=item.duration_us,
                        stream="copy" if item.stream in ("in", "out") else item.stream,
                        depends_on=item.depends_on) for item in stages]
        self.assertEqual(self.schedule["pipelined_time_us"](stages=merged), 124)
        with self.assertRaises(ValueError):
            self.schedule["pipeline_comparison"](stages=stages, exposed_overhead_us=True)

    def test_pipeline_contention_is_compared_with_original_serial_workload(self):
        """Slower candidate phases must not redefine the serial reference duration."""
        stage = self.schedule["TaskStage"]
        baseline = []
        for batch in range(2):
            baseline.extend([
                stage(name=f"in{batch}", duration_us=12, stream="in"),
                stage(name=f"k{batch}", duration_us=40, stream="compute", depends_on=(f"in{batch}",)),
                stage(name=f"out{batch}", duration_us=10, stream="out", depends_on=(f"k{batch}",)),
            ])
        candidate = [
            stage(name=item.name, duration_us=52 if item.stream == "compute" else item.duration_us,
                  stream=item.stream, depends_on=item.depends_on)
            for item in baseline
        ]
        reference_us = self.schedule["sequential_time_us"](baseline)
        candidate_us = self.schedule["pipelined_time_us"](candidate)
        self.assertEqual(reference_us, 124)
        self.assertEqual(candidate_us, 126)
        self.assertEqual(reference_us - candidate_us, -2)
        timeline = self.schedule["schedule_stages"](candidate)
        self.assertEqual(timeline['k1']['start_us'], 64)
        self.assertEqual(timeline['out1']['start_us'], 116)

    def test_warp_packing_is_separate_from_input_tail(self):
        result = self.launch["launch_geometry"](66, 33)
        self.assertEqual(result["inactive_tail_threads"], 0)
        self.assertEqual(result["unassigned_warp_slots"], 62)
        self.assertEqual(self.shared["execution_coverage"](33)["warps"], 2)

    def test_programming_model_reports_slots_not_sm_occupancy(self):
        """Keep logical packing and full-tensor traffic independently checkable."""
        packed = self.programming["block_launch_stats"](num_threads=48)
        self.assertEqual(packed["thread_slot_utilization"], 0.75)
        self.assertEqual(packed["unassigned_slots"], 16)
        self.assertNotIn("occupancy", packed)
        self.assertEqual(self.programming["memory_traffic"](
            num_ops=3, tensor_bytes=8_000_000, fused=True,
        )["traffic_bytes"], 16_000_000)
        for invalid in (True, 1.5, 0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.programming["memory_traffic"](num_ops=invalid, tensor_bytes=8)

    def test_shared_time_model_preserves_zero_and_negative_savings(self):
        """Compare the assumed read savings against varying synchronization costs."""
        inputs = dict(reuse_times=2, tile_bytes=64_000,
                      bandwidth_bytes_per_us=16_000, sync_cost_us=2)
        for sync_points, expected_us in ((1, 2), (2, 0), (3, -2)):
            report = self.programming["shared_tradeoff"](**inputs, sync_points=sync_points)
            self.assertEqual(report["estimated_net_saved_us"], expected_us)
            self.assertNotIn("saved_hbm_bytes", report)

    def test_triton_example_checks_each_condition_and_padding_equality(self):
        """Treat layout and dtype as supplied conditions and verify geometric padding."""
        base = dict(num_elements=96, block_size=128, layout_contiguous=True,
                    dtype_supported=True, max_padding_ratio=0.25)
        self.assertTrue(self.programming["triton_candidate"](**base)["stated_conditions_met"])
        for change, reason in (
            ({"layout_contiguous": False}, "non_contiguous_layout"),
            ({"dtype_supported": False}, "unsupported_dtype"),
            ({"num_elements": 95}, "padding_ratio"),
        ):
            with self.subTest(change=change):
                self.assertEqual(self.programming["triton_candidate"](**(base | change))["reasons"],
                                 [reason])

    def test_launch_gate_rejects_zero_threads_and_per_thread_overflow(self):
        limits = dict(max_threads_per_block=1024, max_shared_bytes_per_block=65536,
                      max_registers_per_thread=128, max_register_bytes_per_block=262144)
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

    def test_shared_reuse_matches_single_block_example(self):
        """Check the diagram's exact bytes before scaling to independent blocks."""
        report = self.shared["shared_reuse_gain"](tile_bytes=32 * 1024, reuse_count=4, blocks=1)
        self.assertEqual(report, {
            "baseline_global_read_bytes": 131072,
            "shared_global_read_bytes": 32768,
            "saved_global_read_bytes": 98304,
            "global_read_reduction_fraction": 0.75,
        })
        scaled = self.shared["shared_reuse_gain"](tile_bytes=32 * 1024, reuse_count=4, blocks=8)
        self.assertEqual(scaled["shared_global_read_bytes"], 262144)

    def test_shared_capacity_equality_and_sync_are_separate_conditions(self):
        """Keep hypothetical savings separate from unmet implementation conditions."""
        base = dict(tile_bytes=32768, reuse_count=4, blocks=1,
                    shared_limit_bytes=32768, synchronization_valid=True)
        self.assertTrue(self.shared["shared_memory_candidate"](**base)["capacity_and_sync_conditions_met"])
        for change, reasons in (
            ({"shared_limit_bytes": 32767}, ["shared_capacity"]),
            ({"synchronization_valid": False}, ["synchronization"]),
            ({"shared_limit_bytes": 32767, "synchronization_valid": False},
             ["shared_capacity", "synchronization"]),
        ):
            with self.subTest(change=change):
                report = self.shared["shared_memory_candidate"](**(base | change))
                self.assertEqual(report["reasons"], reasons)
                self.assertFalse(report["capacity_and_sync_conditions_met"])
                self.assertEqual(report["saved_global_read_bytes"], 98304)

    def test_last_warp_assignment_is_not_instruction_activity(self):
        """Packing counts assigned threads without claiming per-instruction activity."""
        packed = self.shared["execution_coverage"](num_threads=48)
        self.assertEqual(packed["last_warp_assigned_threads"], 16)
        self.assertEqual(packed["thread_slot_utilization"], 0.75)
        self.assertEqual(2 * packed["warps"], 4)
        self.assertEqual(self.shared["execution_coverage"](32)["last_warp_assigned_threads"], 32)

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
        before = self.graph["graph_cost"](2, 2, 128, 4)
        after = self.graph["graph_cost"](2, 1, 128, 4, layout_bytes=2048)
        candidate = self.graph["graph_rewrite_delta"](before, after)
        self.assertEqual(candidate["saved_launches"], 1)
        self.assertEqual(candidate["net_logical_traffic_saved_bytes"], -1024)
        self.assertFalse(candidate["has_net_traffic_saving"])
        with self.assertRaises(ValueError):
            self.fusion["fusion_decision"]("yes", True, True, True)
        with self.assertRaises(ValueError):
            self.fusion["fusion_decision"](True, True, True, True, -1)

    def test_fusion_removes_write_only_when_all_consumers_are_inside(self):
        traffic = self.fusion["intermediate_traffic"]
        only = traffic(4096, {"activation"}, {"activation"})
        external = traffic(4096, {"activation", "aux"}, {"activation"})
        all_inside = traffic(4096, {"activation", "aux"}, {"activation", "aux"})
        self.assertEqual(only["saved_write_bytes"], 4096)
        self.assertEqual(only["saved_read_bytes"], 4096)
        self.assertEqual(only["total_saved_bytes"], 8192)
        self.assertEqual(external["outside_consumers"], ["aux"])
        self.assertEqual(external["write_bytes_after"], 4096)
        self.assertEqual(external["read_bytes_after"], 4096)
        self.assertEqual(external["saved_write_bytes"], 0)
        self.assertEqual(external["total_saved_bytes"], 4096)
        self.assertEqual(all_inside["total_saved_bytes"], 12288)

    def test_fusion_without_inside_consumers_preserves_intermediate_traffic(self):
        traffic = self.fusion["intermediate_traffic"]
        row = traffic(64, {"activation", "aux"}, set())
        self.assertEqual(row["write_bytes_before"], row["write_bytes_after"])
        self.assertEqual(row["read_bytes_before"], row["read_bytes_after"])
        self.assertEqual(row["total_saved_bytes"], 0)
        self.assertEqual(row["outside_consumers"], ["activation", "aux"])
        for consumers, inside in ((set(), set()), ({"activation"}, {"unknown"}),
                                  (["activation"], {"activation"})):
            with self.subTest(consumers=consumers, inside=inside), self.assertRaises(ValueError):
                traffic(64, consumers, inside)

    def test_fusion_single_consumer_agrees_with_linear_chain(self):
        traffic = self.fusion["intermediate_traffic"]
        for size in (4, 4096, 1048576):
            before = self.fusion["fusion_traffic"](2, size, 2)["intermediate_bytes"]
            after = self.fusion["fusion_traffic"](2, size, 1)["intermediate_bytes"]
            self.assertEqual(traffic(size, {"activation"}, {"activation"})["total_saved_bytes"],
                             before - after)

    def test_fusion_checks_identify_each_given_condition_without_speed_claim(self):
        summarize = self.fusion["fusion_decision"]
        base = dict(sample_matches=True, shape_matches=True, u_slice_fits=True,
                    backend_declared=True, reorder_bytes=0)
        for name in ("sample_matches", "shape_matches", "u_slice_fits", "backend_declared"):
            row = summarize(**(base | {name: False}))
            self.assertFalse(row["stated_conditions_met"])
            self.assertEqual(row["failures"], [name])
        row = summarize(**(base | {"reorder_bytes": 24}))
        self.assertTrue(row["stated_conditions_met"])
        self.assertEqual(row["extra_reorder_bytes"], 24)
        self.assertNotIn("speedup", row)

    def test_linear_chain_traffic_agrees_between_lessons(self):
        graph = self.graph["graph_cost"](4, 2, 100, 2)
        fusion = self.fusion["fusion_traffic"](4, 200, 2)
        self.assertEqual(graph["intermediate_traffic_bytes"], fusion["intermediate_bytes"])
        self.assertEqual(fusion["intermediate_bytes"], 400)

    def test_graph_plan_derives_consumers_and_group_boundaries(self):
        inspect = self.graph["execution_plan_report"]
        nodes = {"T": ("X", "W"), "U": ("T", "b"), "Y": ("U",)}
        inputs = {"X", "W", "b"}
        separate = inspect(nodes, [["T"], ["U"], ["Y"]], inputs)
        fused = inspect(nodes, [["T"], ["U", "Y"]], inputs)
        self.assertEqual(separate["users"], {"T": ["U"], "U": ["Y"], "Y": []})
        self.assertEqual(separate["materialized_intermediates"], ["T", "U"])
        self.assertEqual(fused["materialized_intermediates"], ["T"])
        self.assertEqual((separate["planned_calls"], fused["planned_calls"]), (3, 2))

    def test_graph_plan_keeps_intermediate_for_an_outside_consumer(self):
        inspect = self.graph["execution_plan_report"]
        nodes = {"T": ("X",), "U": ("T",), "Y": ("U",), "aux": ("U",)}
        partial = inspect(nodes, [["T"], ["U", "Y"], ["aux"]], {"X"})
        all_consumers = inspect(nodes, [["T"], ["U", "Y", "aux"]], {"X"})
        self.assertEqual(partial["users"]["U"], ["Y", "aux"])
        self.assertIn("U", partial["materialized_intermediates"])
        self.assertNotIn("U", all_consumers["materialized_intermediates"])

    def test_graph_plan_rejects_missing_duplicate_or_unready_nodes(self):
        inspect = self.graph["execution_plan_report"]
        nodes = {"T": ("X",), "U": ("T",)}
        for groups in ([["T"]], [["T"], ["T", "U"]], [["U"], ["T"]], [["T"], [], ["U"]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                inspect(nodes, groups, {"X"})
        with self.assertRaises(ValueError):
            inspect({"T": ("missing",)}, [["T"]], {"X"})
        with self.assertRaises(ValueError):
            inspect({"U": ("T",), "T": ("X",)}, [["T", "U"]], {"X"})

    def test_graph_rewrite_keeps_call_saving_when_layout_erases_byte_saving(self):
        cost = self.graph["graph_cost"]
        before = cost(3, 3, 1024, 4)
        for layout, expected in ((0, 8192), (8192, 0), (12288, -4096)):
            row = self.graph["graph_rewrite_delta"](before, cost(3, 2, 1024, 4, layout_bytes=layout))
            self.assertEqual(row["saved_launches"], 1)
            self.assertEqual(row["saved_launch_overhead_us"], 5)
            self.assertEqual(row["net_logical_traffic_saved_bytes"], expected)

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

    def test_bucket_counts_preserve_workload_and_boundary_lengths(self):
        """Compare exact useful and padded positions for fixed input lengths."""
        lengths = [64, 128, 256, 520, 800]
        direct = self.shapes["padding_overhead"](lengths=lengths)
        bucketed = self.shapes["padding_overhead"](
            lengths=lengths, bucket_boundaries=[128, 256, 512, 1024],
        )
        self.assertEqual(direct["wasted_tokens"], 2232)
        self.assertEqual(bucketed["wasted_tokens"], 792)
        self.assertEqual(direct["useful_tokens"], bucketed["useful_tokens"])
        edge = self.shapes["padding_overhead"](lengths=[128], bucket_boundaries=[256, 128, 128])
        self.assertEqual(edge["wasted_tokens"], 0)

    def test_runtime_branch_edges_do_not_claim_timing_stability(self):
        """Verify both routing thresholds and the meaning of the same-branch flag."""
        result = self.shapes["path_stability"](lengths=[255, 256, 1023, 1024])
        self.assertEqual(result["paths"], ["short", "medium", "medium", "long"])
        self.assertEqual(result["path_transitions"], 2)
        self.assertFalse(result["same_runtime_branch"])
        self.assertNotIn("stable", result)
        for boundaries in ((256, 256), (True, 1024), (1024, 256)):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                self.shapes["path_stability"](lengths=[128], boundaries=boundaries)

    def test_symbol_binding_rejects_unresolved_and_invalid_dimensions(self):
        source = {'op': 'matmul', 'dtype': 'fp16', 'shape': ('M', 256, 64)}
        for bindings in ({}, {'M': 0}, {'M': -1}, {'M': 8.5}, {'M': True}):
            with self.subTest(bindings=bindings), self.assertRaises(ValueError):
                self.compiler["specialize_shape"](source, bindings)
        specialized = self.compiler["specialize_shape"](source, bindings={'M': 128})
        self.assertEqual(source['shape'][0], 'M')
        self.assertEqual(specialized['shape'], (128, 256, 64))
        with self.assertRaises(ValueError):
            self.compiler["specialize_shape"](source)

    def test_plan_requires_legalization_and_all_tile_dimensions(self):
        source = {'op': 'matmul', 'dtype': 'fp16', 'shape': ('M', 256, 64)}
        shape = self.compiler["specialize_shape"](source, bindings={'M': 128})
        with self.assertRaises(ValueError):
            self.compiler["select_schedule"](shape)
        with self.assertRaises(ValueError):
            self.compiler["legalize_ops"]({**shape, 'dtype': 'bf16'})
        legal = self.compiler["legalize_ops"](shape)
        with self.assertRaises(ValueError):
            self.compiler["select_schedule"]({**legal, 'shape': (7, 256, 64)})
        scheduled = self.compiler["select_schedule"](legal)
        self.assertEqual(self.compiler["build_backend_plan"](scheduled)['representation'], 'python_plan')
        with self.assertRaises(ValueError):
            self.compiler["build_backend_plan"]({**scheduled, 'shape': (7, 256, 64)})

    def test_tuning_handles_empty_feasible_set_and_filters_fastest(self):
        rows = [dict(tiles_mnk=(8, 64, 64), warps=4, required_capabilities=set(), latency_us=80),
                dict(tiles_mnk=(64, 256, 64), warps=8, required_capabilities=set(), latency_us=40)]
        select = self.compiler['select_tuning_candidate']
        self.assertEqual(select(rows, (128, 256, 64), 32768)['selected']['tiles_mnk'], (8, 64, 64))
        self.assertIsNone(select(rows, (128, 256, 64), 1024)['selected'])
        self.assertIsNone(select([], (128, 256, 64), 32768)['selected'])
        with self.assertRaises(ValueError):
            select([{**rows[0], 'latency_us': float('inf')}], (128, 256, 64), 32768)

    def test_tuning_checks_each_constraint_before_constructed_latency(self):
        """A faster but ineligible candidate must not displace a feasible one."""
        base = dict(tiles_mnk=(8, 128, 64), warps=8, required_capabilities=set(), latency_us=60)
        rows = [base,
                base | {"tiles_mnk": (7, 128, 64), "latency_us": 10},
                base | {"warps": 9, "latency_us": 10},
                base | {"tiles_mnk": (64, 256, 64), "latency_us": 10}]
        result = self.compiler["select_tuning_candidate"](
            candidates=rows, problem_shape=(128, 256, 64), shared_memory_limit=32768, max_warps=8,
        )
        expected = base | {"shared_memory_bytes": 17408}
        self.assertEqual(result["selected"], expected)
        self.assertEqual(result["feasible"], [expected])
        self.assertEqual([row["reasons"] for row in result["rejected"]],
                         [["tile_divisibility"], ["warps"], ["shared_memory"]])
        with self.assertRaises(ValueError):
            self.compiler["select_tuning_candidate"](
                candidates=[base | {"latency_us": True}], problem_shape=(128, 256, 64),
                shared_memory_limit=65536,
            )

    def test_backend_plan_rechecks_shape_operator_and_schedule_content(self):
        """Reject stale success flags after changing the target plan's contents."""
        source = {"op": "matmul", "dtype": "fp16", "shape": ("M", 256, 64)}
        bound = self.compiler["specialize_shape"](graph=source, bindings={"M": 128})
        legal = self.compiler["legalize_ops"](graph=bound)
        scheduled = self.compiler["select_schedule"](graph=legal)
        self.assertEqual(scheduled["schedule"]["tiles_mnk"], (8, 128, 64))
        self.assertEqual(source["shape"], ("M", 256, 64))
        for change in ({"shape": (128, 256, 65)}, {"op": "other.matmul"},
                       {"schedule": {"tiles_mnk": (7, 64, 64), "warps": 4}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compiler["build_backend_plan"](graph=scheduled | change)
        with self.assertRaises(ValueError):
            self.compiler["legalize_ops"](graph=bound | {"shape_specialized": "yes"})

    def test_matrix_candidates_recompute_footprint_and_target_requirements(self):
        """Declared speed cannot bypass footprint or target capability requirements."""
        select = self.compiler['select_tuning_candidate']
        row = dict(tiles_mnk=(8, 128, 64), warps=4, latency_us=20,
                   required_capabilities={'shared_memory', 'mma_fp16'})
        args = dict(candidates=[row], problem_shape=(128, 256, 64), shared_memory_limit=17408)
        self.assertIsNone(select(**args, target_capabilities={'shared_memory'})['selected'])
        result = select(**args, target_capabilities={'shared_memory', 'mma_fp16'})
        self.assertEqual(result['selected']['shared_memory_bytes'], 17408)
        self.assertIsNone(select(**(args | {'shared_memory_limit': 17407}))['selected'])
        for tiles in ((7, 128, 64), (8, 127, 64), (8, 128, 63)):
            with self.subTest(tiles=tiles):
                report = select(**(args | {'candidates': [row | {'tiles_mnk': tiles}]}))
                self.assertIn('tile_divisibility', report['rejected'][0]['reasons'])

    def test_pipeline_rejects_incomplete_unknown_duplicate_and_bad_order(self):
        check = self.compiler['validate_pipeline']
        source = {'op': 'matmul', 'dtype': 'fp16', 'shape': ('M', 256, 64)}
        bound = self.compiler['specialize_shape'](source, {'M': 128})
        legal = self.compiler['legalize_ops'](bound)
        schedule = self.compiler['select_schedule'](legal)
        plan = self.compiler['build_backend_plan'](schedule)
        records = [('shape_specialize', bound), ('legalize_ops', legal),
                   ('scheduling', schedule), ('backend_plan', plan)]
        self.assertTrue(check(source, records)[0])
        for pipeline in ([], records[:1],
                         [('unknown', bound), *records[1:]],
                         [records[0], records[0], *records[2:]],
                         [records[1], records[0], *records[2:]]):
            with self.subTest(pipeline=pipeline):
                self.assertFalse(check(source, pipeline)[0])
        for index, change in ((0, {'dtype': 'bf16'}),
                              (1, {'shape': (64, 256, 64)}),
                              (1, {'op': 'wrong.matmul'}),
                              (2, {'schedule': {'tiles_mnk': (7, 128, 64), 'warps': 4}}),
                              (3, {'schedule': {'tiles_mnk': (8, 64, 64), 'warps': 4}})):
            broken = list(records)
            stage, content = broken[index]
            broken[index] = (stage, content | change)
            with self.subTest(index=index, change=change):
                self.assertFalse(check(source, broken)[0])


if __name__ == "__main__":
    unittest.main()
