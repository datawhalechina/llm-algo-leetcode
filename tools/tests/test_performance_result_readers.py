"""Check performance result readers and notebook integration without GPU execution."""

import ast
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from performance_result_review import (
    FORMATS,
    read_composition_result,
    read_performance_result,
    read_regression_result,
    review_performance_result,
)


class PerformanceMechanismReaderTests(unittest.TestCase):
    """Use recorded fixtures and the actual notebook readers without importing CUDA."""

    def records(self):
        for folder in ("perf_trace", "perf_memory_timeline", "perf_roofline", "perf_candidate"):
            for path in sorted((ROOT / "benchmarks/results" / folder).glob("*.json")):
                if ".trace." not in path.name:
                    yield path, json.loads(path.read_text())

    def expected(self, record):
        return dict(semantic_id=record["semantic_id"], run_label=record["run_label"],
                    workload_id=record["workload"]["id"],
                    current_version=FORMATS[record["semantic_id"]][2][-1])

    def test_all_historical_records_are_read_without_modification(self):
        records = list(self.records())
        self.assertEqual(len(records), 7)
        for path, record in records:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                loaded, review = read_performance_result(path, **self.expected(record))
                self.assertEqual(loaded, record)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(review["source_sha256"], hashlib.sha256(original).hexdigest())
                self.assertEqual(review["current_protocol"],
                                 record["workload"]["version"] == self.expected(record)["current_version"])

    def test_wrong_identity_and_nonfinite_values_stop_reading(self):
        for _, original in self.records():
            for field, value in (("run_label", "other"), ("schema_version", "unknown/v99"),
                                 ("semantic_id", "OTHER"), ("failure", "failed")):
                with self.subTest(semantic=original["semantic_id"], field=field):
                    record = copy.deepcopy(original)
                    record[field] = value
                    with self.assertRaises(ValueError):
                        review_performance_result(record, **self.expected(original))
            record = copy.deepcopy(original)
            record["auxiliary_metric"] = float("nan")
            with self.assertRaises(ValueError):
                review_performance_result(record, **self.expected(original))
            record = copy.deepcopy(original)
            record["workload"]["warmup_steps"] = True
            with self.assertRaises(ValueError):
                review_performance_result(record, **self.expected(original))

    def test_modified_statistics_and_incomplete_samples_are_rejected(self):
        for _, original in self.records():
            semantic = original["semantic_id"]
            if semantic == "PERF-TRACE" and original["workload"]["version"] == "v2":
                continue
            for corrupt_sample in (False, True):
                with self.subTest(semantic=semantic, version=original["workload"]["version"], sample=corrupt_sample):
                    record = copy.deepcopy(original)
                    if semantic == "PERF-TRACE":
                        samples = record["metrics"]["timing_samples_ms"]["sync_at_end"]
                        target, key = record["metrics"], "sync_at_end_speedup"
                    elif semantic == "PERF-MEMORY-TIMELINE":
                        samples = record["samples"]["candidate"]
                        target, key = record["comparison"], "latency_ratio"
                    elif semantic == "PERF-ROOFLINE":
                        samples = record["metrics"]["copy_samples_ms"]
                        target, key = record["empirical_ceilings"], "bandwidth_bytes_per_s"
                    else:
                        samples = record["matrix"]["variants"]["bf16_serial"]["loads"][0]["latency"]["samples_ms"]
                        target, key = record["comparisons"][0], "speedup"
                    if corrupt_sample:
                        samples.pop()
                    else:
                        target[key] *= 2
                    with self.assertRaises(ValueError):
                        review_performance_result(record, **self.expected(original))

    def test_candidate_protocol_and_per_load_decisions_are_checked(self):
        record = next(r for _, r in self.records() if r["semantic_id"] == "PERF-CANDIDATE"
                      and r["workload"]["version"] == "v2")
        expected = self.expected(record)
        for change in ("protocol", "decision", "duplicate", "bool_sample"):
            with self.subTest(change=change):
                altered = copy.deepcopy(record)
                if change == "protocol":
                    altered["matrix"]["protocol"]["paired_rounds"] -= 1
                elif change == "decision":
                    altered["comparisons"][0]["decision"] = "reject"
                elif change == "duplicate":
                    altered["comparisons"][1] = altered["comparisons"][0]
                else:
                    altered["matrix"]["variants"]["bf16_serial"]["loads"][0]["latency"]["samples_ms"][0] = True
                with self.assertRaises(ValueError):
                    review_performance_result(altered, **expected)

    def test_legacy_candidate_cosine_anomaly_is_visible_not_clipped(self):
        record = next(r for _, r in self.records() if r["semantic_id"] == "PERF-CANDIDATE"
                      and r["workload"]["version"] == "v1")
        original = copy.deepcopy(record)
        review = review_performance_result(record, **self.expected(record))
        self.assertFalse(review["current_protocol"])
        self.assertTrue(any("超出[-1,1]" in note for note in review["notes"]))
        self.assertEqual(record, original)

    def test_trace_counts_and_relocated_artifact_size_are_checked(self):
        path, record = next((p, r) for p, r in self.records() if r["semantic_id"] == "PERF-TRACE"
                            and r["workload"]["version"] == "v4")
        altered = copy.deepcopy(record)
        altered["metrics"]["selected_events"]["sync_at_end.host_sync"]["cpu_annotation_count"] += 1
        with self.assertRaises(ValueError):
            review_performance_result(altered, **self.expected(record))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / path.name
            source.write_text(json.dumps(record))
            _, review = read_performance_result(source, **self.expected(record))
            self.assertFalse(review["local_trace_exists"])
            source.with_suffix(".trace.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "trace_bytes"):
                read_performance_result(source, **self.expected(record))

    def test_actual_notebook_readers_use_the_shared_review(self):
        for semantic in FORMATS:
            notebook_path = next((ROOT / "02_PyTorch_Algorithms").glob(semantic + "_*.ipynb"))
            notebook = json.loads(notebook_path.read_text())
            code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            config = next(s for s in code if s.startswith('RUN_MODE = "dry_run"'))
            reader = next(s for s in code if "read_performance_result(" in s)
            record = next(r for _, r in self.records() if r["semantic_id"] == semantic
                          and r["workload"]["version"] == FORMATS[semantic][2][-1])
            scope = {}
            exec(config, scope)
            scope["RUN_LABEL"] = record["run_label"]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(reader, scope)
            self.assertEqual(scope["result_review"]["status"], "record_statistics_checked")


class PairedRegressionReaderTests(unittest.TestCase):
    """核对历史配对样本重算与错误记录拦截，不运行编译或GPU。"""

    def setUp(self):
        self.path = ROOT / "benchmarks/results/perf_regression/perf-paired-regression_rtx5070ti_regression_v3.json"
        self.record = json.loads(self.path.read_text())

    def read_fixture(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps(record))
            return read_regression_result(path, run_label=self.record["run_label"])

    def test_historical_records_recompute_without_filling_missing_fields(self):
        for path in sorted(self.path.parent.glob("*.json")):
            original = path.read_bytes()
            record = json.loads(original)
            loaded, review = read_regression_result(path, run_label=record["run_label"])
            self.assertEqual(loaded, record)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(review["status"], "historical_record_checked")
            self.assertTrue(any("regression_threshold" in note for note in review["notes"]))

    def test_tampered_intervals_pairs_and_decisions_stop(self):
        for kind in ("interval", "duplicate", "nan", "decision", "overall", "label"):
            with self.subTest(kind=kind):
                record = copy.deepcopy(self.record)
                load = record["metrics"]["load_matrix"][0]
                if kind == "interval":
                    load["summary"]["ci_low"] *= 2
                elif kind == "duplicate":
                    load["paired_runs"][1]["trial_id"] = 0
                elif kind == "nan":
                    load["paired_runs"][0]["candidate_ms"] = float("nan")
                elif kind == "decision":
                    load["decision"]["decision"] = "accept"
                elif kind == "overall":
                    record["decision"]["decision"] = "accept"
                else:
                    record["run_label"] = "other"
                with self.assertRaises(ValueError):
                    self.read_fixture(record)

    def test_execution_failure_is_readable_but_cannot_pass(self):
        record = copy.deepcopy(self.record)
        load = record["metrics"]["load_matrix"][0]
        load["paired_runs"] = []
        load["summary"] = None
        load["first_execution"] = None
        load["quality"]["outputs_allclose"] = False
        load["failure"] = {"type": "RuntimeError", "phase": "first_execution", "trial_id": None}
        load["decision"] = {"decision": "reject", "reason": "candidate_execution_failed"}
        record["failure"] = [{"token_count": load["token_count"], **load["failure"]}]
        self.read_fixture(record)
        load["decision"]["decision"] = "accept"
        with self.assertRaises(ValueError):
            self.read_fixture(record)


class CompositionReaderTests(unittest.TestCase):
    """从保存的样本复算历史和当前决策，当前分支只在CPU构造测试。"""

    def setUp(self):
        self.directory = ROOT / "benchmarks/results/perf_composition"
        self.record = json.loads((self.directory / "perf-composition_rtx5070ti_composition_rotating_v2.json").read_text())

    def read_fixture(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps(record))
            return read_composition_result(path, run_label=self.record["run_label"])

    def current_fixture(self, record):
        """只执行Notebook现有的比较循环，不导入torch或执行采样。"""
        record = copy.deepcopy(record)
        path = next((ROOT / "02_PyTorch_Algorithms").glob("PERF-COMPOSITION*.ipynb"))
        notebook = json.loads(path.read_text())
        code = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "gpu-exec")
        branch = next(node for node in ast.parse(code).body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Compare) and ast.unparse(node.test) == "RUN_MODE == 'real_gpu'")
        statements, collecting = [], False
        for node in branch.body:
            target = node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
            if target == "loads":
                collecting = True
            if target == "record":
                break
            if collecting:
                statements.append(node)
        scope = {"matrix": record["matrix"], "GPU_CONFIG": record["workload"]}
        exec(compile(ast.Module(body=statements, type_ignores=[]), "composition_comparisons", "exec"), scope)
        record["comparisons"] = scope["comparisons"]
        record["schema_version"] = "performance-composition-mechanism/v2"
        record["decision_version"] = "guarded_fallback/v2"
        return record

    def test_history_remains_unchanged_and_marked_historical(self):
        for path in self.directory.glob("*.json"):
            original = path.read_bytes()
            record = json.loads(original)
            loaded, review = read_composition_result(path, run_label=record["run_label"])
            self.assertEqual(loaded, record)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(review["status"], "historical_record_checked")

    def test_current_notebook_comparison_loop_matches_reader(self):
        record = self.current_fixture(self.record)
        _, review = self.read_fixture(record)
        self.assertEqual(review["status"], "record_statistics_checked")
        self.assertTrue(all(item["decision"] == "keep_combination" for item in review["recomputed_comparisons"]))

    def test_unqualified_fastest_single_is_not_the_fallback(self):
        record = copy.deepcopy(self.record)
        for row in record["matrix"]["variants"]["bf16_serial"]["loads"]:
            row["cosine_vs_fp32"] = 0.5
        record = self.current_fixture(record)
        _, review = self.read_fixture(record)
        for item in review["recomputed_comparisons"]:
            self.assertNotEqual(item["rollback_to"], "bf16_serial")
            self.assertFalse(item["single_checks_recomputed"]["bf16_serial"]["quality_passed"])

    def test_no_qualified_single_rolls_back_to_baseline(self):
        record = copy.deepcopy(self.record)
        for name in ("bf16_serial", "fp32_overlap", "bf16_overlap"):
            for row in record["matrix"]["variants"][name]["loads"]:
                row["cosine_vs_fp32"] = 0.5
        record = self.current_fixture(record)
        _, review = self.read_fixture(record)
        for item in review["recomputed_comparisons"]:
            self.assertIsNone(item["best_single"])
            self.assertEqual(item["decision"], "rollback_baseline")
            self.assertEqual(item["rollback_to"], "baseline_fp32_serial")

    def test_actual_reader_uses_shared_validation(self):
        path = next((ROOT / "02_PyTorch_Algorithms").glob("PERF-COMPOSITION*.ipynb"))
        notebook = json.loads(path.read_text())
        scope = {}
        for cell_id in ("gpu-config", "gpu-read"):
            code = "".join(next(c for c in notebook["cells"] if c.get("id") == cell_id)["source"])
            if cell_id == "gpu-read":
                scope["RUN_LABEL"] = self.record["run_label"]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, scope)
        self.assertEqual(scope["result_review"]["status"], "historical_record_checked")

    def test_modified_samples_decisions_and_versions_are_rejected(self):
        for change in ("sample", "median", "decision", "rollback", "version", "checks"):
            with self.subTest(change=change):
                record = self.current_fixture(self.record)
                if change == "sample":
                    record["matrix"]["variants"]["bf16_overlap"]["loads"][0]["latency"]["samples_ms"].pop()
                elif change == "median":
                    record["matrix"]["variants"]["bf16_overlap"]["loads"][0]["latency"]["median_ms"] *= 2
                elif change == "decision":
                    record["comparisons"][0]["decision"] = "keep_single"
                elif change == "rollback":
                    record["comparisons"][0]["rollback_to"] = "baseline_fp32_serial"
                elif change == "version":
                    record["decision_version"] = "unknown"
                else:
                    record["comparisons"][0]["single_checks"]["bf16_serial"]["quality_passed"] = False
                with self.assertRaises(ValueError):
                    self.read_fixture(record)


class InputIOReaderTests(unittest.TestCase):
    """只核验既有输入汇总和I/O流水线样本，不加载GPU或临时产物。"""

    def records(self):
        for folder in ("perf_input_pipeline", "perf_io_critical_path"):
            for path in (ROOT / "benchmarks/results" / folder).glob("*.json"):
                yield path, json.loads(path.read_text())

    def read_fixture(self, record, expected=None):
        from performance_result_review import read_input_io_result
        expected = expected or record
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(record))
            return read_input_io_result(path, semantic_id=expected["semantic_id"], run_label=expected["run_label"])

    def test_history_is_checked_without_mutation(self):
        from performance_result_review import read_input_io_result
        for path, record in self.records():
            original = path.read_bytes()
            loaded, review = read_input_io_result(path, semantic_id=record["semantic_id"], run_label=record["run_label"])
            self.assertEqual(loaded, record)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(review["source_sha256"], hashlib.sha256(original).hexdigest())
            self.assertIn(review["status"], ("summary_fields_checked", "historical_record_checked"))

    def test_wrong_identity_and_invalid_values_stop(self):
        for _, original in self.records():
            for key, value in (("run_label", "other"), ("schema_version", "unknown/v9"), ("failure", "failed")):
                record = copy.deepcopy(original)
                record[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.read_fixture(record, original)
            record = copy.deepcopy(original)
            record["extra_metric"] = float("nan")
            with self.assertRaises(ValueError):
                self.read_fixture(record, original)

    def test_input_checksum_and_decision_are_checked(self):
        original = next(record for _, record in self.records() if record["semantic_id"] == "PERF-INPUT")
        for change in ("checksum", "decision", "boolean_time"):
            record = copy.deepcopy(original)
            if change == "checksum":
                record["correctness"]["checksum_relative_diff"] = 1
            elif change == "decision":
                record["decision"] = "tune_input_pipeline"
            else:
                record["metrics"]["pageable_single"]["step_median_ms"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_fixture(record)

    def test_io_sample_counts_statistics_and_v3_checks(self):
        original = next(record for _, record in self.records() if record["schema_version"] == "performance-io-mechanism/v2")
        for change in ("count", "median", "speedup", "decision"):
            record = copy.deepcopy(original)
            pipeline = record["metrics"]["pipeline_critical_path"]
            if change == "count":
                pipeline["overlap_samples_ms"].pop()
            elif change == "median":
                pipeline["double_buffer_overlap"]["median_ms"] *= 2
            elif change == "speedup":
                pipeline["median_speedup"] *= 2
            else:
                record["decision"] = "keep_serial_path"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_fixture(record)
        # 仅构造v3字段用于CPU分支测试，不保存为新增GPU记录。
        record = copy.deepcopy(original)
        record["schema_version"] = "performance-io-mechanism/v3"
        record["workload"]["version"] = "v3"
        record["correctness"].update(artifact_load_passed=True, output_reload_passed=True)
        self.assertEqual(self.read_fixture(record)[1]["status"], "record_statistics_checked")
        record["correctness"]["output_reload_passed"] = False
        with self.assertRaises(ValueError):
            self.read_fixture(record)

    def test_actual_notebook_readers_support_all_registered_versions(self):
        for _, record in self.records():
            path = next((ROOT / "02_PyTorch_Algorithms").glob(record["semantic_id"] + "_*.ipynb"))
            notebook = json.loads(path.read_text())
            scope = {}
            for cell_id in ("gpu_config", "gpu_read"):
                code = "".join(next(c for c in notebook["cells"] if c.get("id") == cell_id)["source"])
                if cell_id == "gpu_read":
                    scope["RUN_LABEL"] = record["run_label"]
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(code, scope)
            self.assertIn(scope["result_review"]["status"], ("summary_fields_checked", "historical_record_checked"))


if __name__ == "__main__":
    unittest.main()
