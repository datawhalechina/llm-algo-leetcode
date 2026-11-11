"""CPU-only coverage of project 66 integrity, stage safety and immutable registration."""

from __future__ import annotations

import copy
import contextlib
import io
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import inference_baseline_runtime as baseline
from tools import inference_project_runtime as runtime


class BaselineRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        workload = self.root / "benchmarks/workloads/fixed.jsonl"
        workload.parent.mkdir(parents=True)
        workload.write_text('{"prompt":"hello"}\n', encoding="utf-8")
        self.contract_path = self.root / "benchmarks/contracts/inference_baseline_v3.json"
        self.contract_path.parent.mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / "benchmarks/contracts/inference_baseline_v3.json"
        design = json.loads(source.read_text(encoding="utf-8"))
        design["parameters"]["workload_sha256"] = baseline.sha256_file(workload)
        self.contract_path.write_text(json.dumps(design), encoding="utf-8")
        self.registry_path = self.root / "benchmarks/project_registry.json"
        self.registry_path.write_text(json.dumps({"projects": [{
            "semantic_id": "INF-FLOAT-BASELINE",
            "default_experiment_contract": "benchmarks/contracts/inference_baseline_v3.json",
        }]}), encoding="utf-8")
        self.plan = baseline.load_baseline_plan(self.root, "test_baseline_v5", "prepare")
        self.plan["run_stage"] = "close"
        self.contract = self.plan["contract"]

    def make_report(self):
        """Create inspectable synthetic samples; these are not GPU evidence."""
        rows = [{
            "ok": True, "request_id": str(index), "output_text": "reply",
            "output_tokens": 4, "finish_reason": "stop", "estimated_output_tokens": False,
            "submit_offset_ms": index * 50.0, "first_content_offset_ms": index * 50.0 + 10,
            "last_content_offset_ms": index * 50.0 + 30,
            "completion_offset_ms": index * 50.0 + 40, "stream_done_received": True,
        } for index in range(self.contract["num_prompts"])]
        metrics = {"successful_requests": len(rows), "failed_requests": 0,
                   "output_tokens": len(rows) * 4,
                   "output_token_throughput_per_s": len(rows) * 2, "duration_s": 2.0}
        run = {"results": rows, "metrics": metrics,
               "workload": {"requests": len(rows), "max_tokens": 64,
                            "concurrency": 1, "warmup": 5, "repeats": 1}}
        identity = {key: self.contract[key] for key in (
            "model_id", "workload_sha256", "profile_id", "dtype", "timing_protocol")}
        identity.update(model_revision="a" * 40, tokenizer_revision="a" * 40)
        return {
            "schema_version": "inference-benchmark/v1", "run_label": self.plan["run_label"],
            "baseline_protocol_version": baseline.PROTOCOL, "baseline_contract": self.contract,
            "experiment_contract": copy.deepcopy(self.plan["experiment_contract"]),
            "model_identity": {"resolved_snapshot": "a" * 40}, "measurement_identity": identity,
            "environment_preflight": {"ready": True},
            "resource_measurement": {"scope": "visible_gpu_0_total_device_usage", "sample_count": 5},
            "runs": [copy.deepcopy(run) for _ in range(self.contract["repeats"])],
            "metrics": {**metrics, "peak_memory_mb": 1000}, "results": copy.deepcopy(rows),
        }

    def save_evidence(self, report):
        """Write only disposable test assets under a TemporaryDirectory."""
        raw = self.root / self.plan["raw_path"]
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(report), encoding="utf-8")
        report["raw_sha256"] = baseline.sha256_file(raw)
        (self.root / self.plan["result_path"]).write_text(json.dumps(report), encoding="utf-8")
        (self.root / self.plan["log_path"]).write_text("test log", encoding="utf-8")

    def test_ready_is_integrity_not_task_quality(self):
        report = self.make_report()
        evidence = baseline.audit_baseline_report(report, self.contract, self.plan["run_label"])
        self.assertEqual(evidence["decision"]["status"], "ready")
        self.assertIn("not_task_quality", evidence["decision"]["scope"])
        self.assertEqual(len(evidence["runs"]), 3)
        self.assertEqual(evidence["runs"][0]["client_metrics"]["ttft_content_ms"]["samples"], 30)
        self.assertEqual(evidence["runs"][0]["client_metrics"]["stream_tail_ms"]["p95"], 10)
        observation = evidence["runs"][0]["observations"][0]
        self.assertEqual(observation["stream_tail_ms"], 10)
        self.assertAlmostEqual(observation["content_tpot_proxy_ms"], 20 / 3)

    def test_new_design_parameters_are_loaded_without_runtime_changes(self):
        snapshot = json.loads(self.contract_path.read_text())
        snapshot["contract_version"] = "v4"
        snapshot["parameters"].update(model_id="example/another-model", num_prompts=7,
                                       repeats=2, dtype="float16")
        path = self.contract_path.with_name("inference_baseline_v4.json")
        path.write_text(json.dumps(snapshot), encoding="utf-8")
        registry = json.loads(self.registry_path.read_text())
        registry["projects"][0]["default_experiment_contract"] = str(path.relative_to(self.root))
        self.registry_path.write_text(json.dumps(registry), encoding="utf-8")
        plan = baseline.load_baseline_plan(self.root, "new_design", "prepare")
        self.assertEqual(plan["contract"]["model_id"], "example/another-model")
        self.assertEqual(plan["contract"]["num_prompts"], 7)
        self.assertEqual(plan["contract"]["repeats"], 2)
        self.assertEqual(plan["experiment_contract"]["snapshot"]["contract_version"], "v4")

    def test_upgrade_default_does_not_rejudge_saved_measurement(self):
        self.save_evidence(self.make_report())
        original = copy.deepcopy(self.plan["experiment_contract"])
        next_contract = json.loads(self.contract_path.read_text())
        next_contract["contract_version"] = "v4"
        next_contract["parameters"].update(num_prompts=7, repeats=2)
        next_path = self.contract_path.with_name("inference_baseline_v4.json")
        next_path.write_text(json.dumps(next_contract), encoding="utf-8")
        registry = json.loads(self.registry_path.read_text())
        registry["projects"][0]["default_experiment_contract"] = str(next_path.relative_to(self.root))
        self.registry_path.write_text(json.dumps(registry), encoding="utf-8")
        saved = baseline.load_baseline_plan(self.root, self.plan["run_label"], "close")
        self.assertEqual(saved["contract_source"], "saved_measurement")
        self.assertEqual(saved["experiment_contract"], original)
        evidence = baseline.read_baseline_evidence(saved)
        self.assertEqual(evidence["decision"]["status"], "ready")
        self.assertEqual(baseline.close_baseline(saved, evidence)["decision"], "ready")
        reference = json.loads((self.root / saved["reference_path"]).read_text())
        self.assertEqual(reference["experiment_contract"], original)

    def test_read_and_close_do_not_need_current_registry_contract_or_workload(self):
        self.save_evidence(self.make_report())
        self.registry_path.unlink()  # Disposable fixture files only.
        self.contract_path.unlink()
        (self.root / self.contract["workload_path"]).unlink()
        saved = baseline.load_baseline_plan(self.root, self.plan["run_label"], "close")
        evidence = baseline.read_baseline_evidence(saved)
        self.assertEqual(evidence["decision"]["status"], "ready")
        self.assertEqual(baseline.close_baseline(saved, evidence)["status"], "registered")

    def test_contract_rejects_invalid_types_schema_and_unknown_protocol(self):
        original = json.loads(self.contract_path.read_text())
        mutations = [
            ("num_prompts", 0), ("repeats", True), ("warmup", -1),
            ("gpu_memory_utilization", 1.1), ("temperature", float("nan")),
            ("enforce_eager", "true"), ("dtype", "auto"),
            ("timing_protocol", "unsupported/v99"), ("workload_sha256", "invalid"),
            ("surprise_parameter", 1),
        ]
        for key, value in mutations:
            with self.subTest(parameter=key):
                snapshot = copy.deepcopy(original)
                snapshot["parameters"][key] = value
                self.contract_path.write_text(json.dumps(snapshot), encoding="utf-8")
                with self.assertRaises(ValueError):
                    baseline.load_baseline_plan(self.root, "invalid_design", "prepare")
        original["schema_version"] = "unknown/v1"
        self.contract_path.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema"):
            baseline.load_baseline_plan(self.root, "invalid_schema", "prepare")

    def test_workload_content_must_match_contract(self):
        (self.root / self.contract["workload_path"]).write_text('{"prompt":"changed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "工作负载"):
            baseline.load_baseline_plan(self.root, "changed_input", "prepare")

    def test_contract_paths_cannot_escape_repository(self):
        registry = json.loads(self.registry_path.read_text())
        for source in ("../outside.json", str(self.contract_path)):
            registry["projects"][0]["default_experiment_contract"] = source
            self.registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaises(ValueError):
                baseline.load_baseline_plan(self.root, "bad_path", "prepare")
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "external.json"
            external.write_text(self.contract_path.read_text(), encoding="utf-8")
            link = self.contract_path.with_name("outside-link.json")
            link.symlink_to(external)
            registry["projects"][0]["default_experiment_contract"] = str(link.relative_to(self.root))
            self.registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaises(ValueError):
                baseline.load_baseline_plan(self.root, "bad_link", "prepare")

    def test_archived_contract_tampering_never_falls_back_to_default(self):
        report = self.make_report()
        report["experiment_contract"]["snapshot"]["parameters"]["repeats"] = 1
        self.save_evidence(report)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            baseline.load_baseline_plan(self.root, self.plan["run_label"], "read")

    def test_contract_hash_ignores_formatting_but_not_parameter_changes(self):
        original = self.plan["experiment_contract"]["sha256"]
        snapshot = json.loads(self.contract_path.read_text())
        self.contract_path.write_text(json.dumps(snapshot, indent=4, sort_keys=True), encoding="utf-8")
        reformatted = baseline.load_baseline_plan(self.root, "formatting", "prepare")
        self.assertEqual(reformatted["experiment_contract"]["sha256"], original)
        snapshot["parameters"]["repeats"] = 4
        self.contract_path.write_text(json.dumps(snapshot), encoding="utf-8")
        changed = baseline.load_baseline_plan(self.root, "changed_parameter", "prepare")
        self.assertNotEqual(changed["experiment_contract"]["sha256"], original)

    def test_single_token_is_valid_with_no_tpot(self):
        report = self.make_report()
        for run in report["runs"]:
            run["results"][0].update(output_tokens=1, last_content_offset_ms=10)
            run["metrics"]["output_tokens"] -= 3
            run["metrics"]["output_token_throughput_per_s"] -= 1.5
        evidence = baseline.audit_baseline_report(report, self.contract, self.plan["run_label"])
        self.assertEqual(evidence["decision"]["status"], "ready")
        self.assertIsNone(evidence["runs"][0]["observations"][0]["content_tpot_proxy_ms"])

    def test_last_repeat_must_be_audited(self):
        report = self.make_report()
        report["runs"][-1]["results"][0]["ok"] = False
        evidence = baseline.audit_baseline_report(report, self.contract, self.plan["run_label"])
        self.assertEqual(evidence["decision"]["status"], "reject")
        self.assertIn("run3.request1.request_failed", evidence["failures"])

    def test_repeat_and_workload_contract_mismatches(self):
        for change in ("repeat", "workload"):
            report = self.make_report()
            if change == "repeat":
                report["runs"].pop()
            else:
                report["runs"][0]["workload"]["concurrency"] = 2
            evidence = baseline.audit_baseline_report(report, self.contract, self.plan["run_label"])
            self.assertEqual(evidence["decision"]["status"], "reject")

    def test_missing_window_does_not_hide_known_sample_failure(self):
        rows = [{"ok": True, "output_tokens": 4}]
        metrics = {"successful_requests": 1, "failed_requests": 0, "output_tokens": 4,
                   "output_token_throughput_per_s": 4, "duration_s": None}
        checked = baseline.check_baseline_totals(rows, metrics, 2)
        self.assertIn("duration_s", checked["missing"])
        self.assertIn("sample_counts", checked["failures"])

    def test_coherent_rate_cannot_use_window_shorter_than_requests(self):
        report = self.make_report()
        report["runs"][0]["metrics"].update(duration_s=0.5,
                                            output_token_throughput_per_s=240)
        checked = baseline.audit_baseline_report(report, self.contract, self.plan["run_label"])
        self.assertIn("run1.throughput_window", checked["failures"])

    def test_missing_memory_or_estimated_tokens_are_not_ready(self):
        for change in ("memory", "tokens"):
            report = self.make_report()
            if change == "memory":
                report["resource_measurement"]["scope"] = "unknown"
            else:
                report["runs"][0]["results"][0]["estimated_output_tokens"] = True
            evidence = baseline.audit_baseline_report(report, self.contract, self.plan["run_label"])
            self.assertEqual(evidence["decision"]["status"], "tune")

    def test_identity_label_and_nonfinite_record_fail(self):
        for change in ("identity", "label", "time"):
            report = self.make_report()
            if change == "identity":
                report["measurement_identity"]["workload_sha256"] = "other"
            elif change == "label":
                report["run_label"] = "another"
            else:
                report["runs"][0]["results"][0]["completion_offset_ms"] = float("nan")
            self.assertEqual(baseline.audit_baseline_report(
                report, self.contract, self.plan["run_label"])["decision"]["status"], "reject")

    def test_historical_report_is_read_only_and_cannot_close(self):
        historical = {"quality": {"status": "passed"}, "metrics": {"successful_requests": 30}}
        self.save_evidence(historical)
        path = self.root / self.plan["result_path"]
        before = path.read_bytes()
        evidence = baseline.read_baseline_evidence(self.plan)
        self.assertTrue(evidence["historical_protocol"])
        self.assertEqual(evidence["decision"]["status"], "tune")
        with self.assertRaises(RuntimeError):
            baseline.close_baseline(self.plan, evidence)
        self.assertEqual(path.read_bytes(), before)

    def test_raw_hash_and_content_mismatch(self):
        self.save_evidence(self.make_report())
        raw = self.root / self.plan["raw_path"]
        raw.write_text(raw.read_text() + " ", encoding="utf-8")
        self.assertIn("raw_sha256_mismatch", baseline.read_baseline_evidence(self.plan)["failures"])
        self.save_evidence(self.make_report())
        path = self.root / self.plan["result_path"]
        changed = json.loads(path.read_text())
        changed["metrics"]["peak_memory_mb"] = 2000
        path.write_text(json.dumps(changed), encoding="utf-8")
        self.assertIn("raw_content_mismatch", baseline.read_baseline_evidence(self.plan)["failures"])

    def test_close_idempotence_partial_recovery_and_conflict(self):
        self.save_evidence(self.make_report())
        evidence = baseline.read_baseline_evidence(self.plan)
        self.assertEqual(baseline.close_baseline(self.plan, evidence)["status"], "registered")
        self.assertEqual(baseline.close_baseline(self.plan, evidence)["status"], "already_complete")
        reference = self.root / self.plan["reference_path"]
        reference.unlink()  # Disposable fixture, not a repository result.
        self.assertEqual(baseline.close_baseline(self.plan, evidence)["status"], "registered")
        reference.write_text('{}', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            baseline.close_baseline(self.plan, evidence)

    def test_reject_registration_never_produces_reference(self):
        report = self.make_report()
        report["runs"][0]["results"][0]["output_text"] = ""
        self.save_evidence(report)
        receipt = baseline.close_baseline(self.plan, baseline.read_baseline_evidence(self.plan))
        self.assertEqual(receipt["decision"], "reject")
        self.assertFalse((self.root / self.plan["reference_path"]).exists())

    def test_stage_and_label_guards_do_not_launch(self):
        for label, stage in (("../bad", "read"), ("replace_with_unique_label", "measure"),
                             ("valid", "unknown")):
            with self.assertRaises(ValueError):
                baseline.load_baseline_plan(self.root, label, stage)
        with mock.patch.object(runtime, "start_optional_vllm") as start:
            with self.assertRaises(RuntimeError):
                baseline.collect_baseline(self.plan, {"ready": True})
            start.assert_not_called()

    def test_collection_cleanup_and_success_failure_persistence(self):
        plan = {**self.plan, "run_stage": "measure"}
        snapshot = self.root / ("a" * 40)
        snapshot.mkdir()

        def fake_benchmark(**kwargs):
            report = self.make_report()
            Path(kwargs["output"]).write_text(json.dumps(report), encoding="utf-8")
            return report

        (self.root / plan["result_path"]).parent.mkdir(parents=True, exist_ok=True)
        with (
            mock.patch.object(runtime, "start_optional_vllm", return_value=("server", "log", 8001, "bfloat16", str(snapshot))),
            mock.patch.object(runtime, "run_backend_benchmark", side_effect=fake_benchmark),
            mock.patch.object(runtime, "stop_optional_vllm") as stop,
        ):
            baseline.collect_baseline(plan, {"ready": True})
            stop.assert_called_once_with("server", "log")
        report = json.loads((self.root / plan["result_path"]).read_text())
        self.assertEqual(report["baseline_protocol_version"], baseline.PROTOCOL)
        self.assertEqual(report["quality"]["task_quality"], "not_evaluated")
        failed_plan = baseline.load_baseline_plan(self.root, "test_failure", "measure")
        with (
            mock.patch.object(runtime, "start_optional_vllm", return_value=("server", "log", 8002, "bfloat16", str(snapshot))),
            mock.patch.object(runtime, "run_backend_benchmark", side_effect=RuntimeError("simulated")),
            mock.patch.object(runtime, "stop_optional_vllm") as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                baseline.collect_baseline(failed_plan, {"ready": True})
            stop.assert_called_once_with("server", "log")
        self.assertEqual(json.loads((self.root / failed_plan["result_path"]).read_text())["failure"]["message"], "simulated")

    def test_prepare_notebook_is_safe_from_cold_and_hot_imports(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
            "02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb").read_text())
        project_cells = [cell for cell in notebook["cells"] if any(
            tag.startswith("project-") for tag in cell.get("metadata", {}).get("tags", []))]
        scope = {}
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in project_cells:
                    exec(compile("".join(cell["source"]), "66-prepare", "exec"), scope)
            self.assertEqual(scope["registration_receipt"]["status"], "not_registered")
            self.assertEqual(scope["RUN_STAGE"], "prepare")
        self.assertFalse((Path(scope["plan"]["root"]) / scope["plan"]["result_path"]).exists())

    def test_notebook_reads_historical_v4_without_cuda_or_writes(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb").read_text())
        label = "rtx5070ti_vllm_g0_v4"
        path = root / f"benchmarks/results/66_g0_vllm_{label}.json"
        before = path.read_bytes()
        scope = {"RUN_LABEL": label, "RUN_STAGE": "read"}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                tags = cell.get("metadata", {}).get("tags", [])
                if any(tag.startswith("project-") and tag != "project-config" for tag in tags):
                    exec(compile("".join(cell["source"]), "66-read", "exec"), scope)
        self.assertTrue(scope["baseline_evidence"]["historical_protocol"])
        self.assertEqual(scope["registration_receipt"]["status"], "not_registered")
        self.assertEqual(path.read_bytes(), before)

    def test_cpu_answer_and_real_evidence_rules_stay_identical(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb").read_text())
        answer_source = next("".join(cell["source"]) for cell in notebook["cells"]
                             if "tutorial-answer" in cell.get("metadata", {}).get("tags", []))
        helper_source = (root / "tools/inference_baseline_runtime.py").read_text()

        def rules(source):
            """Ignore prose docstrings while comparing the actual teaching predicates."""
            functions = {}
            for node in ast.parse(source).body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    node.body.pop(0)
                functions[node.name] = ast.dump(node, include_attributes=False)
            return functions

        answer_rules, helper_rules = rules(answer_source), rules(helper_source)
        self.assertEqual(len(answer_rules), 4)
        for name, definition in answer_rules.items():
            self.assertEqual(definition, helper_rules[name], name)


if __name__ == "__main__":
    unittest.main()
