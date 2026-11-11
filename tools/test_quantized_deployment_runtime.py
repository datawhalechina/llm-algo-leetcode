"""CPU-only synthetic fixtures for projects 67/83; never writes repository evidence."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import quantization_project_runtime as gate
from tools import quantized_deployment_runtime as runtime
from tools.quantization_runtime import build_candidate_bundle, save_candidate_bundle
from tools import test_quantization_project_runtime as gate_tests


class DeploymentLifecycleTests(unittest.TestCase):
    def setUp(self):
        """Build a complete fake 82 upstream under a temporary directory."""
        fixture = gate_tests.ArtifactEvidenceLifecycleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.fixture = fixture.root, fixture
        fixture.source["workload"]["resolved_model_commit"] = "a" * 40
        fixture.write(fixture.plan["source_path"], fixture.source)
        fixture.write(fixture.artifact / "config.json", {"quantization_config": {"quant_method": "compressed-tensors"}})
        bundle = build_candidate_bundle(fixture.source, candidate_name="gptq", artifact_dir=fixture.artifact,
            project_root=self.root, source_result_path=fixture.plan["source_path"])
        save_candidate_bundle(fixture.artifact / "candidate_bundle.json", bundle)
        fixture.measure()
        gate.close_artifact_gate(fixture.closing_plan())
        workload_path = self.root / "benchmarks/workloads/fixed.jsonl"
        workload_path.parent.mkdir(parents=True, exist_ok=True)
        workload_path.write_text(json.dumps({"prompt": "synthetic CPU fixture"}) + "\n")
        self.plan = runtime.project_plan(self.root, "67", "deploy", "run", "measure")
        cache = self.root / "hub"
        snapshot = cache / "models--test--model/snapshots" / ("a" * 40)
        for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
            fixture.write(snapshot / name, {"synthetic": True})
        (snapshot / "tokenizer.json").write_text((fixture.artifact / "tokenizer.json").read_text())
        self.cache_patch = patch.dict("os.environ", {"HF_HUB_CACHE": str(cache)})
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.calls = []

    def runner(self, plan, prepared, key, work, role):
        """Emit synthetic observations, marked by names; no backend is imported."""
        self.calls.append(key)
        metrics = {"successful_requests": work["num_prompts"], "failed_requests": 0,
                   "tpot_ms": {"p50": 10., "p95": 20.}, "ttft_ms": {"p95": 100.},
                   "output_token_throughput_per_s": 50.}
        row = {"workload": {"requests": work["num_prompts"], "concurrency": work["concurrency"], "max_tokens": work["max_tokens"]},
               "metrics": metrics}
        raw = {**row, "runs": [deepcopy(row) for _ in range(work["repeats"])]} if work["repeats"] > 1 else row
        return {"status": "ok", "raw": raw, "backend_version": "CPU-FIXTURE-NOT-REAL", "hardware": {"device": "synthetic"},
                "dtype": "bfloat16", "resolved_model": prepared["snapshot"]["float_model_path"] if role == "baseline" else prepared["snapshot"]["artifact_path"],
                "execution": {"real_backend": True, "quantized_path": True, "fallback": False}, "failure": None}

    def measure(self, runner=None):
        return runtime.measure_project(self.plan, runner or self.runner)

    def close_plan(self):
        return {**self.plan, "stage": "close"}

    def serve_plan(self):
        self.measure()
        runtime.close_project(self.close_plan())
        return runtime.project_plan(self.root, "83", "serve", "deploy", "measure")

    def test_prepare_read_close_have_no_subprocesses_or_writes(self):
        before = set(self.root.rglob("*"))
        with patch("subprocess.run", side_effect=AssertionError("no backend")):
            self.assertTrue(runtime.preflight_project({**self.plan, "stage": "prepare"})["ready"])
            self.assertEqual(runtime.close_project(self.close_plan())["decision"]["decision"], "pending")
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_alternating_pairs_measure_only_then_idempotent_close(self):
        result = self.measure()
        self.assertEqual(result["decision"]["decision"], "accept")
        self.assertEqual(self.calls, ["pair1_baseline", "pair1_candidate", "pair2_candidate", "pair2_baseline", "pair3_baseline", "pair3_candidate"])
        self.assertFalse(self.plan["reference_path"].exists())
        runtime.close_project(self.close_plan())
        stamps = {k: self.plan[k].stat().st_mtime_ns for k in ("raw_path", "report_path", "reference_path", "receipt_path")}
        runtime.close_project(self.close_plan())
        self.assertEqual(stamps, {k: self.plan[k].stat().st_mtime_ns for k in stamps})
        self.measure()
        self.assertEqual(len(self.calls), 6)

    def test_arguments_only_do_not_prove_quantized_execution(self):
        def unverified(*args):
            result = self.runner(*args)
            result["execution"].update(quantized_path=None, fallback=None)
            return result
        self.assertEqual(self.measure(unverified)["decision"]["decision"], "pending")
        runtime.close_project(self.close_plan())
        self.assertFalse(self.plan["reference_path"].exists())

    def test_request_failure_cannot_accept(self):
        def failed(*args):
            result = self.runner(*args)
            metrics = result["raw"]["metrics"]
            metrics.update(failed_requests=1, successful_requests=2)
            return result
        self.assertEqual(self.measure(failed)["decision"]["decision"], "pending")

    def test_runtime_drift_rejected(self):
        def drift(*args):
            result = self.runner(*args)
            if args[-1] == "candidate":
                result["backend_version"] = "other"
            return result
        self.assertEqual(self.measure(drift)["decision"]["decision"], "reject")

    def test_interruption_after_complete_member_can_resume(self):
        original = gate._gate_save_or_verify
        def interrupt(path, payload):
            if path == runtime.member_path(self.plan, "pair2_candidate"):
                raise OSError("synthetic interrupted publish")
            return original(path, payload)
        with patch.object(gate, "_gate_save_or_verify", side_effect=interrupt), self.assertRaises(OSError):
            self.measure()
        self.assertEqual(self.calls.count("pair1_baseline"), 1)
        self.measure()
        self.assertEqual(self.calls.count("pair1_baseline"), 1)

    def test_close_resumes_missing_handoff_and_receipt(self):
        self.measure()
        original = gate._gate_save_or_verify
        def interrupt(path, payload):
            if path == self.plan["reference_path"]:
                raise OSError("synthetic close interruption")
            return original(path, payload)
        with patch.object(gate, "_gate_save_or_verify", side_effect=interrupt), self.assertRaises(OSError):
            runtime.close_project(self.close_plan())
        runtime.close_project(self.close_plan())
        self.assertTrue(self.plan["receipt_path"].exists())

    def test_source_and_saved_member_hashes_are_checked(self):
        self.measure()
        member = runtime.member_path(self.plan, "pair1_baseline")
        member.write_text(member.read_text() + "\n")
        with self.assertRaises(ValueError):
            runtime.read_project(self.plan)

    def test_upstream_hash_tamper_is_structured(self):
        ref = gate._gate_read_json(self.plan["source_path"])
        self.fixture.write(self.plan["source_path"], {**ref, "sha256": "0" * 64})
        self.assertFalse(runtime.prepare_project(self.plan)["ready"])

    def test_all_three_profiles_required_before_accept(self):
        self.plan = self.serve_plan()
        def partial(plan, prepared, key, work, role):
            if key == "throughput":
                return {"status": "failed", "failure": {"reason": "synthetic failure"}}
            return self.runner(plan, prepared, key, work, role)
        result = self.measure(partial)
        self.assertEqual(result["decision"]["decision"], "pending")
        runtime.close_project(self.close_plan())
        self.assertFalse(self.plan["reference_path"].exists())

    def test_serving_accept_is_per_profile_and_no_ancestor_load(self):
        self.plan = self.serve_plan()
        # The direct deployment carries its gate lineage/manifest; 83 does not reopen 82.
        gate_plan = gate.artifact_gate_plan(self.root, "run", "producer")
        gate_plan["report_path"].unlink()
        result = self.measure()
        self.assertEqual(result["report"]["profile_decisions"], {"latency": "accept", "balanced": "accept", "throughput": "accept"})
        runtime.close_project(self.close_plan())

    def test_serving_nonfinite_metrics_never_pass_slo(self):
        profile = {"name": "latency", "workload": {"id": "x", "num_prompts": 1, "concurrency": 1, "max_tokens": 1, "repeats": 1},
                   "primary_metric": "ttft_p95_ms", "slo": {"ttft_p95_ms": {"direction": "max", "limit": 500.}}}
        row = {"name": "x", "status": "ok", "artifact_id": "x", "profile_name": "latency", "workload_id": "x", "evidence_level": "real_project_benchmark"}
        for value in (float("nan"), float("inf"), True, -1.):
            result = gate.select_serving_candidate([{**row, "metrics": {"ttft_p95_ms": value}}], profile, expected_artifact_id="x")
            self.assertNotEqual(result["decision"], "accept")
        self.assertFalse(gate.validate_serving_profile({**profile, "workload": {**profile["workload"], "repeats": True}})["ready"])

    def test_incomplete_request_counts_are_pending(self):
        self.plan = self.serve_plan()
        def incomplete(*args):
            row = self.runner(*args)
            row["raw"]["runs"][0]["workload"]["requests"] = 0
            return row
        self.assertEqual(self.measure(incomplete)["decision"]["decision"], "pending")

    def test_full_member_matrix_cannot_be_forged_by_partial_summary(self):
        self.measure()
        raw = gate._gate_read_json(self.plan["raw_path"])
        raw["members"].pop("pair3_candidate")
        self.fixture.write(self.plan["raw_path"], raw)
        with self.assertRaises(ValueError):
            runtime.read_project(self.plan)

    def test_notebook_measure_cold_hot_and_close_reuses_saved_members(self):
        repo = Path(__file__).resolve().parents[1]
        notebook = json.loads((repo / "02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb").read_text())
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code" and any(t in c.get("metadata", {}).get("tags", []) for t in
                 ("project-preflight", "project-contract", "project-experiment", "project-evidence", "project-decision"))]
        with redirect_stdout(StringIO()), patch("tools.project_runtime.resolve_project_root", return_value=self.root), patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout='{"ready": true}')) as probe:
            for stage in ("measure", "measure", "close", "close"):
                namespace = {"RUN_LABEL": "deploy", "RUN_STAGE": stage, "UPSTREAM_RUN_LABEL": "run"}
                exec("".join(cells[0]["source"]), namespace)
                actual_measure = runtime.measure_project
                with patch.object(runtime, "measure_project", side_effect=lambda p: actual_measure(p, self.runner)):
                    for cell in cells[1:]:
                        exec("".join(cell["source"]), namespace)
            self.assertEqual(probe.call_count, 1)
            self.assertEqual(len(self.calls), 6)

    def test_notebook_prepare_and_close_cold_without_cuda(self):
        repo = Path(__file__).resolve().parents[1]
        for project, name in (("67", "67_Quantized_Inference_and_Deployment"), ("83", "83_Quantized_Serving_Profile_Benchmark")):
            if project == "83":
                self.plan = self.serve_plan()
            notebook = json.loads((repo / "02_PyTorch_Algorithms" / (name + ".ipynb")).read_text())
            for stage in ("prepare", "close"):
                namespace = {"RUN_LABEL": self.plan["run_label"], "RUN_STAGE": stage, "UPSTREAM_RUN_LABEL": self.plan["upstream_run_label"]}
                with patch("tools.project_runtime.resolve_project_root", return_value=self.root), patch("subprocess.run", side_effect=AssertionError("no process")), redirect_stdout(StringIO()):
                    for cell in notebook["cells"]:
                        if cell["cell_type"] == "code" and any(t in cell.get("metadata", {}).get("tags", []) for t in ("project-preflight", "project-contract", "project-experiment", "project-evidence", "project-decision")):
                            exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
