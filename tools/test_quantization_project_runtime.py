"""CPU-only regression checks for quantization project evidence lifecycles."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import quantization_project_runtime as runtime
from tools.quantization_runtime import build_candidate_bundle, save_candidate_bundle


class ArtifactEvidenceLifecycleTests(unittest.TestCase):
    """All purported loader observations below are synthetic, confined to temporary directories."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = runtime.artifact_gate_plan(self.root, "run", "producer", "measure")
        self.artifact = self.root / "benchmarks/artifacts/40_gptq_awq/producer/gptq"
        self.artifact.mkdir(parents=True)
        for name, text in (("config.json", "{}"), ("model.safetensors", "fixture-weights"), ("tokenizer.json", "{}")):
            (self.artifact / name).write_text(text)
        self.source = {
            "schema_version": "quantization-mechanism-gpu/v5", "chapter": "40", "run_mode": "real_gpu",
            "run_label": "producer", "decision": "accept_weight_ptq_candidate", "failure": None,
            "mechanism_metrics": {"selected_candidate": "gptq"},
            "workload": {"model_id": "test/model", "resolved_model_commit": "abc123", "scheme": "W4A16", "max_nll_ratio_vs_bf16": 1.1},
            "dataset": {"dataset_id": "test/data", "revision": "data1", "config": "raw",
                        "calibration_split": "train", "evaluation_split": "validation", "calibration_samples": 4, "evaluation_samples": 4,
                        "calibration_ids_hash": "cal", "evaluation_ids_hash": "eval"},
            "baseline": {"token_weighted_nll": 2.0},
            "candidates": {"gptq": {"token_weighted_nll": 2.1, "nll_ratio_vs_bf16": 1.05,
                "artifact_manifest": {"quantization_config": {"format": "pack-quantized", "quant_method": "compressed-tensors", "version": "test",
                    "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 128}}}}}}},
            "evidence_level": "artifact_generation",
        }
        self.write(self.plan["source_path"], self.source)
        bundle = build_candidate_bundle(self.source, candidate_name="gptq", artifact_dir=self.artifact,
                                        project_root=self.root, source_result_path=self.plan["source_path"])
        save_candidate_bundle(self.artifact / "candidate_bundle.json", bundle)

    @staticmethod
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def prepared(self):
        prepared = runtime.prepare_artifact_gate(self.plan)
        self.assertTrue(prepared["ready"], prepared["issues"])
        return prepared

    def loader(self, failed=False):
        return {"schema_version": "quantization-artifact-loader/v2", "backend": "vllm",
                "invocation": runtime.artifact_loader_invocation(self.plan, self.prepared()),
                "status": "failed" if failed else "ok", "load_status": "failed" if failed else "loaded",
                "backend_version": "cpu-fixture-not-a-real-run", "hardware": {"cuda_available": True, "device": "synthetic"},
                "roundtrip_quality": {"prompt_count": 2, "non_empty_generations": 0 if failed else 2},
                "peak_memory_mb": None if failed else 10.0, "memory_measurement": "device_free_memory_delta_not_peak",
                "kernel": "not_verified_by_loader_gate", "fallback": "not_verified_by_loader_gate",
                "failure": {"reason": "synthetic failure"} if failed else None}

    def measure(self, failed=False):
        return runtime.save_artifact_gate_measurement(self.plan, self.prepared(), self.loader(failed))

    def closing_plan(self):
        return runtime.artifact_gate_plan(self.root, "run", "producer", "close")

    def test_prepare_is_read_only(self):
        before = set(self.root.rglob("*"))
        with patch("subprocess.run", side_effect=AssertionError("prepare cannot launch a process")):
            self.prepared()
        self.assertEqual(before, set(self.root.rglob("*")))
        self.assertFalse(self.plan["raw_path"].exists())

    def test_bad_input_json_and_source_identity_are_structured(self):
        self.plan["source_path"].write_text("invalid-json")
        self.assertFalse(runtime.prepare_artifact_gate(self.plan)["ready"])
        self.write(self.plan["source_path"], {**self.source, "run_label": "other"})
        self.assertFalse(runtime.prepare_artifact_gate(self.plan)["ready"])

    def test_bundle_quality_and_lineage_must_match_source(self):
        path = self.artifact / "candidate_bundle.json"
        bundle = json.loads(path.read_text())
        for section, key, value in (("evaluation", "candidate_value", 2.0), ("calibration", "sample_count", 0),
                                    ("evidence", "source_result_path", "another.json")):
            bad = deepcopy(bundle)
            bad[section][key] = value
            self.write(path, bad)
            self.assertFalse(runtime.prepare_artifact_gate(self.plan)["ready"])
        self.write(path, bundle)
        self.prepared()

    def test_measure_has_no_final_assets_and_can_repeat(self):
        self.assertEqual(self.measure()["status"], "measured")
        before = {name: self.plan[name].stat().st_mtime_ns for name in ("loader_path", "raw_path")}
        self.assertEqual(self.measure()["status"], "already_measured")
        for name, stamp in before.items():
            self.assertEqual(self.plan[name].stat().st_mtime_ns, stamp)
        for name in ("report_path", "reference_path", "receipt_path"):
            self.assertFalse(self.plan[name].exists())

    def test_loader_only_interruption_recovers_without_execution(self):
        prepared, loader = self.prepared(), self.loader()
        original = runtime._gate_save_or_verify
        def interrupt(path, payload):
            if path == self.plan["raw_path"]:
                raise OSError("interrupted after loader publish")
            return original(path, payload)
        with patch.object(runtime, "_gate_save_or_verify", side_effect=interrupt):
            with self.assertRaises(OSError):
                runtime.save_artifact_gate_measurement(self.plan, prepared, loader)
        self.assertEqual(runtime.read_artifact_gate(self.plan)["status"], "incomplete_measurement")
        with patch("subprocess.run", side_effect=AssertionError("recovery must not start GPU")):
            runtime.save_artifact_gate_measurement(self.plan, self.prepared(), json.loads(self.plan["loader_path"].read_text()))
        self.assertEqual(runtime.read_artifact_gate(self.plan)["status"], "validated")

    def test_read_is_read_only(self):
        self.measure()
        before = {p: p.stat().st_mtime_ns for p in self.plan["raw_path"].parent.iterdir()}
        runtime.read_artifact_gate(self.plan)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in self.plan["raw_path"].parent.iterdir()})

    def test_close_cold_and_idempotent(self):
        self.measure()
        with patch("subprocess.run", side_effect=AssertionError("close cannot start GPU")):
            result = runtime.close_artifact_gate(self.closing_plan())
        self.assertEqual(result["status"], "closed")
        receipt = result["registration_receipt"]
        self.assertEqual(receipt["decision"]["decision"], "accept")
        self.assertIn("handoff", receipt["files"])
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.plan["raw_path"].parent.iterdir()}
        self.assertEqual(runtime.close_artifact_gate(self.closing_plan())["status"], "already_complete")
        self.assertEqual(before, {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.plan["raw_path"].parent.iterdir()})

    def test_close_resumes_after_report_or_handoff(self):
        self.measure()
        original = runtime._gate_save_or_verify
        for stop_key in ("reference_path", "receipt_path"):
            def interrupt(path, payload):
                if path == self.plan[stop_key]:
                    raise OSError("interrupted before next asset")
                return original(path, payload)
            with patch.object(runtime, "_gate_save_or_verify", side_effect=interrupt):
                with self.assertRaises(OSError):
                    runtime.close_artifact_gate(self.closing_plan())
            self.assertTrue(self.plan["report_path"].exists())
        self.assertEqual(runtime.close_artifact_gate(self.closing_plan())["status"], "closed")

    def test_failure_closes_without_handoff(self):
        self.measure(failed=True)
        result = runtime.close_artifact_gate(self.closing_plan())
        self.assertEqual(result["decision"]["decision"], "reject")
        self.assertIsNone(result["handoff_path"])
        self.assertTrue(self.plan["receipt_path"].exists())
        self.assertFalse(self.plan["reference_path"].exists())

    def test_close_without_measurement_does_not_write(self):
        self.assertEqual(runtime.close_artifact_gate(self.closing_plan())["status"], "pending")
        self.assertFalse(self.plan["receipt_path"].parent.exists())

    def test_tampered_raw_identity_or_decision_is_rejected(self):
        self.measure()
        original = json.loads(self.plan["raw_path"].read_text())
        for key, value in (("run_label", "wrong"), ("decision", {"decision": "reject"})):
            self.write(self.plan["raw_path"], {**original, key: value})
            with self.assertRaises(ValueError):
                runtime.close_artifact_gate(self.closing_plan())
            self.assertFalse(self.plan["report_path"].exists())

    def test_conflicting_report_cannot_be_overwritten(self):
        self.measure()
        self.write(self.plan["report_path"], {"decision": "tampered"})
        before = self.plan["report_path"].read_bytes()
        with self.assertRaises(ValueError):
            runtime.close_artifact_gate(self.closing_plan())
        self.assertEqual(self.plan["report_path"].read_bytes(), before)

    def test_loader_invocation_and_legacy_schema_are_rejected(self):
        loader = self.loader()
        for field, value in (("invocation", {}), ("schema_version", "old")):
            with self.assertRaises(ValueError):
                runtime.save_artifact_gate_measurement(self.plan, self.prepared(), {**loader, field: value})
        self.assertFalse(self.plan["raw_path"].exists())

    def test_artifact_or_source_changed_after_measurement_is_rejected(self):
        self.measure()
        self.plan["source_path"].write_text(self.plan["source_path"].read_text() + "\n")
        with self.assertRaises(ValueError):
            runtime.read_artifact_gate(self.plan)
        (self.artifact / "model.safetensors").write_text("modified")
        self.assertFalse(runtime.prepare_artifact_gate(self.plan)["ready"])

    def test_handoff_and_receipt_hash_tampering_is_rejected(self):
        self.measure()
        runtime.close_artifact_gate(self.closing_plan())
        original = self.plan["reference_path"].read_text()
        reference = json.loads(original)
        self.write(self.plan["reference_path"], {**reference, "sha256": "0" * 64})
        with self.assertRaises(ValueError):
            runtime.read_artifact_gate(self.plan)
        self.plan["reference_path"].write_text(original)
        receipt = json.loads(self.plan["receipt_path"].read_text())
        receipt["files"]["loader"]["sha256"] = "0" * 64
        self.write(self.plan["receipt_path"], receipt)
        with self.assertRaises(ValueError):
            runtime.read_artifact_gate(self.plan)

    def test_stage_and_label_guards(self):
        for label in ("../escape", "/absolute", "", "with space"):
            with self.assertRaises(ValueError):
                runtime.artifact_gate_plan(self.root, label, "producer")
        with self.assertRaises(ValueError):
            runtime.artifact_gate_plan(self.root, "replace_with_unique_label", "producer", "measure")
        with self.assertRaises(ValueError):
            runtime.close_artifact_gate(self.plan)

    def test_notebook_cold_and_hot_stages_without_gpu(self):
        notebook = Path(__file__).resolve().parents[1] / "02_PyTorch_Algorithms/82_Quantization_Artifact_Evaluation_Project.ipynb"
        cells = {c["id"]: "".join(c["source"]) for c in json.loads(notebook.read_text())["cells"]}
        ids = ("82_step52_preflight", "82_step53_cpu_gate", "82_step54_backend", "82_step55_save", "82_step56_decision")
        calls = []
        def simulated_child(command, **kwargs):
            calls.append(command)
            if command[1].endswith("environment_preflight.py"):
                return SimpleNamespace(returncode=0, stdout='{"status":"ok","ready":true}', stderr="")
            invocation = json.loads(command[command.index("--invocation-json") + 1])
            loader = self.loader()
            self.assertEqual(loader["invocation"], invocation)
            self.write(self.plan["loader_path"], loader)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch("tools.project_runtime.resolve_project_root", return_value=self.root), patch("subprocess.run", side_effect=simulated_child):
            namespace = {"RUN_LABEL": "run", "UPSTREAM_RUN_LABEL": "producer", "RUN_STAGE": "prepare"}
            with redirect_stdout(StringIO()):
                for name in ids:
                    exec(cells[name], namespace)
            self.assertEqual(calls, [])
            namespace["RUN_STAGE"] = "measure"
            with redirect_stdout(StringIO()):
                for name in ids:
                    exec(cells[name], namespace)
            self.assertEqual(len(calls), 2)
            self.assertFalse(self.plan["reference_path"].exists())
            # A cold close does not rely on measure's variables or an active CUDA environment.
            closing = {"RUN_LABEL": "run", "UPSTREAM_RUN_LABEL": "producer", "RUN_STAGE": "close"}
            with redirect_stdout(StringIO()):
                for name in ids:
                    exec(cells[name], closing)
                for name in ids:
                    exec(cells[name], closing)
            self.assertEqual(closing["registration"]["status"], "already_complete")
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
