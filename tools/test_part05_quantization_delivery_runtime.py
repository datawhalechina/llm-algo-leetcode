"""CPU-only contract tests for the staged Part 05 quantization project."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import yaml

from tools.part05_quantization_delivery_runtime import (
    SEMANTIC_ID,
    decide_quantization_delivery,
    hash_tree,
    quantize_and_evaluate,
    run_service_pair,
    validate_protocol,
    sha256_file,
    save_once,
    _result_dir,
    prepare_quantization_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load(
    (ROOT / "05_Integrated_Projects_and_Decisions/04_Quantization_Delivery/project_config.example.yaml").read_text(encoding="utf-8")
)


class TextRows(list):
    """Minimal local stand-in for dataset filtering and indexed selection."""

    def filter(self, predicate):
        return TextRows(row for row in self if predicate(row))

    def select(self, indices):
        return TextRows(self[index] for index in indices)


class QuantizationDeliveryContractTests(unittest.TestCase):
    def test_registered_protocol_is_valid(self) -> None:
        self.assertEqual(validate_protocol(PROTOCOL), [])

    def test_rejects_unpinned_model(self) -> None:
        changed = {**PROTOCOL, "model": {**PROTOCOL["model"], "revision": "main"}}
        self.assertIn("model.id and immutable 40-character revision are required", validate_protocol(changed))

    def _write(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _case(self, *, nll_ratio: float = 1.05, runtime_seen: bool = True) -> tuple[Path, str]:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        label = "unit_quant_delivery"
        run_dir = root / PROTOCOL["outputs"]["directory"] / label
        artifact_dir = run_dir / "artifacts/gptq_w4a16_g128"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "weights.safetensors").write_bytes(b"fixture")
        hashes = hash_tree(artifact_dir)
        workload_target = root / PROTOCOL["serving"]["workload"]
        workload_target.parent.mkdir(parents=True, exist_ok=True)
        workload_target.write_text(
            (ROOT / PROTOCOL["serving"]["workload"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        base_dir = root / "base-model"
        base_dir.mkdir()
        (base_dir / "weights.safetensors").write_bytes(b"base-fixture")
        inputs = {
            "schema_version": "part05-quantization-inputs/v1", "semantic_id": SEMANTIC_ID,
            "run_label": label, "protocol": deepcopy(PROTOCOL),
            "model": {**PROTOCOL["model"], "path": str(base_dir)},
            "model_sha256": hash_tree(base_dir), "dataset": {"id": "fixture"},
            "quantization": {key: PROTOCOL["quantization"][key] for key in ("method", "format", "bits", "group_size", "max_sequence_length")},
        }
        self._write(run_dir / "inputs.json", inputs)
        artifact = {
            "schema_version": "part05-quantization-artifact/v1",
            "protocol": deepcopy(PROTOCOL),
            "semantic_id": SEMANTIC_ID,
            "run_label": label,
            "artifact_path": str(artifact_dir.relative_to(root)),
            "artifact_sha256": hashes,
            "model": inputs["model"], "dataset": inputs["dataset"], "quantization": inputs["quantization"],
            "inputs_sha256": sha256_file(run_dir / "inputs.json"),
            "quality": {"nll_ratio_vs_bf16": nll_ratio, "bf16_evaluation_nll": 1.0, "quantized_evaluation_nll": nll_ratio},
        }
        self._write(run_dir / "quantization.json", artifact)
        workload_path = root / PROTOCOL["serving"]["workload"]
        self._write(run_dir / "service_pair.json", {
            "schema_version": "part05-quantization-service-pair/v1",
            "semantic_id": SEMANTIC_ID,
            "run_label": label,
            "protocol": deepcopy(PROTOCOL),
            "artifact_manifest": str((run_dir / "quantization.json").relative_to(root)),
            "artifact_sha256": hashes,
            "workload_sha256": sha256_file(workload_path),
            "members": {
                "bf16": str((run_dir / "service_bf16.json").relative_to(root)),
                "gptq": str((run_dir / "service_gptq.json").relative_to(root)),
            },
            "status": "complete",
        })
        ids = [json.loads(line)["request_id"] for line in (ROOT / PROTOCOL["serving"]["workload"]).read_text().splitlines()]
        results = [{"request_id": request_id} for _ in range(PROTOCOL["serving"]["repeats"]) for request_id in ids]
        report_base = {
            "workload": {
                "requests": PROTOCOL["serving"]["num_prompts"],
                **{key: PROTOCOL["serving"][key] for key in ("max_tokens", "concurrency", "warmup", "repeats")},
            },
            "normalized_result": {"metrics": {
                "ttft_ms": {"p95": 50}, "tpot_ms": {"p95": 10},
                "throughput_tokens_per_s": 20, "peak_memory_mb": 3000,
            }},
            "metrics": {"complete_correct_rate": 0.9, "successful_requests": 20, "failed_requests": 0},
            "results": results,
        }
        self._write(run_dir / "service_bf16.json", {
            **report_base,
            "run_label": f"{label}_bf16",
            "label": f"{label}_bf16",
            "service_execution": {
                "requested_model_path": str(base_dir), "resolved_model_path": str(base_dir),
                "protocol": deepcopy(PROTOCOL), "workload_sha256": sha256_file(workload_path),
            },
        })
        self._write(run_dir / "service_gptq.json", {
            **report_base,
            "run_label": f"{label}_gptq",
            "label": f"{label}_gptq",
            "service_execution": {
                "requested_model_path": str(artifact_dir),
                "resolved_model_path": str(artifact_dir),
                "quantization_runtime_observed": runtime_seen,
                "protocol": deepcopy(PROTOCOL), "workload_sha256": sha256_file(workload_path),
            },
        })
        pair_path = run_dir / "service_pair.json"
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        pair["member_sha256"] = {role: sha256_file(run_dir / f"service_{role}.json") for role in ("bf16", "gptq")}
        self._write(pair_path, pair)
        return root, label

    def test_paired_valid_evidence_can_be_accepted(self) -> None:
        root, label = self._case()
        result = decide_quantization_delivery(root, label, PROTOCOL)
        self.assertEqual(result["decision"], "accept")

    def test_missing_quantized_runtime_evidence_is_rejected(self) -> None:
        root, label = self._case(runtime_seen=False)
        result = decide_quantization_delivery(root, label, PROTOCOL)
        self.assertEqual(result["decision"], "reject")

    def test_quality_guardrail_rejects_candidate(self) -> None:
        root, label = self._case(nll_ratio=1.2)
        result = decide_quantization_delivery(root, label, PROTOCOL)
        self.assertEqual(result["decision"], "reject")

    def test_completed_quantization_artifact_is_reused(self) -> None:
        root, label = self._case()
        run_dir = root / PROTOCOL["outputs"]["directory"] / label
        result = quantize_and_evaluate(
            torch=None, auto_model_class=None, tokenizer_class=None, load_dataset=None,
            protocol=PROTOCOL, project_root=root, run_label=label,
        )
        self.assertEqual(result["run_label"], label)

    def test_completed_service_pair_is_reused_without_relaunch(self) -> None:
        root, label = self._case()
        run_dir = root / PROTOCOL["outputs"]["directory"] / label
        for role, model_path in (("bf16", str(root / "base-model")), ("gptq", str(run_dir / "artifacts/gptq_w4a16_g128"))):
            report_path = run_dir / f"service_{role}.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["run_label"] = f"{label}_{role}"
            report["service_execution"]["requested_model_path"] = model_path
            self._write(report_path, report)

        def should_not_launch(**kwargs):
            raise AssertionError("完整服务证据不应重新启动后端")

        result = run_service_pair(
            protocol=PROTOCOL, project_root=root, run_label=label,
            start_server=should_not_launch, stop_server=should_not_launch,
            run_benchmark=should_not_launch,
        )
        self.assertEqual(result["status"], "complete")

    def _change_report(self, root, label, mutate, *, refresh_hash=True):
        run_dir = root / PROTOCOL["outputs"]["directory"] / label
        path = run_dir / "service_gptq.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        mutate(report)
        self._write(path, report)
        if refresh_hash:
            pair_path = run_dir / "service_pair.json"
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            pair["member_sha256"]["gptq"] = sha256_file(path)
            self._write(pair_path, pair)

    def test_invalid_configuration_is_rejected(self):
        for section, key, value in (
            ("model", "revision", "z" * 40), ("serving", "warmup", -1),
            ("serving", "repeats", True), ("serving", "minimum_answer_accuracy", float("nan")),
            ("serving", "maximum_ttft_p95_ms", float("inf")),
            ("serving", "maximum_answer_accuracy_regression", 2),
            ("quantization", "evaluation_split", "train"),
            ("quantization", "calibration_samples", 0),
        ):
            with self.subTest(section=section, key=key, value=value):
                protocol = deepcopy(PROTOCOL)
                protocol[section][key] = value
                self.assertTrue(validate_protocol(protocol))
        self.assertTrue(validate_protocol(None))
        self.assertTrue(validate_protocol({"serving": []}))

    def test_invalid_labels_cannot_escape_result_directory(self):
        for label in (None, "", "../other", "/tmp/other", "replace_with_unique_label"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                _result_dir(ROOT, label)

    def test_bad_accuracy_remains_pending(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "invalid"):
            with self.subTest(value=value):
                root, label = self._case()
                self._change_report(root, label, lambda row: row["metrics"].update(complete_correct_rate=value))
                self.assertEqual(decide_quantization_delivery(root, label, PROTOCOL)["decision"], "pending")

    def test_bad_service_metrics_remain_pending(self):
        for value in (float("nan"), -1, True, "invalid"):
            with self.subTest(value=value):
                root, label = self._case()
                self._change_report(root, label, lambda row: row["normalized_result"]["metrics"].update(ttft_ms={"p95": value}))
                self.assertEqual(decide_quantization_delivery(root, label, PROTOCOL)["decision"], "pending")

    def test_invalid_nll_cannot_be_accepted(self):
        for ratio in (float("nan"), float("inf"), -1, "invalid"):
            with self.subTest(ratio=ratio):
                root, label = self._case(nll_ratio=ratio)
                self.assertEqual(decide_quantization_delivery(root, label, PROTOCOL)["decision"], "pending")

    def test_report_tampering_breaks_pair_hash(self):
        root, label = self._case()
        self._change_report(root, label, lambda row: row["metrics"].update(complete_correct_rate=1), refresh_hash=False)
        result = decide_quantization_delivery(root, label, PROTOCOL)
        self.assertEqual(result["decision"], "pending")
        self.assertFalse(result["gates"]["artifact_hash_match"])

    def test_changed_measurement_budget_is_not_comparable(self):
        root, label = self._case()
        self._change_report(root, label, lambda row: row["workload"].update(max_tokens=1))
        self.assertEqual(decide_quantization_delivery(root, label, PROTOCOL)["decision"], "pending")

    def test_protocol_change_prevents_service_reuse(self):
        root, label = self._case()
        protocol = deepcopy(PROTOCOL)
        protocol["serving"]["max_tokens"] += 1
        launch = Mock(side_effect=AssertionError("不得启动服务"))
        with self.assertRaises(ValueError):
            run_service_pair(protocol=protocol, project_root=root, run_label=label,
                             start_server=launch, stop_server=launch, run_benchmark=launch)
        launch.assert_not_called()

    def test_new_pair_accepts_real_client_label_field(self):
        root, label = self._case()
        run_dir = root / PROTOCOL["outputs"]["directory"] / label
        reports = {}
        for role in ("bf16", "gptq"):
            path = run_dir / f"service_{role}.json"
            reports[role] = json.loads(path.read_text(encoding="utf-8"))
            reports[role].pop("run_label")
            path.unlink()
        (run_dir / "service_pair.json").unlink()
        def start(**kwargs):
            return object(), None, 12345, "bfloat16", kwargs["model_id"]
        stop = Mock()
        def benchmark(**kwargs):
            return deepcopy(reports["bf16" if kwargs["role"] == "baseline" else "gptq"])
        pair = run_service_pair(protocol=PROTOCOL, project_root=root, run_label=label,
                                start_server=start, stop_server=stop, run_benchmark=benchmark)
        self.assertEqual(stop.call_count, 2)
        self.assertEqual(pair["status"], "complete")
        for role in ("bf16", "gptq"):
            row = json.loads((run_dir / f"service_{role}.json").read_text(encoding="utf-8"))
            self.assertEqual(row["run_label"], f"{label}_{role}")

    def test_save_once_is_order_independent_and_protects_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_once(path, {"a": 1, "b": 2})
            save_once(path, {"b": 2, "a": 1})
            with self.assertRaises(FileExistsError):
                save_once(path, {"a": 2})
            with self.assertRaises(ValueError):
                save_once(Path(directory) / "invalid.json", {"a": float("nan")})

    def _prepared_case(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        model = root / "base-model"
        model.mkdir()
        (model / "weights.safetensors").write_bytes(b"base-fixture")
        runtime = Mock()
        runtime.resolve_model_snapshot.return_value = (model, "local_fixture")
        def dataset(*args, **kwargs):
            return TextRows({"text": f"{kwargs['split']} text {index}"} for index in range(64))
        label = "prepared_quant_test"
        path = prepare_quantization_inputs(protocol=PROTOCOL, project_root=root, run_label=label,
                                            model_runtime=runtime, load_dataset=dataset)
        return root, label, path, runtime, dataset

    def test_prepare_records_actual_files_and_texts_without_network(self):
        root, label, path, runtime, dataset = self._prepared_case()
        original = path.read_bytes()
        record = json.loads(original)
        self.assertEqual(record["model_sha256"], hash_tree(root / "base-model"))
        self.assertNotEqual(record["dataset_sha256"]["calibration"], record["dataset_sha256"]["evaluation"])
        self.assertEqual(prepare_quantization_inputs(protocol=PROTOCOL, project_root=root, run_label=label,
                                                    model_runtime=runtime, load_dataset=dataset), path)
        self.assertEqual(path.read_bytes(), original)

    def test_changed_texts_stop_before_loading_model(self):
        root, label, path, runtime, dataset = self._prepared_case()
        changed = lambda *args, **kwargs: TextRows({"text": f"changed {index}"} for index in range(64))
        model_class = Mock()
        with self.assertRaisesRegex(ValueError, "文本与准备清单不一致"):
            quantize_and_evaluate(torch=None, auto_model_class=model_class, tokenizer_class=Mock(),
                                 load_dataset=changed, protocol=PROTOCOL, project_root=root, run_label=label)
        model_class.from_pretrained.assert_not_called()

    def test_changed_base_files_stop_before_loading_tokenizer(self):
        root, label, path, runtime, dataset = self._prepared_case()
        (root / "base-model/weights.safetensors").write_bytes(b"changed")
        tokenizer = Mock()
        with self.assertRaisesRegex(ValueError, "基础模型文件"):
            quantize_and_evaluate(torch=None, auto_model_class=Mock(), tokenizer_class=tokenizer,
                                 load_dataset=dataset, protocol=PROTOCOL, project_root=root, run_label=label)
        tokenizer.from_pretrained.assert_not_called()

    def test_changed_input_manifest_prevents_closeout(self):
        root, label = self._case()
        path = root / PROTOCOL["outputs"]["directory"] / label / "inputs.json"
        inputs = json.loads(path.read_text(encoding="utf-8"))
        inputs["dataset"]["id"] = "changed"
        self._write(path, inputs)
        with self.assertRaisesRegex(ValueError, "准备清单的 SHA256"):
            decide_quantization_delivery(root, label, PROTOCOL)

    def test_missing_quantized_files_prevent_acceptance(self):
        root, label = self._case()
        path = root / PROTOCOL["outputs"]["directory"] / label / "artifacts/gptq_w4a16_g128/weights.safetensors"
        path.unlink()
        result = decide_quantization_delivery(root, label, PROTOCOL)
        self.assertEqual(result["decision"], "pending")

    def test_interrupted_service_log_is_not_overwritten(self):
        root, label = self._case()
        run_dir = root / PROTOCOL["outputs"]["directory"] / label
        (run_dir / "service_bf16.json").unlink()
        log = run_dir / "vllm_bf16.log"
        log.write_text("interrupted evidence", encoding="utf-8")
        launch = Mock(side_effect=AssertionError("不得启动服务"))
        with self.assertRaises(FileExistsError):
            run_service_pair(protocol=PROTOCOL, project_root=root, run_label=label,
                             start_server=launch, stop_server=launch, run_benchmark=launch)
        launch.assert_not_called()
        self.assertEqual(log.read_text(encoding="utf-8"), "interrupted evidence")


if __name__ == "__main__":
    unittest.main()
