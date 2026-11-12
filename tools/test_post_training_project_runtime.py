"""Focused tests for shared post-training project fact collection."""

from __future__ import annotations

import tempfile
import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

from tools import post_training_project_runtime as runtime

from tools.post_training_project_runtime import (
    audit_preference_records,
    audit_sft_records,
    build_causal_sft_examples,
    build_preference_pair_sequences,
    extract_final_number,
    hash_directory,
    save_json_record,
)


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": list(range(len(text.split())))}


class _BoundarySensitiveTokenizer(_Tokenizer):
    """受控替身：整段分词可跨提示/回答边界合并，分段分词不会。"""

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        merged = text.replace("Response:\nanswer", "Response_answer")
        return {"input_ids": list(range(len(merged.split())))}


def test_sft_audit_matches_training_sequence() -> None:
    """同一手工模板下，审计的监督数应与实际训练标签一致。"""
    tokenizer = _BoundarySensitiveTokenizer()
    records = [
        {"instruction": "solve one", "input": "", "output": "answer one"},
        {"instruction": "solve two", "input": "context", "output": "answer two three"},
        {"instruction": "solve three", "input": "", "output": "answer"},
    ]
    for max_length in (10, 16, 32):
        audit = audit_sft_records(records, tokenizer, max_length=max_length, train_count=2)
        examples = build_causal_sft_examples(records, tokenizer, max_length=max_length)
        counts = [sum(label != -100 for label in row["labels"][1:]) for row in examples]
        assert audit["minimum_supervised_tokens"] == min(counts)
        assert audit["mean_supervised_tokens"] == sum(counts) / len(counts)
        assert all(
            label == -100
            for row in examples
            for label, mask in zip(row["labels"], row["attention_mask"])
            if mask == 0
        )
    prompt = "### Instruction:\nsolve one\n### Response:\n"
    segmented = tokenizer(prompt)["input_ids"] + tokenizer("answer one")["input_ids"]
    whole = tokenizer(prompt + "answer one")["input_ids"]
    assert len(segmented) != len(whole), "该替身必须能区分两种序列协议"
    empty = [records[0], {**records[1], "output": ""}, records[2]]
    audit = audit_sft_records(empty, tokenizer, max_length=32, train_count=2)
    assert audit["empty_response_count"] == 1
    assert audit["zero_supervision_count"] == 0, "EOS仍可能保留，空回答须单独阻断"


def _sft_gate_and_cells():
    """Load the notebook's visible production rules without executing collection."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    tree = ast.parse(cells["project-contract-64"])
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    namespace = {}
    exec(compile(functions, "sft_visible_gate", "exec"), namespace)
    return root, namespace["apply_data_admission_limits"], cells


def _sft_record_fixture():
    """Construct small saved facts with a test-only contract, without network access."""
    root, gate, _ = _sft_gate_and_cells()
    contract = json.loads((root / "benchmarks/contracts/sft_data_admission_v2.json").read_text())
    contract["workload"].update(sample_count=4, train_count=2, max_length=32)
    contract["limits"]["minimum_evaluation_count"] = 2
    rows = [{"instruction": f"solve {index}", "input": "", "output": f"answer {index}"} for index in range(4)]
    audit = audit_sft_records(rows, _Tokenizer(), max_length=32, train_count=2)
    ids = [runtime.stable_text_hash((row["instruction"], "", row["output"])) for row in rows]
    record = {key: copy.deepcopy(contract[key]) for key in (
        "schema_version", "semantic_id", "measurement_version", "environment_profile",
        "rule_version", "sequence_protocol", "limits", "workload", "dataset", "tokenizer",
    )}
    record.update({"run_label": "fixture_v2", "run_mode": "real_data", "source_revision": "a" * 40,
                   "contract_sha256": hashlib.sha256(runtime.json_record_bytes(contract)).hexdigest(),
                   "raw_audit": copy.deepcopy(audit), "raw_admission": gate(audit, contract["limits"]),
                   "audit": audit, "admission": gate(audit, contract["limits"]),
                   "cleaning": {"rule": "drop_empty_instruction_response_and_exact_duplicates",
                                "removed_count": 0, "retained_ratio": 1.0},
                   "quality": {"independent_evaluation": True}, "decision": "accept",
                   "snapshot": {}})
    record["decision_details"] = copy.deepcopy(record["admission"])
    record["dataset"]["resolved_commit"] = "a" * 40
    record["tokenizer"]["resolved_commit"] = "b" * 40
    for split, indices in (("train", [0, 1]), ("evaluation", [2, 3])):
        record["snapshot"][f"{split}_record_indices"] = indices
        record["snapshot"][f"{split}_sample_ids"] = [ids[index] for index in indices]
        record["snapshot"][f"{split}_sample_ids_hash"] = runtime.stable_text_hash([ids[index] for index in indices])
    return contract, record, gate


def test_sft_registration_recovery() -> None:
    """Cover independent checks, blocked handoffs, conflicts and interrupted close."""
    contract, record, gate = _sft_record_fixture()
    runtime.validate_sft_admission_record(record, contract, "fixture_v2", gate)
    mutations = [
        lambda r: r.update(run_label="wrong"),
        lambda r: r.update(measurement_version="old"),
        lambda r: r.update(sequence_protocol="whole_chat"),
        lambda r: r.update(contract_sha256="0" * 64),
        lambda r: r["tokenizer"].update(resolved_commit="main"),
        lambda r: r["workload"].update(seed=0),
        lambda r: r["audit"].update(duplicate_ratio=float("nan")),
        lambda r: r["audit"].update(sample_count=True),
        lambda r: r["audit"].update(minimum_supervised_tokens=0),
        lambda r: r["audit"].update(train_prompt_ids_hash="unknown"),
        lambda r: r["audit"].update(truncation_ratio=0.5),
        lambda r: r["snapshot"]["evaluation_record_indices"].__setitem__(0, 0),
        lambda r: r["snapshot"]["evaluation_sample_ids"].pop(),
        lambda r: r["snapshot"].update(train_sample_ids_hash="0" * 64),
        lambda r: r["admission"].update(soft_failures=["fake"]),
    ]
    for mutate in mutations:
        broken = copy.deepcopy(record)
        mutate(broken)
        try:
            runtime.validate_sft_admission_record(broken, contract, "fixture_v2", gate)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered evidence was accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result, manifest, receipt = (root / name for name in ("raw.json", "manifest.json", "receipt.json"))
        runtime.publish_json_record(result, record)
        args = (root, result, manifest, receipt, contract, "fixture_v2", gate)
        actual_publish = runtime.publish_json_record

        def interrupt_receipt(path, payload):
            """Simulate interruption after publishing the handoff."""
            if path == receipt:
                raise RuntimeError("injected interruption")
            return actual_publish(path, payload)

        with patch.object(runtime, "publish_json_record", side_effect=interrupt_receipt):
            try:
                runtime.close_sft_admission(*args)
            except RuntimeError as error:
                assert str(error) == "injected interruption"
            else:
                raise AssertionError("interruption was not injected")
        assert manifest.exists() and not receipt.exists()
        assert runtime.close_sft_admission(*args)["status"] == "completed"
        mtimes = [path.stat().st_mtime_ns for path in (result, manifest, receipt)]
        assert runtime.close_sft_admission(*args)["status"] == "already_complete"
        assert mtimes == [path.stat().st_mtime_ns for path in (result, manifest, receipt)]
        manifest.unlink()
        assert runtime.close_sft_admission(*args)["status"] == "completed"
        manifest.write_text("{}")
        try:
            runtime.close_sft_admission(*args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("conflicting handoff was overwritten")
        assert manifest.read_text() == "{}"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result, manifest, receipt = (root / name for name in ("raw.json", "manifest.json", "receipt.json"))
        runtime.publish_json_record(result, record)
        runtime.publish_json_record(receipt, {"conflicting": True})
        try:
            runtime.close_sft_admission(root, result, manifest, receipt, contract, "fixture_v2", gate)
        except FileExistsError:
            pass
        else:
            raise AssertionError("conflicting receipt was accepted")
        assert not manifest.exists(), "existing conflicts must be checked before publishing handoff"
    for risk, expected in (("empty_response_count", "reject"), ("truncated_count", "tune")):
        blocked = copy.deepcopy(record)
        blocked["audit"][risk] = 1
        if risk == "truncated_count":
            blocked["audit"]["truncation_ratio"] = 0.25
        blocked["admission"] = gate(blocked["audit"], contract["limits"])
        blocked["decision_details"] = copy.deepcopy(blocked["admission"])
        blocked["decision"] = expected
        blocked["quality"]["independent_evaluation"] = blocked["admission"]["hard_gate_passed"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, handoff, receipt = (root / name for name in ("raw.json", "handoff.json", "receipt.json"))
            runtime.publish_json_record(raw, blocked)
            output = runtime.close_sft_admission(root, raw, handoff, receipt, contract, "fixture_v2", gate)
            assert not handoff.exists()
            assert output["registration_receipt"]["decision"] == expected
            assert output["registration_receipt"]["failed_checks"]
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "atomic.json"
        with patch.object(runtime.os, "link", side_effect=OSError("injected publication failure")):
            try:
                runtime.publish_json_record(output, {"ok": True})
            except OSError:
                pass
            else:
                raise AssertionError("publication failure was not injected")
        assert not output.exists() and not list(Path(directory).iterdir())


def test_sft_prepare_and_gate_isolation() -> None:
    """Run all six default layers with no collection dependencies or publication."""
    _, _, cells = _sft_gate_and_cells()
    namespace = {}
    for cell_id in ("project-config-64", "project-preflight-64", "project-contract-64",
                    "project-run-64", "project-evidence-64", "project-decision-64"):
        exec(compile(cells[cell_id], cell_id, "exec"), namespace)
    assert namespace["RUN_STAGE"] == "prepare"
    assert namespace["RUN_LABEL"] == "replace_with_unique_label"


def test_sft_collection_and_cold_close() -> None:
    """Exercise notebook collection and cold/hot close with injected local sources."""
    _, _, cells = _sft_gate_and_cells()
    contract, _, _ = _sft_record_fixture()
    rows = [{"instruction": f"solve {index}", "input": "", "output": f"answer {index}"} for index in range(4)]
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: rows
    hub = types.ModuleType("huggingface_hub")

    class LocalApi:
        """Return fixed test commits; never contact a remote repository."""

        def dataset_info(self, *args, **kwargs):
            return types.SimpleNamespace(sha="a" * 40)

        def model_info(self, *args, **kwargs):
            return types.SimpleNamespace(sha="b" * 40)

    hub.HfApi = LocalApi
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: _Tokenizer())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry_path = root / "benchmarks/project_registry/part02.json"
        contract_path = root / "benchmarks/contracts/sft_data_admission_v2.json"
        runtime.publish_json_record(registry_path, {"projects": [{
            "semantic_id": "PT-SFT-DATA-QUALITY",
            "default_experiment_contract": str(contract_path.relative_to(root)),
        }]})
        runtime.publish_json_record(contract_path, contract)
        namespace = {"RUN_STAGE": "collect", "RUN_LABEL": "fixture_v2"}
        with patch("tools.project_runtime.bootstrap_project_root", return_value=root):
            exec(cells["project-preflight-64"], namespace)
        exec(cells["project-contract-64"], namespace)
        with patch.dict(sys.modules, {"datasets": datasets, "transformers": transformers, "huggingface_hub": hub}):
            exec(cells["project-run-64"], namespace)
        exec(cells["project-evidence-64"], namespace)
        exec(cells["project-decision-64"], namespace)
        assert namespace["RESULT_PATH"].exists() and not namespace["MANIFEST_PATH"].exists()
        cold_namespace = {"RUN_STAGE": "close", "RUN_LABEL": "fixture_v2"}
        with patch("tools.project_runtime.bootstrap_project_root", return_value=root):
            for cell in ("project-preflight-64", "project-contract-64", "project-run-64",
                         "project-evidence-64", "project-decision-64"):
                exec(cells[cell], cold_namespace)
            assert cold_namespace["registration"]["status"] == "completed"
            for cell in ("project-preflight-64", "project-contract-64", "project-run-64",
                         "project-evidence-64", "project-decision-64"):
                exec(cells[cell], cold_namespace)
        assert cold_namespace["registration"]["status"] == "already_complete"
        manifest = json.loads(cold_namespace["MANIFEST_PATH"].read_text())
        assert manifest["decision"] == "accept"
        assert manifest["sample_ids_hash"] == manifest["audit"]["sample_ids_hash"]
        assert manifest["snapshot"]["train_record_indices"] and manifest["snapshot"]["evaluation_record_indices"]


def test_lora_scan_reader() -> None:
    """Read historical facts without training; reject invalid metrics and unstable repeats."""
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb").read_text())
    source = next("".join(c["source"]) for c in nb["cells"] if c["id"] == "project-contract-63")
    tree = ast.parse(source)
    namespace = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                            type_ignores=[]), "lora_reader", "exec"), namespace)
    record = json.loads((root / "benchmarks/results/63_lora_variants/lora_variant_sweep_lora_variant_selection_v2.json").read_text())
    choose = namespace["select_lora_candidate_for_run"]
    args = (record["gates"], ["rank4", "rank8", "rank16"], record["workload"]["run_seeds"])
    assert choose(record["candidates"], *args)["selected"]["name"] == "rank4"
    for value in (float("nan"), float("inf"), -1):
        rows = copy.deepcopy(record["candidates"])
        rows[0]["median_step_time_ms"] = value
        try:
            choose(rows, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid step time entered candidate ranking")
    rows = copy.deepcopy(record["candidates"])
    rows[0]["final_eval_loss"] = 1.9
    assert choose(rows, *args)["decision"] == "tune"


def test_lora_stage_handoff() -> None:
    """Exercise a synthetic v3 scan: identities, adapters, interruption and replay."""
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    cold = {}
    exec(cells["project-config-63"], cold)
    exec(cells["project-preflight-63"], cold)
    with patch.object(runtime, "run_lora_candidate", side_effect=AssertionError("prepare started training")), \
         patch.object(runtime, "publish_json_record", side_effect=AssertionError("prepare wrote assets")):
        for cell_id in ("project-contract-63", "project-run-63", "project-evidence-63", "project-decision-63"):
            exec(cells[cell_id], cold)
    assert cold["RUN_STAGE"] == "prepare" and cold["RUN_LABEL"] == "replace_with_unique_label"
    tree = ast.parse(cells["project-contract-63"])
    namespace = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                            type_ignores=[]), "lora_policy", "exec"), namespace)
    choose = namespace["select_lora_candidate_for_run"]
    contract = json.loads((root / "benchmarks/contracts/lora_variant_selection_v3.json").read_text())
    historical = json.loads((root / "benchmarks/results/63_lora_variants/lora_variant_sweep_lora_variant_selection_v2.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        upstream = target / contract["upstream_manifest"]
        upstream.parent.mkdir(parents=True)
        upstream.write_bytes((root / contract["upstream_manifest"]).read_bytes())
        record = {key: copy.deepcopy(contract[key]) for key in (
            "schema_version", "semantic_id", "measurement_version", "environment_profile",
            "evaluation_reduction", "sequence_protocol", "model", "workload", "gates",
            "upstream_manifest", "upstream_manifest_sha256",
        )}
        record.update(run_mode="real_gpu", run_label="fixture", candidates=copy.deepcopy(historical["candidates"]),
                      contract_sha256=hashlib.sha256(runtime.json_record_bytes(contract)).hexdigest(),
                      upstream_sample_ids_hash=historical["upstream_sample_ids_hash"])
        for row in record["candidates"]:
            row["artifact"] = None
            if row["run_seed"] == contract["workload"]["run_seeds"][0]:
                path = target / f"benchmarks/results/63_lora_variants/adapters_fixture/{row['name']}"
                path.mkdir(parents=True)
                (path / "adapter.bin").write_bytes(b"synthetic-adapter")
                row["artifact"] = {"path": path.relative_to(target).as_posix(),
                                   "sha256": runtime.hash_directory(path)}
        raw, report, handoff, receipt = (target / name for name in ("raw.json", "report.json", "handoff.json", "receipt.json"))
        runtime.publish_json_record(raw, record)
        args = (target, raw, report, handoff, receipt, contract, "fixture", choose)
        self_check = runtime.close_lora_selection(*args)
        assert self_check["status"] == "completed"
        assert self_check["registration_receipt"]["decision"] == "accept"
        delivery, scan, admitted = runtime.read_lora_delivery_inputs(target, handoff, contract)
        assert delivery["selected_config"]["name"] == "rank4"
        assert scan == record and admitted["decision"] == "accept"
        # The consumer must resolve repository-relative inputs independently of
        # the notebook working directory and reject changed files or bindings.
        with patch("os.getcwd", return_value="/tmp"):
            assert runtime.read_lora_delivery_inputs(target, handoff, contract)[0] == delivery
        for mutate in (
            lambda m: m.update(source_evidence_sha256="0" * 64),
            lambda m: m.update(selection_evidence="../outside.json"),
            lambda m: m["selected_config"].update(alpha=32),
            lambda m: m["artifact"].update(seed=999),
            lambda m: m["artifact"].update(sha256="0" * 64),
            lambda m: m["repeat_results"].pop(),
            lambda m: m.update(next_consumer="another-project"),
        ):
            changed = copy.deepcopy(delivery)
            mutate(changed)
            handoff.write_bytes(runtime.json_record_bytes(changed))
            try:
                runtime.read_lora_delivery_inputs(target, handoff, contract)
            except ValueError:
                pass
            else:
                raise AssertionError("delivery accepted changed selection inputs")
        handoff.write_bytes(runtime.json_record_bytes(delivery))
        selected_adapter = target / delivery["artifact"]["adapter_path"] / "adapter.bin"
        selected_adapter.write_bytes(b"changed-adapter")
        try:
            runtime.read_lora_delivery_inputs(target, handoff, contract)
        except ValueError:
            pass
        else:
            raise AssertionError("delivery accepted changed adapter bytes")
        selected_adapter.write_bytes(b"synthetic-adapter")
        upstream_bytes = upstream.read_bytes()
        upstream.write_bytes(b"{}")
        try:
            runtime.read_lora_delivery_inputs(target, handoff, contract)
        except ValueError:
            pass
        else:
            raise AssertionError("delivery accepted changed data manifest")
        upstream.write_bytes(upstream_bytes)
        delivery_nb = json.loads((root / "02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb").read_text())
        delivery_scope = {}
        for cell in delivery_nb["cells"]:
            if cell["id"] in ("fd7f19d6", "7898f0b4"):
                exec("".join(cell["source"]), delivery_scope)
        assert delivery_scope["validate_selection_manifest"](delivery)["consumable"]
        mtime = [p.stat().st_mtime_ns for p in (raw, report, handoff, receipt)]
        assert runtime.close_lora_selection(*args)["status"] == "already_complete"
        assert mtime == [p.stat().st_mtime_ns for p in (raw, report, handoff, receipt)]
        handoff.unlink()
        assert runtime.close_lora_selection(*args)["status"] == "completed"
        receipt.unlink()
        actual_publish = runtime.publish_json_record
        def interrupt(path, payload):
            """Interrupt before receipt publication, leaving reusable earlier files."""
            if path == receipt:
                raise RuntimeError("injected interruption")
            return actual_publish(path, payload)
        with patch.object(runtime, "publish_json_record", side_effect=interrupt):
            try:
                runtime.close_lora_selection(*args)
            except RuntimeError as error:
                assert str(error) == "injected interruption"
            else:
                raise AssertionError("expected interruption")
        assert runtime.close_lora_selection(*args)["status"] == "completed"
        for mutate in (
            lambda r: r.update(run_label="wrong"),
            lambda r: r.update(contract_sha256="0" * 64),
            lambda r: r["candidates"].pop(),
            lambda r: r["candidates"][0]["config"].update(alpha=32),
            lambda r: r["candidates"][0]["artifact"].update(sha256="0" * 64),
        ):
            broken = copy.deepcopy(record)
            mutate(broken)
            raw.write_bytes(runtime.json_record_bytes(broken))
            try:
                runtime.read_lora_selection(target, raw, contract, "fixture", choose)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid scan accepted")
        raw.write_bytes(runtime.json_record_bytes(record))
        handoff.write_text("{}")
        try:
            runtime.close_lora_selection(*args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("conflicting handoff overwritten")
        assert handoff.read_text() == "{}"
        handoff.unlink()
        report.unlink()
        receipt.unlink()
        # A complete matrix with no candidate under the memory budget retains
        # the raw facts and a reject report, but never a consumable handoff.
        for row in record["candidates"]:
            row["peak_allocated_bytes"] = contract["gates"]["maximum_peak_allocated_bytes"] + 1
        raw.write_bytes(runtime.json_record_bytes(record))
        assert runtime.close_lora_selection(*args)["registration_receipt"]["decision"] == "reject"
        assert report.exists() and receipt.exists() and not handoff.exists()


def test_lora_checkpoint_validation() -> None:
    """Reject changed state bytes, identities and invalid sequential cursors."""
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory)
        (checkpoint / "adapter").mkdir()
        (checkpoint / "adapter/weights.bin").write_bytes(b"synthetic-adapter")
        for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
            (checkpoint / name).write_bytes(b"synthetic-state")
        identity = {"model_commit": "commit", "train_examples_sha256": "train",
                    "evaluation_examples_sha256": "eval", "lora_config": {"rank": 4},
                    "total_steps": 9, "checkpoint_step": 5, "train_example_count": 3}
        progress = {"schema_version": "lora-checkpoint/v1", "training_identity": identity,
                    "global_step": 5, "next_sample_index": 2}
        path = checkpoint / "progress.json"
        path.write_bytes(runtime.json_record_bytes(progress))
        digest = runtime.hash_directory(checkpoint)
        assert runtime.read_lora_checkpoint(checkpoint, digest, identity) == progress
        (checkpoint / "rng.pt").write_bytes(b"changed")
        try:
            runtime.read_lora_checkpoint(checkpoint, digest, identity)
        except ValueError:
            pass
        else:
            raise AssertionError("changed checkpoint bytes accepted")
        (checkpoint / "rng.pt").write_bytes(b"synthetic-state")
        for field in ("model_commit", "train_examples_sha256", "evaluation_examples_sha256", "lora_config"):
            changed = copy.deepcopy(identity)
            changed[field] = "different"
            try:
                runtime.read_lora_checkpoint(checkpoint, digest, changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"changed {field} accepted")
        for mutate in (
            lambda p: p.update(global_step=True),
            lambda p: p.update(global_step=9),
            lambda p: p.update(global_step=4),
            lambda p: p.update(next_sample_index=0),
            lambda p: p.update(next_sample_index=2.0),
            lambda p: p.update(schema_version="old"),
        ):
            changed = copy.deepcopy(progress)
            mutate(changed)
            path.write_bytes(runtime.json_record_bytes(changed))
            try:
                runtime.read_lora_checkpoint(checkpoint, runtime.hash_directory(checkpoint), identity)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid checkpoint progress accepted")
        path.write_bytes(runtime.json_record_bytes(progress))
        (checkpoint / "optimizer.pt").unlink()
        try:
            runtime.read_lora_checkpoint(checkpoint, runtime.hash_directory(checkpoint), identity)
        except ValueError:
            pass
        else:
            raise AssertionError("missing optimizer state accepted")


def test_delivery_gate_requires_verified_recovery() -> None:
    """The visible delivery gate rejects old or incompletely checked recovery."""
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in nb["cells"]}
    scope = {}
    exec(cells["fd7f19d6"], scope)
    tree = ast.parse(cells["2ababa5c"])
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                            type_ignores=[]), "delivery_gate", "exec"), scope)
    gate = scope["decide_sft_delivery_for_run"]
    evidence = {"measurement_version": "sft-delivery-recovery/v4", "status": "ok",
                "initial_eval_loss": 2.0, "final_eval_loss": 1.5, "reload_loss_abs_diff": 0.0,
                "update_dynamics": {"finite_all": True}, "artifact": {"sha256": "fixture"},
                "checkpoint": {"completed_step": 9, "saved_step": 5, "resumed_from_step": 5,
                               "training_update_count": 9, "optimizer_state_present": True,
                               "scheduler_state_present": True, "rng_state_present": True,
                               "identity_verified": True, "files_sha256_verified": True,
                               "sample_cursor_verified": True}}
    gates = {"maximum_eval_loss": 2.0, "maximum_eval_loss_regression": 0.0,
             "maximum_reload_loss_abs_diff": 1e-4}
    assert gate(evidence, gates, 9)["decision"] == "accept"
    for key in ("identity_verified", "files_sha256_verified", "sample_cursor_verified"):
        changed = copy.deepcopy(evidence)
        changed["checkpoint"].pop(key)
        assert gate(changed, gates, 9)["decision"] == "reject"
    changed = copy.deepcopy(evidence)
    changed.pop("measurement_version")
    assert gate(changed, gates, 9)["decision"] == "reject"


def test_sft_delivery_stage_registration() -> None:
    """Exercise cold prepare/close with synthetic local assets, never training."""
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb").read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in nb["cells"]}
    stage_ids = ("32710103", "106f968f", "48fd60f3", "1ed04b37", "2ababa5c", "project-decision-62")
    cold = {}
    with patch("importlib.reload", return_value=runtime), \
         patch.object(runtime, "run_lora_delivery", side_effect=AssertionError("prepare trained")), \
         patch.object(runtime, "publish_json_record", side_effect=AssertionError("prepare published")):
        for cell_id in stage_ids:
            exec(cells[cell_id], cold)
    assert cold["RUN_STAGE"] == "prepare" and "torch" not in cold
    decide = cold["decide_sft_delivery_for_run"]
    contract = cold["CONTRACT"]
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        for key in ("selection_manifest", "selection_evidence", "upstream_manifest"):
            dest = target / contract[key]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((root / contract[key]).read_bytes())
        selection, _, data = runtime.read_sft_delivery_inputs(target, contract)
        result_dir = target / "benchmarks/results/62_sft_delivery"
        checkpoint, adapter = result_dir / "checkpoint_fixture", result_dir / "adapter_fixture"
        (checkpoint / "adapter").mkdir(parents=True)
        (checkpoint / "adapter/weights.bin").write_bytes(b"synthetic")
        for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
            (checkpoint / name).write_bytes(b"synthetic")
        adapter.mkdir()
        (adapter / "weights.bin").write_bytes(b"synthetic")
        work = contract["workload"]
        identity = {"model_id": contract["model"]["id"], "model_commit": contract["model"]["resolved_commit"],
                    "lora_config": selection["selected_config"], "train_example_count": work["train_examples"],
                    **{key: work[key] for key in ("max_length", "total_steps", "checkpoint_step", "learning_rate", "seed", "dtype")},
                    "optimizer": "AdamW_default", "scheduler": "constant_lambda_1",
                    "sample_order": "sequential_step_mod_count",
                    "train_examples_sha256": "a" * 64, "evaluation_examples_sha256": "b" * 64}
        progress = {"schema_version": "lora-checkpoint/v1", "global_step": 12,
                    "next_sample_index": 12, "training_identity": identity}
        (checkpoint / "progress.json").write_bytes(runtime.json_record_bytes(progress))
        evidence = {"status": "ok", "measurement_version": contract["measurement_version"],
                    "initial_eval_loss": 1.9, "final_eval_loss": 1.8, "reload_loss_abs_diff": 0.0,
                    "checkpoint": {"path": str(checkpoint), "sha256": runtime.hash_directory(checkpoint),
                                   "saved_step": 12, "resumed_from_step": 12, "completed_step": 24,
                                   "training_update_count": 24, "optimizer_state_present": True,
                                   "scheduler_state_present": True, "rng_state_present": True,
                                   "identity_verified": True, "files_sha256_verified": True, "sample_cursor_verified": True,
                                   "next_sample_index": 12, "training_identity": identity,
                                   "recovery_scope": contract["recovery_scope"]},
                    "artifact": {"path": str(adapter), "sha256": runtime.hash_directory(adapter)},
                    "update_dynamics": {"finite_all": True}}
        record = {key: copy.deepcopy(contract[key]) for key in ("schema_version", "semantic_id", "measurement_version",
                  "environment_profile", "evaluation_reduction", "sequence_protocol", "model", "workload", "gates")}
        record.update(run_mode="real_gpu", run_label="fixture", evidence=evidence,
                      contract_sha256=hashlib.sha256(runtime.json_record_bytes(contract)).hexdigest(),
                      selection_manifest=contract["selection_manifest"], selection_manifest_sha256=contract["selection_manifest_sha256"],
                      selected_config=selection["selected_config"],
                      data={"manifest": contract["upstream_manifest"], "sample_ids_hash": data["audit"]["sample_ids_hash"]})
        raw, report, manifest, receipt = (result_dir / f"sft_delivery_{kind}_fixture.json"
                                          for kind in ("evidence", "report", "manifest", "registration"))
        runtime.publish_json_record(raw, record)
        args = (target, raw, report, manifest, receipt, contract, "fixture", decide)
        assert runtime.close_sft_delivery(*args)["status"] == "completed"
        timestamps = [p.stat().st_mtime_ns for p in (raw, report, manifest, receipt)]
        assert runtime.close_sft_delivery(*args)["status"] == "already_complete"
        assert timestamps == [p.stat().st_mtime_ns for p in (raw, report, manifest, receipt)]
        manifest.unlink()
        assert runtime.close_sft_delivery(*args)["status"] == "completed"
        receipt.unlink()
        publish = runtime.publish_json_record
        def interrupt(path, payload):
            """Inject an interruption before receipt publication."""
            if path == receipt:
                raise RuntimeError("interrupted")
            return publish(path, payload)
        with patch.object(runtime, "publish_json_record", side_effect=interrupt):
            try:
                runtime.close_sft_delivery(*args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected interrupted registration")
        assert runtime.close_sft_delivery(*args)["status"] == "completed"
        # A fresh close kernel derives everything from contract + files.
        contract_path = target / "contract.json"
        contract_path.write_bytes(runtime.json_record_bytes(contract))
        registry_path = target / "benchmarks/project_registry/part02.json"
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text(json.dumps({"projects": [{"semantic_id": "PT-SFT-DELIVERY", "default_experiment_contract": "contract.json"}]}))
        close_scope = {"RUN_LABEL": "fixture", "RUN_STAGE": "close"}
        with patch("tools.project_runtime.bootstrap_project_root", return_value=target):
            for cell_id in stage_ids[1:]:
                exec(cells[cell_id], close_scope)
        assert close_scope["registration"]["status"] == "already_complete" and "torch" not in close_scope
        for mutate in (
            lambda r: r.update(run_label="wrong"),
            lambda r: r["workload"].update(total_steps=99),
            lambda r: r.update(contract_sha256="0" * 64),
            lambda r: r["selected_config"].update(rank=8),
            lambda r: r["evidence"]["artifact"].update(sha256="0" * 64),
            lambda r: r["evidence"]["checkpoint"].update(next_sample_index=0),
        ):
            changed = copy.deepcopy(record)
            mutate(changed)
            raw.write_bytes(runtime.json_record_bytes(changed))
            try:
                runtime.read_sft_delivery(target, raw, contract, "fixture", decide)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid delivery accepted")
        raw.write_bytes(runtime.json_record_bytes(record))
        manifest.write_text("{}")
        try:
            runtime.close_sft_delivery(*args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("conflicting baseline overwritten")
        assert manifest.read_text() == "{}"
        for path in (report, manifest, receipt):
            path.unlink()
        rejected = copy.deepcopy(record)
        rejected["evidence"]["final_eval_loss"] = 2.1
        raw.write_bytes(runtime.json_record_bytes(rejected))
        assert runtime.close_sft_delivery(*args)["registration_receipt"]["decision"] == "reject"
        assert report.exists() and receipt.exists() and not manifest.exists()


def test_qlora_stages_and_actual_token_count() -> None:
    """Check real counter code on CPU tensors and replay registration fixtures."""
    import torch
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    labels = torch.tensor([[-100, 4, 5, -100], [9, 8, -100, -100]])
    assert runtime.causal_supervised_token_count(labels) == 3
    tree = ast.parse(cells["65gpu53code"])
    counter = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "CountingTrainer")
    class BaseTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return 1
    counter_scope = {"Trainer": BaseTrainer, "project_runtime": runtime}
    exec(compile(ast.Module(body=[counter], type_ignores=[]), "actual_training_counter", "exec"), counter_scope)
    trainer = counter_scope["CountingTrainer"]()
    trainer.compute_loss(types.SimpleNamespace(training=True), {"labels": labels})
    trainer.compute_loss(types.SimpleNamespace(training=False), {"labels": labels})
    trainer.compute_loss(types.SimpleNamespace(training=True), {"labels": labels[:1]})
    assert trainer.supervised_tokens_seen == 5, "count actual batches; exclude evaluation and shifted first position"
    scope = {}
    with patch("importlib.reload", return_value=runtime), \
         patch.object(runtime, "publish_json_record", side_effect=AssertionError("prepare published")):
        for cell_id in ("65gpu51code", "65gpu52code", "65gpu53code", "65gpu54code", "65gpu55code", "65gpu56code"):
            exec(cells[cell_id], scope)
    assert scope["RUN_STAGE"] == "prepare" and "torch" not in scope
    contract = scope["CONTRACT"]
    summarize, compare, build_manifest = (scope[name] for name in ("summarize_variant", "_compare_reports", "_build_artifact_manifest"))
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory)
        upstream = target / contract["upstream_manifest"]
        upstream.parent.mkdir(parents=True)
        upstream.write_bytes((root / contract["upstream_manifest"]).read_bytes())
        data = runtime.read_lora_data_manifest(target, contract)
        result_dir = target / "benchmarks/results/65_qlora_selection"
        result_dir.mkdir(parents=True)
        scope.update(REPO_ROOT=target, RESULTS_DIR=result_dir, RUN_LABEL="fixture",
                     UPSTREAM_DATA_MANIFEST=upstream, RESULT_PATH=result_dir / "qlora_selection_fixture.json")
        def member(label, seed, batch=1, steps=12, saved=True):
            """Synthetic paired member; bytes below are not trained adapters."""
            row = {"label": label, "model_id": contract["model"]["id"], "seed": seed,
                   "steps": steps, "fixed_micro_batch": batch, "train_examples": 96, "evaluation_examples": 32,
                   "compute_dtype": "torch.bfloat16", "low_bit_active": label.startswith("qlora"),
                   "gradient_checkpointing": False, "preparation": "kbit_no_gradient_checkpointing" if label.startswith("qlora") else "bf16_default",
                   "supervised_tokens_seen": 100, "elapsed_training_s": 1.0, "step_time_s": 1.0 / steps,
                   "supervised_tokens_per_s": 100.0, "peak_memory_mb": 700 if label.startswith("qlora") else 1000,
                   "val_loss": 1.01 if label.startswith("qlora") else 1.0,
                   "artifact_path": None, "artifact_hash": None}
            if saved:
                adapter = result_dir / f"fixture_{label}_seed{seed}_adapter"
                adapter.mkdir()
                (adapter / "weights.bin").write_bytes(b"synthetic-only")
                row.update(artifact_path=str(adapter), artifact_hash=runtime.hash_directory(adapter))
            return row
        groups = {label: [member(label, seed) for seed in contract["workload"]["seeds"]]
                  for label in ("lora_baseline", "qlora_candidate")}
        probe = {"status": "completed", "reports": []}
        for label, maximum in (("lora_baseline", 2), ("qlora_candidate", 4)):
            attempts = []
            for size in (1, 2, 4, 8):
                if size > maximum:
                    attempts.append({"fixed_micro_batch": size, "status": "oom", "error": "synthetic OOM"})
                    break
                attempts.append(member(f"{label}_mb{size}", 17, size, 2, False))
            probe["reports"].append({"label": label, "attempts": attempts, "max_micro_batch": maximum})
        record = {key: copy.deepcopy(contract[key]) for key in ("schema_version", "semantic_id", "measurement_version",
                  "environment_profile", "evaluation_reduction", "timing_scope", "sequence_protocol", "model",
                  "workload", "adapter", "preparation", "capacity_probe", "gates", "upstream_manifest", "upstream_manifest_sha256")}
        record.update(run_label="fixture", run_mode="real_gpu", status="ok", paired_reports=groups,
                      baseline=summarize(groups["lora_baseline"]), candidate=summarize(groups["qlora_candidate"]),
                      scale_probe=probe, environment={"kind": "synthetic"}, evidence_level="fixture_only",
                      contract_sha256=hashlib.sha256(runtime.json_record_bytes(contract)).hexdigest(),
                      data_sample_ids_hash=data["audit"]["sample_ids_hash"])
        raw, report, manifest, receipt = (result_dir / name for name in ("qlora_selection_raw_fixture.json", "qlora_selection_fixture.json",
                                                                       "qlora_artifact_manifest_fixture.json", "qlora_registration_fixture.json"))
        runtime.publish_json_record(raw, record)
        args = (target, raw, report, manifest, receipt, contract, "fixture", summarize, compare, build_manifest)
        assert runtime.close_qlora_selection(*args)["registration_receipt"]["decision"] == "accept"
        times = [p.stat().st_mtime_ns for p in (raw, report, manifest, receipt)]
        assert runtime.close_qlora_selection(*args)["status"] == "already_complete"
        assert times == [p.stat().st_mtime_ns for p in (raw, report, manifest, receipt)]
        manifest.unlink()
        assert runtime.close_qlora_selection(*args)["status"] == "completed"
        receipt.unlink()
        publish = runtime.publish_json_record
        def interrupt(path, payload):
            if path == receipt:
                raise RuntimeError("interrupt receipt")
            return publish(path, payload)
        with patch.object(runtime, "publish_json_record", side_effect=interrupt):
            try:
                runtime.close_qlora_selection(*args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("interruption not exercised")
        assert runtime.close_qlora_selection(*args)["status"] == "completed"
        contract_path = target / "contract.json"
        contract_path.write_bytes(runtime.json_record_bytes(contract))
        registry_path = target / "benchmarks/project_registry/part02.json"
        registry_path.parent.mkdir(parents=True)
        registry_path.write_text(json.dumps({"projects": [{"semantic_id": "PT-QLORA-SELECTION", "default_experiment_contract": "contract.json"}]}))
        fresh_close = {"RUN_LABEL": "fixture", "RUN_STAGE": "close"}
        with patch("tools.project_runtime.bootstrap_project_root", return_value=target):
            for cell_id in ("65gpu52code", "65gpu53code", "65gpu54code", "65gpu55code", "65gpu56code"):
                exec(cells[cell_id], fresh_close)
        assert fresh_close["registration"]["status"] == "already_complete" and "torch" not in fresh_close
        for mutate in (
            lambda r: r.update(run_label="wrong"),
            lambda r: r["paired_reports"]["qlora_candidate"].pop(),
            lambda r: r["paired_reports"]["qlora_candidate"][0].update(supervised_tokens_seen=5),
            lambda r: r["paired_reports"]["qlora_candidate"][0].update(gradient_checkpointing=True),
            lambda r: r["baseline"].update(val_loss_median=0.01),
            lambda r: r["paired_reports"]["qlora_candidate"][0].update(artifact_hash="0" * 64),
            lambda r: r["scale_probe"]["reports"][0].update(max_micro_batch=8),
        ):
            changed = copy.deepcopy(record)
            mutate(changed)
            raw.write_bytes(runtime.json_record_bytes(changed))
            try:
                runtime.read_qlora_selection(target, raw, contract, "fixture", summarize, compare)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid QLoRA record accepted")
        raw.write_bytes(runtime.json_record_bytes(record))
        manifest.write_text("{}")
        try:
            runtime.close_qlora_selection(*args)
        except FileExistsError:
            pass
        else:
            raise AssertionError("conflicting QLoRA manifest overwritten")
        assert manifest.read_text() == "{}"
        for path in (report, manifest, receipt):
            path.unlink()
        rejected = copy.deepcopy(record)
        for row in rejected["paired_reports"]["qlora_candidate"]:
            row["val_loss"] = 2.0
        rejected["candidate"] = summarize(rejected["paired_reports"]["qlora_candidate"])
        raw.write_bytes(runtime.json_record_bytes(rejected))
        assert runtime.close_qlora_selection(*args)["registration_receipt"]["decision"] == "tune"
        assert report.exists() and receipt.exists() and not manifest.exists()


def main() -> None:
    test_qlora_stages_and_actual_token_count()
    test_sft_delivery_stage_registration()
    test_lora_checkpoint_validation()
    test_delivery_gate_requires_verified_recovery()
    test_lora_stage_handoff()
    test_lora_scan_reader()
    test_sft_registration_recovery()
    test_sft_prepare_and_gate_isolation()
    test_sft_collection_and_cold_close()
    test_sft_audit_matches_training_sequence()
    records = [
        {"instruction": "solve one", "input": "", "output": "answer one"},
        {"instruction": "solve two", "input": "", "output": "answer two"},
        {"instruction": "solve one", "input": "", "output": "answer one"},
        {"instruction": "solve two", "input": "", "output": "new answer"},
    ]
    audit = audit_sft_records(records, _Tokenizer(), max_length=32, train_count=2)
    assert audit["duplicate_count"] == 1
    assert audit["split_leakage_count"] == 2
    assert audit["zero_supervision_count"] == 0
    assert audit["train_prompt_ids_hash"] == audit["evaluation_prompt_ids_hash"]
    examples = build_causal_sft_examples(records[:2], _Tokenizer(), max_length=32)
    assert len(examples) == 2
    assert all(len(example["input_ids"]) == 32 for example in examples)
    assert all(-100 in example["labels"] for example in examples)

    preference_audit = audit_preference_records(
        [
            {
                "chosen": [{"content": "question", "role": "user"}, {"content": "good", "role": "assistant"}],
                "rejected": [{"content": "question", "role": "user"}, {"content": "bad", "role": "assistant"}],
            },
            {
                "chosen": [{"content": "question", "role": "user"}, {"content": "same", "role": "assistant"}],
                "rejected": [{"content": "question", "role": "user"}, {"content": "same", "role": "assistant"}],
            },
        ]
    )
    assert preference_audit["valid_ratio"] == 1.0
    assert preference_audit["identical_response_count"] == 1
    long_pair = build_preference_pair_sequences(
        _Tokenizer(),
        " ".join(f"prompt-{index}" for index in range(40)),
        "chosen answer",
        "rejected answer with more tokens",
        max_length=12,
    )
    assert long_pair["prompt_truncated"]
    assert long_pair["chosen"]["response_start"] == long_pair["rejected"]["response_start"]
    assert 0 < long_pair["chosen"]["response_start"] < len(long_pair["chosen"]["input_ids"])
    assert 0 < long_pair["rejected"]["response_start"] < len(long_pair["rejected"]["input_ids"])
    assert len(long_pair["chosen"]["input_ids"]) <= 12
    assert len(long_pair["rejected"]["input_ids"]) <= 12
    assert extract_final_number("reasoning... #### 1,024") == "1024"

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "result.json"
        save_json_record(output, {"status": "ok"})
        assert output.is_file()
        artifact = Path(directory) / "artifact"
        artifact.mkdir()
        (artifact / "weights.bin").write_bytes(b"weights")
        first_hash = hash_directory(artifact)
        assert first_hash == hash_directory(artifact)
        try:
            save_json_record(output, {"status": "unexpected overwrite"})
        except FileExistsError:
            pass
        else:
            raise AssertionError("save_json_record must reject an existing path")

    print("post-training project runtime: ok")


if __name__ == "__main__":
    main()
