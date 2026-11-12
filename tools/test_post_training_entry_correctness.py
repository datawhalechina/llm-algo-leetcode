"""CPU-only regression tests for post-training recovery and delivery entry points."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools import distillation_project_runtime as runtime
from tools.distillation_matrix_runtime import resolve_g1_close_record, rank_validation_candidates, summarize_confirmation_runs

ROOT = Path(__file__).resolve().parents[1]


def notebook_source(prefix, cell_id):
    """Read source only; never run notebook initialization or GPU cells."""
    path = next((ROOT / "02_PyTorch_Algorithms").glob(prefix + "*.ipynb"))
    cells = json.loads(path.read_text())["cells"]
    return "".join(next(c for c in cells if c["id"] == cell_id)["source"])


def evaluation(quality=0.8, latency=20):
    """Build complete finite evaluation facts for one fixed category."""
    return {"quality_score": quality, "mean_latency_ms": latency,
            "category_scores": {"core": {"quality_score": quality}},
            "format_compliance": {"compliance_ratio": 1.0},
            "generated_tokens": {"saturation_ratio": 0.0, "mean_per_request": 5},
            "latency_protocol": {"repeats": 2, "sample_count": 4}}


class EntryCorrectnessTests(unittest.TestCase):
    def test_real_entry_requires_unique_label(self):
        """Real training/artifact checks must not publish placeholder runs."""
        for prefix, cell_id, mode in (("84_", "84-config", "real_gpu"),
                                      ("85_", "85-config", "real_gpu"),
                                      ("86_", "86-config", "real_artifact")):
            source = notebook_source(prefix, cell_id)
            scope = {}
            exec(source, scope)
            self.assertEqual(scope["RUN_LABEL"], "replace_with_unique_label")
            real = source.replace('RUN_MODE = "dry_run"', f'RUN_MODE = "{mode}"', 1)
            with self.assertRaises(ValueError):
                exec(real, {})
            invalid = source.replace('RUN_MODE = "dry_run"', 'RUN_MODE = "unknown"', 1)
            with self.assertRaises(ValueError):
                exec(invalid, {})

    def test_actual_recovery_entry(self):
        """The real function must stop rather than invent a missing rollback."""
        tree = ast.parse(notebook_source("86_DPO", "86-scenarios"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose_recovery_action_for_run")
        env = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "86-real-recovery", "exec"), env)
        choose = env[function.name]
        for available in ({}, {"pre_update_checkpoint": False, "verified_baseline": False}, {"verified_baseline": "missing"}):
            row = choose({"stage": "update", "available": available})
            self.assertEqual(row["action"], "block_update")
            self.assertIsNone(row["required_state"])
            self.assertFalse(row["reused_state"])
        self.assertEqual(choose({"stage": "update", "available": {"verified_baseline": True}})["action"], "rollback_to_verified_baseline")
        self.assertEqual(choose({"stage": "update", "available": {"pre_update_checkpoint": True, "verified_baseline": True}})["action"], "restore_pre_update_checkpoint")
        self.assertEqual(choose({"stage": "artifact_validation", "available": {}})["action"], "block_release")

    def summary_inputs(self):
        """Construct valid teacher, baseline and independently loadable candidate."""
        teacher = {"quality_score": 0.8, "artifact_size_mb": 3000, "latency_ms": 60, "category_scores": {"core": {"quality_score": 0.8}}}
        baseline = {**teacher, "quality_score": 0.5, "artifact_size_mb": 1000, "latency_ms": 20}
        candidate = {**baseline, "quality_score": 0.75, "artifact_verified": True, "load_status": "loaded", "latency_protocol": {"repeats": 2, "sample_count": 4}}
        return teacher, baseline, candidate

    def test_shared_summary_rejects_invalid_evidence(self):
        """NaN, infinity, missing categories and fractional counts fail closed."""
        for index in range(3):
            for field in ("quality_score", "latency_ms", "artifact_size_mb"):
                for bad in (float("nan"), float("inf"), None, True):
                    records = list(copy.deepcopy(self.summary_inputs()))
                    records[index][field] = bad
                    with self.assertRaises(ValueError):
                        runtime.summarize_distilled_candidate(*records)
        for categories in ({}, {"other": {"quality_score": 0.8}}, {"core": {"quality_score": float("nan")}}):
            teacher, baseline, candidate = self.summary_inputs()
            candidate["category_scores"] = categories
            with self.assertRaises(ValueError):
                runtime.summarize_distilled_candidate(teacher, baseline, candidate)

    def test_direct_decision_cannot_accept_nan(self):
        """Even callers bypassing the summarizer cannot accept invalid metrics."""
        summary = runtime.summarize_distilled_candidate(*self.summary_inputs())
        gates = {"min_quality_gain": 0.05, "min_teacher_retention": 0.85,
                 "max_teacher_size_ratio": 0.5, "max_teacher_latency_ratio": 0.7,
                 "max_baseline_latency_ratio": 1.05, "max_regressed_categories": 0}
        self.assertEqual(runtime.decide_distilled_student({"ready": True}, summary, gates)["decision"], "accept")
        for field in ("quality_gain", "teacher_retention", "teacher_size_ratio", "teacher_latency_ratio", "baseline_latency_ratio", "regressed_category_count"):
            with self.assertRaises(ValueError):
                runtime.decide_distilled_student({"ready": True}, {**summary, field: float("nan")}, gates)
        with self.assertRaises(ValueError):
            runtime.decide_distilled_student({"ready": True}, summary, {**gates, "min_quality_gain": float("nan")})

    def test_matrix_ranking_requires_categories(self):
        """An empty category map must not become zero category regressions."""
        candidate = {"candidate_id": "c", "validation": evaluation()}
        self.assertEqual(rank_validation_candidates(evaluation(0.5), [candidate], keep=1)["selected_candidate_ids"], ["c"])
        candidate["validation"]["category_scores"] = {}
        with self.assertRaises(ValueError):
            rank_validation_candidates(evaluation(0.5), [candidate], keep=1)

    def test_confirmation_rejects_missing_or_nan_categories(self):
        """Confirmation cannot qualify a seed by silently dropping a category."""
        gates = dict(min_quality_gain=0.05, max_regressed_categories=0,
                     max_saturation_increase=0.1, min_format_compliance=0.8,
                     max_baseline_latency_ratio=1.1, required_qualified_runs=1)
        run = {"seed": 42, "artifact_verified": True, "artifact": {}, "evaluation": evaluation()}
        self.assertTrue(summarize_confirmation_runs(evaluation(0.5), [run], **gates)["confirmed"])
        for categories in ({}, {"core": {"quality_score": float("nan")}}):
            broken = copy.deepcopy(run)
            broken["evaluation"]["category_scores"] = categories
            with self.assertRaises(ValueError):
                summarize_confirmation_runs(evaluation(0.5), [broken], **gates)

    def test_matrix_to_actual_close(self):
        """Execute close source with fixture files, without importing torch or models."""
        tree = ast.parse(notebook_source("88_Distilled", "88-run"))
        branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "RUN_STAGE == 'close'")
        code = compile(ast.Module(body=branch.body, type_ignores=[]), "88-actual-close", "exec")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            training = {"train_steps": 128, "trajectory_mode": "fixed", "dose_selection": {"selected_step": 128, "selected_admissible": True, "dose_summaries": []}}
            screen = {"name": "g1", "training": training, "evaluation": evaluation(), "artifact": {"persisted": False}, "artifact_verified": False, "load_status": "evaluated_in_memory"}
            member = {"run_label": "r", "plan_hash": "plan", "common_config_hash": "common", "candidate": {"candidate_id": "c"}, "evidence": screen}
            member_path = target / "member.json"
            member_path.write_text(json.dumps(member))
            g1 = {"record_type": "g1_validation_matrix", "run_label": "r", "config_hash": "config", "student_start_snapshot": "student", "evidence": {"plan_hash": "plan", "common_config_hash": "common", "ranking": [{"candidate_id": "c"}], "selected_candidate_ids": [], "candidate_paths": [str(member_path)]}}
            identity = {key: key for key in ("development_prompt_ids_hash", "development_reference_answers_hash", "evaluation_prompt_ids_hash", "scoring_contract_sha256")}
            g0 = {"config_hash": "smaller-baseline-manifest", "config": {}, "workload": identity, "evidence": {"teacher": evaluation(0.9, 60), "student": evaluation(0.5, 20)}}
            g2 = {"config_hash": "config", "student_start_snapshot": "student", "evidence": {**screen, "name": "g2", "training": {**training, "trajectory_mode": "on_policy"}, "artifact": {"weight_bytes": 1024**2}}}
            paths = {name: target / (name + ".json") for name in ("g0", "g1", "g2")}
            for name, record in (("g0", g0), ("g1", g1), ("g2", g2)):
                paths[name].write_text(json.dumps(record))
            env = {"json": json, "distillation_runtime": runtime, "G0_PATH": paths["g0"], "G1_PATH": paths["g1"], "G2_PATH": paths["g2"], "G1_CONFIRM_PATH": target / "confirm.json", "CONFIG_HASH": "config", "SCHEMA_VERSION": "test", "SEMANTIC_ID": "test", "RUN_LABEL": "r", "DECISION_PATH": target / "decision.json", "DELIVERY_GATES": {}, "assets": {"teacher": {"snapshot": "teacher", "model_id": "teacher", "weight_bytes": 3*1024**2}, "student": {"snapshot": "student", "model_id": "student", "weight_bytes": 1024**2, "path": "rollback"}}, "workload": {"identity": {"selected_train_prompt_ids_hash": "train", "development_scoring_contract_sha256": "dev", "evaluation_prompt_ids_hash": "eval"}}, "validate_distillation_contract": runtime.validate_distillation_contract, "summarize_distilled_candidate": runtime.summarize_distilled_candidate, "decide_distilled_student": lambda *args: {"decision": "reject", "reason": "fixture has no persisted artifact"}, "save_json_record": lambda path, record: None}
            env.update(PROJECT_CONFIG={}, VALIDATION_IDENTITY=identity)
            exec(code, env)
            result = env["STAGE_RESULT"]
            self.assertEqual(result["candidate_decisions"]["g1"]["decision"], "tune")
            self.assertFalse(result["deployment_ready"])
            self.assertIsNone(result["handoff"])
            self.assertEqual(result["source_evidence"]["matrix_member"], str(member_path))
            # No eligible member and eligible-but-unconfirmed member are both nonpromotable.
            for selected in ([], ["c"]):
                g1["evidence"]["selected_candidate_ids"] = selected
                self.assertFalse(resolve_g1_close_record(g1, target / "confirm.json")["close_promotable"])
            g1["evidence"]["ranking"][0]["selected_step"] = 128
            chosen = {"candidate_id": "c", "selected_step": 128}
            persisted = {**screen, "artifact": {"path": "fixture-model", "weight_bytes": 1024**2}, "artifact_verified": True, "load_status": "loaded"}
            confirmation = {"run_label": "r", "config_hash": "config", "selected_candidate": chosen, "evidence": {"confirmed": True, "selected_seed": 42, "selected_artifact": persisted["artifact"]}}
            confirm_path = target / "confirm.json"
            confirm_path.write_text(json.dumps(confirmation))
            seed_path = target / "confirm_seed_42.json"
            seed_record = {"run_label": "r", "config_hash": "config", "candidate": chosen, "seed": 42, "evidence": persisted}
            seed_path.write_text(json.dumps(seed_record))
            resolved = resolve_g1_close_record(g1, confirm_path)
            self.assertTrue(resolved["close_promotable"])
            self.assertEqual(resolved["evidence"], persisted)
            paths["g1"].write_text(json.dumps(g1))
            exec(code, env)
            self.assertEqual(env["STAGE_RESULT"]["candidate_decisions"]["g1"]["decision"], "reject")
            self.assertEqual(env["STAGE_RESULT"]["source_evidence"]["seed"], str(seed_path))
            seed_record["seed"] = 43
            seed_path.write_text(json.dumps(seed_record))
            with self.assertRaises(ValueError):
                resolve_g1_close_record(g1, confirm_path)
            confirm_path.unlink()
            member["plan_hash"] = "wrong"
            member_path.write_text(json.dumps(member))
            with self.assertRaises(ValueError):
                resolve_g1_close_record(g1, target / "confirm.json")


if __name__ == "__main__":
    unittest.main()
