"""Check project 73 exercises and recorded measurements on CPU only."""

import ast
import copy
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TrainingBaselineExerciseTests(unittest.TestCase):
    """核对可填写的题目骨架与答案，而非只运行完整答案。"""

    def setUp(self):
        path = ROOT / "02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.functions = {}
        self.test_functions = []
        names = {
            "_valid_baseline_number", "validate_training_baseline",
            "build_training_baseline_evidence", "classify_training_bottleneck",
        }
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tags = cell.get("metadata", {}).get("tags", [])
            tree = ast.parse("".join(cell["source"]))
            for tag in ("tutorial-question", "tutorial-answer"):
                if tag in tags:
                    self.functions[tag] = {
                        node.name: node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name in names
                    }
            if "tutorial-test" in tags:
                self.test_functions.extend(
                    node for node in tree.body if isinstance(node, ast.FunctionDef)
                )

    def _scope(self, functions):
        """只执行基线统计函数，不导入模型或运行GPU阶段。"""
        scope = {"math": math, "statistics": statistics}
        module = ast.Module(body=list(functions), type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "baseline_exercise", "exec"), scope)
        return scope

    def test_only_filling_todos_restores_the_answer(self):
        """只补TODO表达式即可得到答案；遗漏samples等骨架行会失败。"""
        answers = self.functions["tutorial-answer"]
        assignments = {
            node.targets[0].id: node
            for function in answers.values() for node in ast.walk(function)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        todo_names = {
            1: ("sample_is_usable",),
            2: ("stage_medians", "dominant_stage"),
            3: ("capacity_is_high",),
        }

        class FillTodos(ast.NodeTransformer):
            def visit_Raise(self, node):
                call = node.exc
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "NotImplementedError" and call.args
                        and isinstance(call.args[0], ast.Constant)):
                    match = re.match(r"TODO ([123])", str(call.args[0].value))
                    if match:
                        return [copy.deepcopy(assignments[name])
                                for name in todo_names[int(match.group(1))]]
                return node

        filled = {}
        for name, question in self.functions["tutorial-question"].items():
            question = FillTodos().visit(copy.deepcopy(question))
            answer = copy.deepcopy(answers[name])
            for function in (question, answer):
                if ast.get_docstring(function) is not None:
                    function.body.pop(0)
            self.assertEqual(ast.dump(question), ast.dump(answer), name)
            filled[name] = question
        scope = self._scope(list(filled.values()) + self.test_functions)
        scope["test_baseline_evidence_summarizes_stability_and_hotspot"]()
        scope["test_baseline_contract_rejects_invalid_metrics"]()
        scope["test_baseline_evidence_rejects_invalid_inputs"]()

    def test_answer_rejects_invalid_reports_before_statistics(self):
        """非法测量、容量和账本必须在中位数及比例计算前停止。"""
        scope = self._scope(list(self.functions["tutorial-answer"].values())
                            + self.test_functions)
        scope["test_baseline_contract_rejects_incomplete_reports"]()
        scope["test_baseline_contract_rejects_invalid_metrics"]()
        scope["test_baseline_evidence_rejects_invalid_inputs"]()


class TrainingBaselineMeasurementTests(unittest.TestCase):
    """只用CPU核对Step 5的样本、汇总与关闭前重算函数。"""

    def setUp(self):
        notebook = json.loads((ROOT / "02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb").read_text())
        reader = next(cell for cell in notebook["cells"] if cell.get("id") == "73_result_reader")
        functions = [node for node in ast.parse("".join(reader["source"])).body
                     if isinstance(node, ast.FunctionDef)]
        scope = dict(math=math, SCHEMA_VERSION="training-performance-project/v1",
                     SEMANTIC_ID="PERF-TRAINING-BASELINE",
                     MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct", WORKLOAD="pressure",
                     BATCH_SIZE=1, SEQ_LEN=768, WARMUP=2, ITERS=5, SEED=42)
        exec(compile(ast.Module(body=functions, type_ignores=[]), "measurement_review", "exec"), scope)
        self.review = scope["review_measured_training_report"]
        self.scope = scope
        self.close_code = "".join(next(cell for cell in notebook["cells"]
                                      if cell.get("id") == "73_decision_handoff")["source"])
        self.report = json.loads((ROOT / "benchmarks/results/73_real_gpu_training_rtx5070ti_training_baseline_v1.json").read_text())

    def test_historical_samples_reproduce_summary_without_mutation(self):
        """历史v1无需改写即可通过重算；时间改善不等于显存改善。"""
        original = copy.deepcopy(self.report)
        review = self.review(self.report)
        self.assertEqual(review["issues"], [])
        self.assertEqual(review["recomputed_summary"], self.report["summary"])
        self.assertFalse(review["recomputed_summary"]["meaningful_memory_improved"])
        self.assertEqual(self.report, original)

    def test_modified_summary_and_aggregates_are_rejected(self):
        """改汇总、方向或组均值不能绕过原始样本核对。"""
        mutations = [
            ("summary", "step_time_delta_ms", 999),
            ("summary", "memory_improved", True),
            ("summary", "time_improved", 1),
            ("baseline", "step_time_ms", 1),
        ]
        for group, field, value in mutations:
            with self.subTest(group=group, field=field):
                report = copy.deepcopy(self.report)
                report[group][field] = value
                self.assertTrue(self.review(report)["issues"])
        report = copy.deepcopy(self.report)
        report.pop("summary")
        self.assertIn("summary_missing", self.review(report)["issues"])

    def test_invalid_samples_and_configuration_are_rejected(self):
        """计数、有限数值和实验配置都要先于决策检查。"""
        for value in (True, float("nan"), float("inf"), -1, 0, "1"):
            with self.subTest(value=value):
                report = copy.deepcopy(self.report)
                report["baseline"]["runs"][0]["step_time_ms"] = value
                self.assertTrue(self.review(report)["issues"])
        for field, value in (("repeats", 2), ("repeats", True), ("seq_len", 512), ("batch_size", True)):
            with self.subTest(field=field, value=value):
                report = copy.deepcopy(self.report)
                report["config"][field] = value
                self.assertTrue(self.review(report)["issues"])
        report = copy.deepcopy(self.report)
        report["baseline"]["runs"].pop()
        self.assertTrue(self.review(report)["issues"])
        report = copy.deepcopy(self.report)
        report["semantic_id"] = "another-experiment"
        self.assertIn("semantic_id_mismatch", self.review(report)["issues"])

    def test_probe_does_not_create_a_paired_conclusion(self):
        """只有BF16探测组时仍可读取，但不生成FP32对照结论。"""
        report = copy.deepcopy(self.report)
        report["config"]["mode"] = "bf16_probe"
        report["candidate"] = report.pop("tuned")
        report.pop("baseline")
        report.pop("summary")
        review = self.review(report)
        self.assertEqual(review["issues"], [])
        self.assertIsNone(review["recomputed_summary"])

    def test_close_rechecks_source_and_is_idempotent(self):
        """关闭使用刚重算的summary；重复运行不改写，源文件变化立即停止。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(self.report))
            reviewed = {"path": "source.json", "ready_for_decision": True,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            summaries = []

            def decide(summary):
                summaries.append(summary)
                return {"decision": "tune", "reason": "test"}

            def save_once(path, record):
                with path.open("x") as handle:
                    json.dump(record, handle)

            scope = dict(self.scope, RUN_STAGE="close", PROJECT_ROOT=root,
                         RUN_LABEL=self.report["run_label"], evidence_review=reviewed,
                         DECISION_OUTPUT_RELATIVE_PATH=Path("decision.json"),
                         report=self.report, json=json, hashlib=hashlib,
                         recommend_training_decision=decide,
                         save_json_record_once=save_once)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.close_code, scope)
                original = (root / "decision.json").read_bytes()
                exec(self.close_code, scope)
            self.assertEqual((root / "decision.json").read_bytes(), original)
            self.assertEqual(summaries, [self.report["summary"]] * 2)
            source.write_text(json.dumps(self.report) + "\n")
            with self.assertRaisesRegex(RuntimeError, "发生变化"):
                exec(self.close_code, scope)
            source.write_text(json.dumps(self.report))
            (root / "decision.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "同名决策"):
                exec(self.close_code, scope)

    def test_close_rejects_forged_summary_even_with_current_hash(self):
        """即便传入ready标记和当前哈希，关闭仍须独立复核指标。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = copy.deepcopy(self.report)
            report["summary"]["memory_improved"] = True
            source = root / "source.json"
            source.write_text(json.dumps(report))
            scope = dict(self.scope, RUN_STAGE="close", PROJECT_ROOT=root,
                         RUN_LABEL=report["run_label"], json=json, hashlib=hashlib,
                         evidence_review={"path": "source.json", "ready_for_decision": True,
                                          "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            with self.assertRaisesRegex(RuntimeError, "关闭登记前核对失败"):
                exec(self.close_code, scope)
            self.assertEqual(list(root.iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
