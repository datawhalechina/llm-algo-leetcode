"""Static structure checks for the Part 05 integrated project notebooks."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import yaml

from tools.project_contracts import load_project_registry
from tools import inference_delivery_runtime as inference
from tools import minimind_post_training_runtime as post_training
from tools import performance_delivery_runtime as performance
from tools import part05_quantization_delivery_runtime as quantization
from tools import part05_operator_delivery_runtime as operator


ROOT = Path(__file__).resolve().parents[2]
PART = ROOT / "05_Integrated_Projects_and_Decisions"
EXPECTED = {
    "01_Inference_Service/Inference_Service_Project.ipynb": (
        "INF-INTEGRATED-DELIVERY", ["contract_check", "measure"]
    ),
    "02_Post_Training_Delivery/Post_Training_Delivery_Project.ipynb": (
        "PT-INTEGRATED-DELIVERY", ["contract_check", "prepare", "probe", "train", "close"]
    ),
    "03_Performance_Optimization_Delivery/Performance_Optimization_Delivery_Project.ipynb": (
        "PERF-INTEGRATED-DELIVERY", ["contract_check", "prepare", "measure", "close"]
    ),
    "04_Quantization_Delivery/Quantization_Delivery_Project.ipynb": (
        "QUANT-INTEGRATED-DELIVERY", ["contract_check", "prepare", "quantize", "serve", "close"]
    ),
    "05_Operator_Optimization_Delivery/Operator_Optimization_Delivery_Project.ipynb": (
        "OP-INTEGRATED-DELIVERY", ["contract_check", "measure", "close"]
    ),
}
REGISTERED_PROJECTS = {
    "INF-INTEGRATED-DELIVERY": (
        "01_Inference_Service/Inference_Service_Project.ipynb",
        inference.RESULT_SCHEMA,
    ),
    "PT-INTEGRATED-DELIVERY": (
        "02_Post_Training_Delivery/Post_Training_Delivery_Project.ipynb",
        post_training.RESULT_SCHEMA,
    ),
    "PERF-INTEGRATED-DELIVERY": (
        "03_Performance_Optimization_Delivery/Performance_Optimization_Delivery_Project.ipynb",
        performance.RESULT_SCHEMA,
    ),
    "QUANT-INTEGRATED-DELIVERY": (
        "04_Quantization_Delivery/Quantization_Delivery_Project.ipynb",
        quantization.RESULT_SCHEMA,
    ),
    "OP-INTEGRATED-DELIVERY": (
        "05_Operator_Optimization_Delivery/Operator_Optimization_Delivery_Project.ipynb",
        operator.RESULT_SCHEMA,
    ),
}


def _heading_lines(notebook: dict) -> list[str]:
    return [
        line.strip()
        for cell in notebook["cells"] if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines()
        if line.startswith("#")
    ]


def main() -> None:
    projects = load_project_registry()
    for semantic_id, (notebook_relative_path, contract) in REGISTERED_PROJECTS.items():
        project = projects[semantic_id]
        assert project["project_type"] == "integrated_delivery"
        assert project["primary_owner"] == "integrated_projects"
        assert project["target_contract"] == contract
        assert project["notebook"] == str(
            (PART / notebook_relative_path).relative_to(ROOT)
        )
    for relative_path, (semantic_id, stages) in EXPECTED.items():
        notebook_path = PART / relative_path
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        headings = _heading_lines(notebook)
        assert headings[1] == "## 本节导读", (relative_path, headings[:3])
        assert headings[-1] == "## 相关阅读", (relative_path, headings[-1:])
        step_headings = [heading for heading in headings if heading.startswith("### ")]
        assert len(step_headings) == 6
        assert all(heading.startswith(f"### {number}.") for number, heading in enumerate(step_headings, 1))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, cell in enumerate(code_cells):
            compile("".join(cell["source"]), f"{relative_path}:code-{index}", "exec")
        config_tree = ast.parse("".join(code_cells[0]["source"]))
        assert all(isinstance(node, ast.Assign) for node in config_tree.body)
        config_values = {}
        for node in config_tree.body:
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            config_values[node.targets[0].id] = ast.literal_eval(node.value)
        assignments = {
            target.id
            for node in ast.walk(ast.parse("".join(code_cells[0]["source"])))
            if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        allowed = {"RUN_LABEL", "RUN_STAGE"}
        if semantic_id == "INF-INTEGRATED-DELIVERY":
            allowed |= {"ACTIVE_CANDIDATE", "DEPLOYMENT_HANDOFF"}
        assert assignments == allowed, (relative_path, assignments)
        assert config_values["RUN_LABEL"] == "replace_with_unique_label"
        assert config_values["RUN_STAGE"] == "contract_check"
        protocol = yaml.safe_load((notebook_path.parent / "project_config.example.yaml").read_text(encoding="utf-8"))
        assert protocol["semantic_id"] == semantic_id
        if "stages" in protocol:
            assert protocol["stages"] == stages
        stage_checks = [node for cell in code_cells for node in ast.walk(ast.parse("".join(cell["source"])))
                        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                        and node.left.id == "RUN_STAGE" and isinstance(node.ops[0], ast.NotIn)]
        assert len(stage_checks) == 1
        if isinstance(stage_checks[0].comparators[0], ast.Set):
            assert ast.literal_eval(stage_checks[0].comparators[0]) == set(stages)
        else:
            assert protocol.get("stages") == stages
        assert projects[semantic_id]["notebook"] == str(notebook_path.relative_to(ROOT))
    print("Part 05 integrated project structure: ok")


class IntegratedProjectContracts(unittest.TestCase):
    """Validate source structure and actual final-report schema declarations."""

    def test_project_structure_and_registry(self):
        main()

    def test_report_writers_use_the_declared_final_schema(self):
        for module, function in ((inference, "build_delivery_record"), (post_training, "close_project"),
                                 (performance, "review_delivery"), (quantization, "decide_quantization_delivery"),
                                 (operator, "evaluate_measurements")):
            with self.subTest(module=module.__name__):
                tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
                body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
                schemas = [value for node in ast.walk(body) if isinstance(node, ast.Dict)
                           for key, value in zip(node.keys, node.values)
                           if isinstance(key, ast.Constant) and key.value == "schema_version"]
                self.assertTrue(any(isinstance(value, ast.Name) and value.id == "RESULT_SCHEMA" for value in schemas))


if __name__ == "__main__":
    unittest.main()
