"""Read-only Part02 project structure and saved manifest checks."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ast
import hashlib
import json

from tools.project_contracts import get_project


def _top_level_literals(source: str) -> dict[str, object]:
    """Return simple literal assignments from a notebook configuration cell."""
    values: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values

def _assert_profiling_project_step5_contract(root: Path) -> None:
    """Keep 74 as the migration reference for the six-layer project Step 5."""
    notebook_path = root / "02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells_by_id = {cell.get("id"): cell for cell in notebook["cells"]}
    expected_tags = {
        "74_profile_config_primary": "project-config",
        "74-gpu-environment-preflight": "project-preflight",
        "74-real-profile-config": "project-contract",
        "74-real-profile-run": "project-experiment",
        "74-upstream-report-code": "project-evidence",
        "74-decision-handoff": "project-decision",
    }
    for cell_id, required_tag in expected_tags.items():
        tags = cells_by_id[cell_id].get("metadata", {}).get("tags", [])
        assert required_tag in tags, (cell_id, tags)
        assert "skip_answer_test" in tags, (cell_id, tags)
        assert not any(tag.startswith("tutorial-gpu-") for tag in tags), (cell_id, tags)

    config_source = "".join(cells_by_id["74_profile_config_primary"]["source"])
    config = _top_level_literals(config_source)
    assert set(config) == {"RUN_LABEL", "RUN_STAGE"}
    assert config["RUN_LABEL"] == "replace_with_unique_label"
    assert config["RUN_STAGE"] == "prepare"

    step5_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("id") in expected_tags
    )
    assert "TRACE_SOURCE_LABEL" not in step5_source
    assert "PHASE_SOURCE_LABEL" not in step5_source
    assert "EVIDENCE_MANIFEST_RELATIVE_PATH" in step5_source
    assert "top_level_decision_scope" in step5_source
    assert "promotable_to_full_training" in step5_source
    assert "already_complete" in step5_source
    assert "OUTPUT_PATH.exists()" in step5_source
    assert "HANDOFF_PATH.exists()" in step5_source

    manifest_path = root / "benchmarks/results/74_evidence_manifest_rtx5070ti_profiling_closure_v5.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "performance-evidence-manifest/v1"
    assert manifest["semantic_id"] == "PERF-PROFILING-CLOSURE"
    entries = {
        **manifest["source_artifacts"],
        "profile": manifest["profile"]["report"],
        "phase_memory": manifest["profile"]["phase_memory"]["report"],
    }
    assert set(manifest["source_artifacts"]) == {
        "baseline", "strategy", "activation_memory_scope", "budget"
    }
    for name, entry in entries.items():
        artifact_path = root / entry["path"]
        assert artifact_path.is_file(), (name, artifact_path)
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == entry["sha256"], name

def _assert_simple_project_step5_contract(
    root: Path,
    notebook_name: str,
    cell_ids: tuple[str, str, str, str, str, str],
    allowed_stages: set[str],
) -> None:
    """Check the shared six-layer layout used by the smaller performance projects."""
    notebook_path = root / "02_PyTorch_Algorithms" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells_by_id = {cell.get("id"): cell for cell in notebook["cells"]}
    expected_tags = (
        "project-config",
        "project-preflight",
        "project-contract",
        "project-experiment",
        "project-evidence",
        "project-decision",
    )
    for cell_id, required_tag in zip(cell_ids, expected_tags, strict=True):
        cell = cells_by_id[cell_id]
        tags = cell.get("metadata", {}).get("tags", [])
        assert required_tag in tags, (cell_id, tags)
        assert "skip_answer_test" in tags, (cell_id, tags)
        assert not any(tag.startswith("tutorial-gpu-") for tag in tags), (cell_id, tags)
        ast.parse("".join(cell["source"]))

    config_source = "".join(cells_by_id[cell_ids[0]]["source"])
    config = _top_level_literals(config_source)
    assert set(config) == {"RUN_LABEL", "RUN_STAGE"}
    assert config["RUN_LABEL"] == "replace_with_unique_label"
    assert config["RUN_STAGE"] == "prepare"

    step5_source = "\n".join("".join(cells_by_id[cell_id]["source"]) for cell_id in cell_ids)
    assert "RUN_REAL_PROJECT" not in step5_source
    assert "RUN_REAL_GPU" not in step5_source
    assert all(repr(stage) in step5_source for stage in allowed_stages)
    assert "if RUN_STAGE == 'close':" in step5_source

def _assert_tagged_project_step5_contract(
    root: Path,
    notebook_name: str,
    allowed_stages: set[str],
    expected_config: set[str] | None = None,
) -> None:
    """Check six-layer projects without coupling the test to notebook cell IDs."""
    notebook_path = root / "02_PyTorch_Algorithms" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    expected_tags = (
        "project-config",
        "project-preflight",
        "project-contract",
        "project-experiment",
        "project-evidence",
        "project-decision",
    )
    cells = []
    for required_tag in expected_tags:
        matches = [
            cell
            for cell in notebook["cells"]
            if required_tag in cell.get("metadata", {}).get("tags", [])
        ]
        assert len(matches) == 1, (notebook_name, required_tag, len(matches))
        cell = matches[0]
        tags = cell.get("metadata", {}).get("tags", [])
        assert "skip_answer_test" in tags, (notebook_name, required_tag, tags)
        assert not any(tag.startswith("tutorial-gpu-") for tag in tags), tags
        ast.parse("".join(cell.get("source", [])))
        cells.append(cell)

    config = _top_level_literals("".join(cells[0]["source"]))
    if expected_config is None:
        expected_config = {"RUN_LABEL", "RUN_STAGE"}
    assert set(config) == expected_config, (notebook_name, config)
    assert config["RUN_LABEL"] == "replace_with_unique_label"
    assert config["RUN_STAGE"] == "prepare"
    step5_source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "RUN_MODE" not in step5_source, notebook_name
    assert "RUN_REAL_GPU" not in step5_source, notebook_name
    assert "RUN_REAL_PROJECT" not in step5_source, notebook_name
    assert all(
        repr(stage) in step5_source or json.dumps(stage) in step5_source
        for stage in allowed_stages
    ), notebook_name
    assert (
        "RUN_STAGE == 'close'" in step5_source
        or 'RUN_STAGE == "close"' in step5_source
    ), notebook_name

def _assert_memory_strategy_decision_scopes(root: Path) -> None:
    """Keep 76 on the six-layer contract and preserve decision scope boundaries."""
    notebook_path = root / "02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells_by_id = {cell.get("id"): cell for cell in notebook["cells"]}
    expected_tags = {
        "76_real_gpu_config_primary": "project-config",
        "76-gpu-environment-preflight": "project-preflight",
        "76_real_gpu_config": "project-contract",
        "76_real_gpu_run": "project-experiment",
        "76_result_reader_intro": "project-evidence",
        "76_decision_handoff": "project-decision",
    }
    for cell_id, required_tag in expected_tags.items():
        tags = cells_by_id[cell_id].get("metadata", {}).get("tags", [])
        assert required_tag in tags, (cell_id, tags)
        assert "skip_answer_test" in tags, (cell_id, tags)
        assert not any(tag.startswith("tutorial-gpu-") for tag in tags), (cell_id, tags)

    config_source = "".join(cells_by_id["76_real_gpu_config_primary"]["source"])
    config = _top_level_literals(config_source)
    assert set(config) == {"RUN_LABEL", "RUN_STAGE"}
    assert config["RUN_LABEL"] == "replace_with_unique_label"
    assert config["RUN_STAGE"] == "prepare"

    step5_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("id") in expected_tags
    )
    assert "RUN_REAL_GPU" not in step5_source
    assert "{'prepare', 'g0', 'g1', 'close'}" in step5_source
    assert "G1 要求同一 RUN_LABEL 且口径匹配的 G0" in step5_source
    assert "g0_incomplete_repeats" in step5_source
    assert "if RUN_STAGE == 'close':" in step5_source
    assert "source_baseline_sha256" in step5_source

    decision_cell = next(
        cell for cell in notebook["cells"] if cell.get("id") == "76_decision_handoff"
    )
    source = "".join(decision_cell["source"])
    assert "top_level_decision_scope" in source
    assert "full_training_adoption" in source
    assert "activation_memory_mechanism" in source
    assert "promotable_to_full_training" in source
    assert "source_g1_sha256" in source
    assert "budget_sensitivity" in source
    assert "75_memory_budget_compression" not in source


class Part02ProjectStructureTests(unittest.TestCase):
    """Inspect source cells and saved evidence without executing experiments."""

    def test_registered_special_contracts(self):
        """Check scope-specific project registrations."""
        assert get_project("PERF-PROFILING-CLOSURE")["additional_contracts"] == ["performance-optimization-handoff/v1"]
        assert get_project("PERF-OPTIMIZER-PATH")["project_type"] == "benchmark"
        assert get_project("DISTILL-STUDENT-DELIVERY")["target_contract"] == "distilled-student-delivery/v3"

    def test_training_baseline_layout(self):
        """Check the baseline project six-layer layout."""
        root = ROOT
        _assert_simple_project_step5_contract(
            root,
            "73_Training_Performance_Analysis.ipynb",
            (
                "73_real_gpu_config_primary",
                "73_colab_runtime_bootstrap",
                "73_real_gpu_config",
                "73_real_gpu_run",
                "73_result_reader",
                "73_decision_handoff",
            ),
            {"prepare", "measure", "close"},
        )

    def test_optimizer_layout(self):
        """Check the optimizer project six-layer layout."""
        root = ROOT
        _assert_simple_project_step5_contract(
            root,
            "75_Optimizer_Execution_Path_Benchmark.ipynb",
            (
                "config",
                "preflight",
                "contract",
                "experiment",
                "evidence",
                "decision",
            ),
            {"prepare", "measure", "close"},
        )

    def test_profiling_layout_and_manifest(self):
        """Check profiling cells and every saved source SHA256."""
        root = ROOT
        _assert_profiling_project_step5_contract(root)

    def test_memory_strategy_scopes(self):
        """Keep activation-memory and full-training decisions separate."""
        root = ROOT
        _assert_memory_strategy_decision_scopes(root)

    def test_tagged_project_layouts(self):
        """Check remaining tagged project stages and configurations."""
        root = ROOT
        for notebook_name, allowed_stages, expected_config in (
            ("66_Inference_Performance_Comparison.ipynb", {"prepare", "measure", "read", "close"}, None),
            ("60_Decoder_Block_Stability_Benchmark.ipynb", {"prepare", "measure", "close"}, None),
            ("78_Attention_Access_Pattern_Benchmark.ipynb", {"prepare", "probe", "measure", "close"}, None),
            ("71_MLA_KV_Cache_Architecture_Benchmark.ipynb", {"prepare", "measure", "close"}, None),
            ("77_Long_Sequence_Memory_Architecture_Benchmark.ipynb", {"prepare", "measure", "close"}, None),
            ("61_Model_Architecture_Exploration.ipynb", {"prepare", "audit", "close"}, None),
            (
                "82_Quantization_Artifact_Evaluation_Project.ipynb",
                {"prepare", "measure", "close"},
                {"RUN_LABEL", "RUN_STAGE", "UPSTREAM_RUN_LABEL"},
            ),
            ("67_Quantized_Inference_and_Deployment.ipynb", {"prepare", "measure", "close"}, {"RUN_LABEL", "RUN_STAGE", "UPSTREAM_RUN_LABEL"}),
            ("83_Quantized_Serving_Profile_Benchmark.ipynb", {"prepare", "measure", "close"}, {"RUN_LABEL", "RUN_STAGE", "UPSTREAM_RUN_LABEL"}),
            ("88_Distilled_Student_Delivery_Project.ipynb", {"prepare", "preflight", "g0", "g1", "g2", "close"}, None),
        ):
            with self.subTest(notebook=notebook_name):
                _assert_tagged_project_step5_contract(
                    root, notebook_name, allowed_stages, expected_config
                )


if __name__ == "__main__":
    unittest.main()
