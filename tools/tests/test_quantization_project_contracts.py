"""CPU-only quantization artifact and serving decision contracts."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quantization_project_runtime import (
    decide_artifact_adoption, select_serving_candidate, validate_candidate_contract, validate_serving_profile,
)


class QuantizationProjectContractTests(unittest.TestCase):
    """Exercise fixed artifact and serving fixtures without loading a model."""

    def test_artifact_adoption_contract(self):
        """Validate a local artifact declaration and quality gate."""
        candidate = {
            "schema_version": "quantization-artifact-candidate/v1",
            "artifact_id": "candidate-1",
            "model": {"id": "model", "revision": "revision"},
            "quantization": {"method": "gptq", "format": "w4a16", "bits": 4, "group_size": 128},
            "files": {
                "weights": ["model.safetensors"],
                "config": "config.json",
                "tokenizer": ["tokenizer.json"],
                "inventory": [{"path": path, "sha256": "a" * 64} for path in ("model.safetensors", "config.json", "tokenizer.json")],
            },
        }
        assert validate_candidate_contract(candidate)["ready"] is True
        assert decide_artifact_adoption(
            {"ready": True, "issues": []},
            {"ready": True, "issues": []},
            {
                "load_status": "loaded",
                "failure": None,
                "roundtrip_quality": {"prompt_count": 2, "non_empty_generations": 2},
                "peak_memory_mb": 100.0,
            },
            max_memory_mb=200.0,
        )["decision"] == "accept"

    def test_serving_selection_contract(self):
        """Validate the serving profile and measured candidate selection."""
        profile = {
            "name": "latency",
            "workload": {
                "id": "smoke",
                "num_prompts": 1,
                "concurrency": 1,
                "max_tokens": 8,
                "repeats": 1,
            },
            "primary_metric": "ttft_p95_ms",
            "slo": {"ttft_p95_ms": {"direction": "max", "limit": 100.0}},
        }
        assert validate_serving_profile(profile)["ready"] is True
        assert select_serving_candidate(
            [
                {
                    "name": "candidate",
                    "status": "ok",
                    "artifact_id": "artifact",
                    "profile_name": "latency",
                    "workload_id": "smoke",
                    "evidence_level": "real_project_benchmark",
                    "metrics": {"ttft_p95_ms": 50.0},
                }
            ],
            profile,
            expected_artifact_id="artifact",
        )["decision"] == "accept"


if __name__ == "__main__":
    unittest.main()
