"""Shared evidence helpers for quantization mechanism notebooks.

Only repetitive measurement and artifact bookkeeping belongs here.  The
quantization algorithm, candidate construction and acceptance policy stay in
the teaching notebook so learners can inspect them directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_BUNDLE_SCHEMA_VERSION = "quantization-artifact-candidate/v1"


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def build_candidate_bundle(
    mechanism_result: dict[str, Any],
    *,
    candidate_name: str,
    artifact_dir: str | Path,
    project_root: str | Path,
    source_result_path: str | Path,
    producer_semantic_id: str = "QUANT-WEIGHT-PTQ",
) -> dict[str, Any]:
    """Create one portable handoff bundle from a mechanism-level PTQ result."""

    candidates = mechanism_result.get("candidates", {})
    if candidate_name not in candidates:
        raise KeyError(f"candidate not found: {candidate_name}")
    candidate = candidates[candidate_name]
    baseline = mechanism_result.get("baseline", {})
    workload = mechanism_result.get("workload", {})
    dataset = mechanism_result.get("dataset", {})
    manifest = candidate.get("artifact_manifest", {})
    quant_config = manifest.get("quantization_config") or {}
    config_groups = quant_config.get("config_groups") or {}
    first_group = next(iter(config_groups.values()), {})
    weight_config = first_group.get("weights") or {}
    root = Path(project_root).resolve()
    artifact = Path(artifact_dir).resolve()
    if not artifact.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {artifact}")

    inventory = []
    for item in sorted(path for path in artifact.rglob("*") if path.is_file()):
        if item.name == "candidate_bundle.json":
            continue
        inventory.append({
            "path": str(item.relative_to(artifact)),
            "bytes": item.stat().st_size,
            "sha256": _sha256_file(item),
        })
    weight_files = [item["path"] for item in inventory if item["path"].endswith((".safetensors", ".bin"))]
    tokenizer_files = [item["path"] for item in inventory if item["path"].startswith("tokenizer")]
    model_revision = workload.get("resolved_model_commit") or workload.get("loader_revision")
    artifact_seed = json.dumps(
        {
            "model": workload.get("model_id"),
            "revision": model_revision,
            "candidate": candidate_name,
            "bits": weight_config.get("num_bits"),
            "group_size": weight_config.get("group_size"),
        },
        sort_keys=True,
    ).encode("utf-8")
    artifact_id = f"{candidate_name}-{hashlib.sha256(artifact_seed).hexdigest()[:16]}"
    ratio = candidate.get("nll_ratio_vs_bf16")
    max_ratio = workload.get("max_nll_ratio_vs_bf16")
    quality_passed = ratio is not None and max_ratio is not None and float(ratio) <= float(max_ratio)
    return {
        "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "artifact_path": _portable_path(artifact, root),
        "model": {"id": workload.get("model_id"), "revision": model_revision},
        "quantization": {
            "method": candidate_name,
            "format": quant_config.get("format"),
            "bits": weight_config.get("num_bits"),
            "group_size": weight_config.get("group_size"),
            "scheme": workload.get("scheme"),
            "tool": quant_config.get("quant_method"),
            "tool_version": quant_config.get("version"),
        },
        "calibration": {
            "dataset": dataset.get("dataset_id"),
            "config": dataset.get("config"),
            "revision": dataset.get("revision"),
            "split": dataset.get("calibration_split"),
            "sample_count": dataset.get("calibration_samples"),
            "sample_ids_hash": dataset.get("calibration_ids_hash"),
        },
        "evaluation": {
            "dataset": dataset.get("dataset_id"),
            "config": dataset.get("config"),
            "revision": dataset.get("revision"),
            "split": dataset.get("evaluation_split"),
            "sample_count": dataset.get("evaluation_samples"),
            "sample_ids_hash": dataset.get("evaluation_ids_hash"),
            "metric_name": "token_weighted_nll",
            "direction": "lower_is_better",
            "baseline_value": baseline.get("token_weighted_nll"),
            "candidate_value": candidate.get("token_weighted_nll"),
            "ratio_vs_baseline": ratio,
            "max_degradation_ratio": max_ratio,
            "passed": quality_passed,
        },
        "files": {
            "weights": weight_files,
            "config": "config.json" if (artifact / "config.json").is_file() else None,
            "tokenizer": tokenizer_files,
            "inventory": inventory,
        },
        "evidence": {
            "evidence_level": mechanism_result.get("evidence_level"),
            "failure": mechanism_result.get("failure"),
            "producer_semantic_id": producer_semantic_id,
            "source_result_path": _portable_path(Path(source_result_path), root),
        },
    }


def save_candidate_bundle(path: str | Path, bundle: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def validate_candidate_bundle(
    bundle: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    """Validate identity, quality direction, data isolation and file hashes."""

    issues: list[str] = []
    if not isinstance(bundle, dict):
        return {"ready": False, "issues": ["candidate bundle must be a mapping"], "artifact_path": None}
    from tools.quantization_project_runtime import validate_candidate_contract, evaluate_candidate_evidence

    for label, check in (("contract", validate_candidate_contract(bundle)),
                         ("evidence", evaluate_candidate_evidence(bundle))):
        issues.extend(f"{label}: {issue}" for issue in check["issues"])
    if bundle.get("schema_version") != CANDIDATE_BUNDLE_SCHEMA_VERSION:
        issues.append(f"schema_version must be {CANDIDATE_BUNDLE_SCHEMA_VERSION}")
    for section, fields in {
        "model": ("id", "revision"),
        "quantization": ("method", "format", "bits", "group_size"),
        "calibration": ("dataset", "revision", "split", "sample_count", "sample_ids_hash"),
        "evaluation": (
            "dataset", "revision", "split", "sample_ids_hash", "metric_name",
            "direction", "baseline_value", "candidate_value", "ratio_vs_baseline",
            "passed",
        ),
        "files": ("weights", "config", "tokenizer", "inventory"),
        "evidence": ("evidence_level", "producer_semantic_id", "source_result_path"),
    }.items():
        values = bundle.get(section)
        if not isinstance(values, dict):
            issues.append(f"{section} must be a mapping")
            continue
        for field in fields:
            if field not in values or values[field] in (None, "", []):
                issues.append(f"missing {section}.{field}")

    if any(not isinstance(bundle.get(key), dict) for key in ("model", "quantization", "calibration", "evaluation", "files", "evidence")):
        return {"ready": False, "issues": issues, "artifact_path": None}
    calibration = bundle.get("calibration") or {}
    evaluation = bundle.get("evaluation") or {}
    if calibration.get("dataset") != evaluation.get("dataset") or calibration.get("revision") != evaluation.get("revision"):
        issues.append("calibration and evaluation must use the registered dataset snapshot")
    if calibration.get("split") == evaluation.get("split"):
        issues.append("calibration and evaluation splits must differ")
    if calibration.get("sample_ids_hash") == evaluation.get("sample_ids_hash"):
        issues.append("calibration and evaluation sample hashes must differ")
    if evaluation.get("direction") not in {"lower_is_better", "higher_is_better"}:
        issues.append("evaluation.direction is invalid")
    if evaluation.get("passed") is not True:
        issues.append("candidate quality gate did not pass")
    if bundle["evidence"].get("failure"):
        issues.append("source candidate records a failure")

    root = Path(project_root).resolve()
    artifact_name = bundle.get("artifact_path")
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        issues.append("artifact_path must name a directory")
        return {"ready": False, "issues": issues, "artifact_path": None}
    artifact = Path(artifact_name)
    if not artifact.is_absolute():
        artifact = root / artifact
    if not artifact.is_dir():
        issues.append(f"artifact directory not found: {artifact}")
    else:
        inventory = bundle["files"].get("inventory")
        for item in inventory if isinstance(inventory, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                issues.append("invalid inventory entry")
                continue
            file_path = (artifact / item["path"]).resolve()
            try:
                file_path.relative_to(artifact.resolve())
            except ValueError:
                issues.append("inventory path escapes artifact directory")
                continue
            if not file_path.is_file():
                issues.append(f"artifact file missing: {file_path}")
                continue
            if type(item.get("bytes")) is not int or item["bytes"] < 0:
                issues.append(f"artifact byte count is invalid: {file_path}")
            elif file_path.stat().st_size != item.get("bytes"):
                issues.append(f"artifact size mismatch: {file_path}")
            elif _sha256_file(file_path) != item.get("sha256"):
                issues.append(f"artifact hash mismatch: {file_path}")
    return {"ready": not issues, "issues": issues, "artifact_path": str(artifact)}


def token_prefix_agreement(reference: Iterable[Any], candidate: Iterable[Any]) -> float:
    """Return the matching greedy prefix ratio of two token sequences."""

    left = list(reference)
    right = list(candidate)
    width = max(1, min(len(left), len(right)))
    matched = 0
    for reference_token, candidate_token in zip(left, right):
        if reference_token != candidate_token:
            break
        matched += 1
    return matched / width


def mean_prefix_agreement(reference_outputs, candidate_outputs) -> float:
    """Aggregate greedy-prefix agreement over paired generated outputs."""

    scores = [
        token_prefix_agreement(reference, candidate)
        for reference, candidate in zip(reference_outputs, candidate_outputs)
    ]
    return sum(scores) / max(1, len(scores))


def artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Describe files, bytes and quantization metadata in one artifact directory."""

    root = Path(path)
    files = [item for item in sorted(root.rglob("*")) if item.is_file()]
    config_path = root / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    return {
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "files": [
            {"path": str(item.relative_to(root)), "bytes": item.stat().st_size}
            for item in files
        ],
        "quantization_config": config.get("quantization_config"),
        "has_config": config_path.exists(),
        "has_weight_file": any(
            item.suffix == ".safetensors" or item.name.endswith(".bin")
            for item in files
        ),
    }


def tensor_storage_components(tensor, prefix: str = "weight", seen_storages=None):
    """Flatten tensor subclasses and report unique physical storage components."""

    seen = seen_storages if seen_storages is not None else set()
    if hasattr(tensor, "__tensor_flatten__"):
        names, _ = tensor.__tensor_flatten__()
        components = []
        for name in names:
            components.extend(
                tensor_storage_components(getattr(tensor, name), f"{prefix}.{name}", seen)
            )
        return components

    storage = tensor.untyped_storage()
    key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
    if key in seen:
        return []
    seen.add(key)
    return [{
        "name": prefix,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "storage_bytes": int(storage.nbytes()),
    }]


def tensor_storage_bytes(tensor, seen_storages=None) -> int:
    """Return deduplicated bytes for a native tensor or tensor subclass."""

    return sum(
        item["storage_bytes"]
        for item in tensor_storage_components(
            tensor,
            prefix="tensor",
            seen_storages=seen_storages,
        )
    )


def state_bytes(module) -> int:
    """Return deduplicated physical bytes referenced by a module state dict."""

    seen_storages = set()
    return sum(
        tensor_storage_bytes(value, seen_storages)
        for value in module.state_dict().values()
    )


def model_storage_manifest(model) -> dict[str, Any]:
    """Summarize unique parameter and buffer storage by dtype."""

    seen_storages = set()
    components = []
    for name, tensor in model.named_parameters(recurse=True, remove_duplicate=True):
        components.extend(tensor_storage_components(tensor, name, seen_storages))
    for name, tensor in model.named_buffers(recurse=True, remove_duplicate=True):
        components.extend(tensor_storage_components(tensor, name, seen_storages))

    by_dtype: dict[str, int] = {}
    for component in components:
        dtype = component["dtype"]
        by_dtype[dtype] = by_dtype.get(dtype, 0) + component["storage_bytes"]
    return {
        "physical_storage_bytes": sum(item["storage_bytes"] for item in components),
        "component_count": len(components),
        "bytes_by_dtype": dict(sorted(by_dtype.items())),
    }
