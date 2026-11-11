"""Machine-readable producer/consumer relationships for Part 02 projects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .project_contracts import load_project_registry


RELATION_SCHEMA_VERSION = "project-asset-relations/v1"
REFERENCE_SCHEMA_VERSION = "project-asset-reference/v1"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_relations_path() -> Path:
    return repository_root() / "benchmarks" / "project_asset_relations.json"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lookup(payload: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def load_asset_relations(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    relation_path = Path(path) if path is not None else default_relations_path()
    payload = _read_json(relation_path)
    if payload.get("schema_version") != RELATION_SCHEMA_VERSION:
        raise ValueError(f"未知资产关系版本: {payload.get('schema_version')!r}")
    registry = load_project_registry()
    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("relations", []):
        asset_id = item.get("asset_id") if isinstance(item, Mapping) else None
        if not asset_id or asset_id in result:
            raise ValueError(f"资产关系存在空或重复 asset_id: {asset_id!r}")
        producer = item.get("producer")
        consumers = item.get("consumers")
        if producer not in registry:
            raise ValueError(f"{asset_id} 的 producer 未登记: {producer!r}")
        if not isinstance(consumers, list) or not consumers:
            raise ValueError(f"{asset_id} 至少需要一个 consumer")
        unknown = sorted(set(consumers) - set(registry))
        if unknown:
            raise ValueError(f"{asset_id} 包含未登记 consumer: {unknown}")
        if not item.get("contract") or not isinstance(item.get("required_keys"), list):
            raise ValueError(f"{asset_id} 缺少 contract 或 required_keys")
        registered_contracts = {
            registry[producer].get("target_contract"),
            *registry[producer].get("additional_contracts", []),
        }
        if item.get("contract") not in registered_contracts:
            raise ValueError(
                f"{asset_id} 的 contract={item.get('contract')!r}，"
                f"不在 producer 注册契约 {sorted(registered_contracts)} 中"
            )
        result[str(asset_id)] = dict(item)
    return result


def get_asset_relation(asset_id: str) -> dict[str, Any]:
    try:
        return load_asset_relations()[asset_id]
    except KeyError as exc:
        raise KeyError(f"未登记的项目资产: {asset_id}") from exc


def validate_asset_payload(
    asset_id: str,
    payload: Mapping[str, Any],
    *,
    consumer: str | None = None,
) -> list[str]:
    relation = get_asset_relation(asset_id)
    errors: list[str] = []
    if consumer is not None and consumer not in relation["consumers"]:
        errors.append(f"{consumer} 不是 {asset_id} 的登记 consumer")
    if payload.get("schema_version") != relation["contract"]:
        errors.append(
            f"schema_version={payload.get('schema_version')!r}，期望 {relation['contract']!r}"
        )
    semantic_id = payload.get("semantic_id")
    if semantic_id is not None and semantic_id != relation["producer"]:
        errors.append(f"semantic_id={semantic_id!r}，期望 {relation['producer']!r}")
    for key in relation["required_keys"]:
        try:
            value = _lookup(payload, key)
        except KeyError:
            errors.append(f"缺少字段 {key}")
            continue
        if value is None or value == "":
            errors.append(f"字段 {key} 为空")
    return errors


def make_asset_reference(asset_id: str, result_path: str | Path) -> dict[str, Any]:
    path = Path(result_path).resolve()
    payload = _read_json(path)
    errors = validate_asset_payload(asset_id, payload)
    if errors:
        raise ValueError(f"{asset_id} 结果不符合契约: {'; '.join(errors)}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    relation = get_asset_relation(asset_id)
    try:
        stored_path = str(path.relative_to(repository_root()))
        path_kind = "repository_relative"
    except ValueError:
        stored_path = str(path)
        path_kind = "absolute"
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "asset_id": asset_id,
        "producer": relation["producer"],
        "contract": relation["contract"],
        "result_path": stored_path,
        "path_kind": path_kind,
        "sha256": digest,
    }


def load_asset_reference(reference: Mapping[str, Any], *, consumer: str) -> dict[str, Any]:
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError("未知项目资产引用版本")
    asset_id = str(reference.get("asset_id", ""))
    relation = get_asset_relation(asset_id)
    if consumer not in relation["consumers"]:
        raise ValueError(f"{consumer} 不是 {asset_id} 的登记 consumer")
    path = Path(str(reference.get("result_path", "")))
    if reference.get("path_kind") == "repository_relative" or (
        "path_kind" not in reference and not path.is_absolute()
    ):
        path = repository_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"项目资产结果不存在: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != reference.get("sha256"):
        raise ValueError(f"项目资产哈希不匹配: {asset_id}")
    payload = _read_json(path)
    errors = validate_asset_payload(asset_id, payload, consumer=consumer)
    if errors:
        raise ValueError(f"{asset_id} 结果不符合契约: {'; '.join(errors)}")
    return payload
