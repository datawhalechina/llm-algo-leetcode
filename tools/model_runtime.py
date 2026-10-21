"""Shared model download and cache helpers for the real-backend notebooks."""

from __future__ import annotations

import os
import json
import re
from pathlib import Path


def snapshot_files_complete(path: Path) -> bool:
    """Check required local assets, including every indexed weight shard."""
    def present(name):
        candidate = Path(name)
        return (not candidate.is_absolute() and '..' not in candidate.parts
                and (path / candidate).is_file() and (path / candidate).stat().st_size > 0)

    if not all(present(name) for name in ('config.json', 'tokenizer_config.json')):
        return False
    if not (present('tokenizer.json') or present('tokenizer.model') or present('spiece.model')
            or (present('vocab.json') and present('merges.txt'))):
        return False
    for index in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (path / index).exists():
            try:
                weights = json.loads((path / index).read_text())['weight_map']
                return bool(weights) and all(present(name) for name in weights.values())
            except (ValueError, KeyError, TypeError, AttributeError, OSError):
                return False
    return present('model.safetensors') or present('pytorch_model.bin')


def resolve_model_snapshot(
    model_id: str, revision: str, *, cache_dir: str | Path | None = 'auto',
    allow_download: bool = False,
) -> tuple[str, str]:
    """Resolve an exact HF commit on this host; never fall back to newest/main.

    This strict path is additive: older unpinned callers retain their behavior.
    Cache inspection neither creates directories nor contacts the network.
    """
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model_id) or '..' in model_id:
        raise ValueError('Expected a Hugging Face owner/model ID')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Frozen model revision must be an immutable 40-character commit')
    roots = model_cache_candidates(cache_dir)
    for root in roots:
        snapshot = root / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
        if snapshot_files_complete(snapshot):
            return str(snapshot.resolve()), 'local_exact_commit'
    if not allow_download:
        raise FileNotFoundError(
            f'MODEL_CACHE_MISSING: {model_id}@{revision} 未完整缓存。预检不下载；'
            '确认其余环境检查通过后，用 real_gpu 准备该固定模型，或先手动缓存相同 commit。')
    from huggingface_hub import snapshot_download

    destination = roots[0] if cache_dir and str(cache_dir).lower() != 'auto' else roots[-1]
    snapshot = Path(snapshot_download(repo_id=model_id, revision=revision, cache_dir=str(destination))).resolve()
    if snapshot.name != revision or not snapshot_files_complete(snapshot):
        raise RuntimeError('Downloaded model identity or required assets do not match the frozen commit')
    return str(snapshot), 'downloaded_exact_commit'


def model_cache_candidates(cache_dir: str | Path | None = "auto") -> list[Path]:
    """Return existing model-cache roots in deterministic lookup order."""

    candidates: list[Path] = []
    if cache_dir and str(cache_dir).lower() != "auto":
        candidates.append(Path(cache_dir).expanduser())
    candidates.append(project_root() / "model_cache")
    if os.environ.get("HF_HUB_CACHE"):
        candidates.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    candidates.append(hf_home / "hub")

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_cached_snapshot(
    model_id: str, cache_dir: str | Path | None = "auto"
) -> Path | None:
    """Find the newest complete local snapshot without downloading anything."""

    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    model_key = "models--" + model_id.replace("/", "--")
    snapshots: list[Path] = []
    for cache_root in model_cache_candidates(cache_dir):
        snapshot_root = cache_root / model_key / "snapshots"
        snapshots.extend(path for path in snapshot_root.glob("*") if path.is_dir())
    complete = [path for path in snapshots if (path / "config.json").is_file()]
    return max(complete, key=lambda path: path.stat().st_mtime) if complete else None


def project_root() -> Path:
    override = os.environ.get("LLM_ALGO_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "benchmarks").is_dir() and (candidate / "02_PyTorch_Algorithms").is_dir():
            return candidate
    return current


def resolve_model(model_id: str, source: str = "auto", cache_dir: str | Path | None = "auto") -> str:
    """Return a local model directory, downloading once when necessary.

    ``source='auto'`` reuses a local path and otherwise downloads through
    ModelScope when ``MODELSCOPE_ENDPOINT`` is set, or Hugging Face Hub.
    """
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    cached_snapshot = find_cached_snapshot(model_id, cache_dir)
    if cached_snapshot is not None:
        return str(cached_snapshot)

    root = project_root()
    if cache_dir and str(cache_dir).lower() != "auto":
        cache_path = Path(cache_dir).expanduser()
    else:
        cache_path = model_cache_candidates("auto")[-1]
    cache_path.mkdir(parents=True, exist_ok=True)

    selected = source
    if selected == "auto":
        selected = "modelscope" if os.environ.get("MODELSCOPE_ENDPOINT") else "huggingface"
    if selected == "local":
        raise FileNotFoundError(f"本地模型目录不存在: {model_id}")
    if selected == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError("请先安装 modelscope，或将 MODEL_SOURCE 改为 huggingface。") from exc
        return str(snapshot_download(model_id, cache_dir=str(cache_path)))
    if selected == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("请先安装 huggingface_hub。") from exc
        try:
            return str(snapshot_download(repo_id=model_id, cache_dir=str(cache_path)))
        except ImportError as exc:
            if "socksio" in str(exc).lower() or "socks proxy" in str(exc).lower():
                raise RuntimeError(
                    "检测到 SOCKS 代理但缺少 socksio，请运行 pip install 'httpx[socks]'，"
                    "或改用 MODEL_SOURCE='modelscope'。"
                ) from exc
            raise
    raise ValueError("MODEL_SOURCE 必须是 auto / local / huggingface / modelscope")
