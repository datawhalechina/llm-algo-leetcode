"""Pinned model lookup tests; all download calls are replaced with fakes."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools import model_runtime as runtime

COMMIT = 'a' * 40
MODEL = 'owner/model'


def snapshot(root, commit=COMMIT, *, sharded=False):
    path = root / 'models--owner--model' / 'snapshots' / commit
    path.mkdir(parents=True)
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (path / name).write_text('{}')
    if sharded:
        (path / 'model.safetensors.index.json').write_text(json.dumps(
            {'weight_map': {'a':'part1.safetensors', 'b':'part2.safetensors'}}))
        for name in ('part1.safetensors', 'part2.safetensors'):
            (path / name).write_bytes(b'fixture')
    else:
        (path / 'model.safetensors').write_bytes(b'fixture')
    return path


def test_exact_commit_reused_even_with_newer_alternative(tmp_path, monkeypatch):
    expected = snapshot(tmp_path)
    snapshot(tmp_path, 'b' * 40)
    monkeypatch.setattr(runtime, 'model_cache_candidates', lambda cache: [tmp_path])
    path, origin = runtime.resolve_model_snapshot(MODEL, COMMIT)
    assert Path(path) == expected and origin == 'local_exact_commit'


def test_multiple_cache_roots_and_partial_snapshot(tmp_path, monkeypatch):
    first, second = tmp_path/'first', tmp_path/'second'
    incomplete = snapshot(first)
    (incomplete/'tokenizer.json').unlink()
    expected = snapshot(second)
    monkeypatch.setattr(runtime, 'model_cache_candidates', lambda cache: [first, second])
    assert Path(runtime.resolve_model_snapshot(MODEL, COMMIT)[0]) == expected


def test_offline_miss_has_no_network_or_directory_creation(tmp_path, monkeypatch):
    missing = tmp_path/'absent'
    monkeypatch.setattr(runtime, 'model_cache_candidates', lambda cache: [missing])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kwargs: pytest.fail('Unexpected network call')))
    with pytest.raises(FileNotFoundError, match='MODEL_CACHE_MISSING'):
        runtime.resolve_model_snapshot(MODEL, COMMIT)
    assert not missing.exists()


def test_other_commit_is_not_a_fallback(tmp_path, monkeypatch):
    snapshot(tmp_path, 'b' * 40)
    monkeypatch.setattr(runtime, 'model_cache_candidates', lambda cache: [tmp_path])
    with pytest.raises(FileNotFoundError):
        runtime.resolve_model_snapshot(MODEL, COMMIT)


@pytest.mark.parametrize('revision', ['main', '../escape', 'short', 'B'*40])
def test_only_immutable_commits_accepted(revision):
    with pytest.raises(ValueError, match='immutable'):
        runtime.resolve_model_snapshot(MODEL, revision)


def test_download_is_explicit_and_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'model_cache_candidates', lambda cache: [tmp_path])
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot(tmp_path))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    path, origin = runtime.resolve_model_snapshot(MODEL, COMMIT, cache_dir=tmp_path, allow_download=True)
    assert origin == 'downloaded_exact_commit' and Path(path).name == COMMIT
    assert calls == [{'repo_id':MODEL, 'revision':COMMIT, 'cache_dir':str(tmp_path)}]


def test_wrong_download_revision_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'model_cache_candidates', lambda cache: [tmp_path])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        snapshot_download=lambda **kwargs: str(snapshot(tmp_path, 'b'*40))))
    with pytest.raises(RuntimeError, match='identity'):
        runtime.resolve_model_snapshot(MODEL, COMMIT, allow_download=True)


def test_missing_weight_shard_or_invalid_index_is_incomplete(tmp_path):
    path = snapshot(tmp_path, sharded=True)
    assert runtime.snapshot_files_complete(path)
    (path/'part2.safetensors').unlink()
    assert not runtime.snapshot_files_complete(path)
    (path/'model.safetensors.index.json').write_text('not-json')
    assert not runtime.snapshot_files_complete(path)


def test_cache_priority_respects_project_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'project_root', lambda: tmp_path/'repo')
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path/'hub_override'))
    monkeypatch.setenv('HF_HOME', str(tmp_path/'hf_home'))
    assert runtime.model_cache_candidates(tmp_path/'explicit') == [
        tmp_path/'explicit', tmp_path/'repo/model_cache', tmp_path/'hub_override', tmp_path/'hf_home/hub']
