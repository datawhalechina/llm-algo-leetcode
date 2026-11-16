"""Check immutable result writes using temporary files only."""

import json

import pytest

from tools.project_runtime import save_json_record_once


def test_save_json_record_once_creates_parent_and_record(tmp_path):
    output = tmp_path / "nested" / "result.json"

    saved = save_json_record_once(output, {"run_label": "smoke-v1"})

    assert saved == output
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "run_label": "smoke-v1"
    }


def test_save_json_record_once_rejects_existing_result(tmp_path):
    output = tmp_path / "result.json"
    save_json_record_once(output, {"status": "original"})

    with pytest.raises(FileExistsError, match="RUN_LABEL"):
        save_json_record_once(output, {"status": "replacement"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "original"
    }
