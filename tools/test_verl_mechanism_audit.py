from tools.verl_mechanism_audit import audit_history, inspect_api, inspect_symbol


def test_symbol_lookup_is_exact_and_handles_async_methods():
    source = "class Worker:\n    async def update(self, data):\n        return data\n"
    found = inspect_symbol(source, "Worker.update")
    assert found["found"] and found["async"]
    assert found["parameters"] == ["self", "data"]
    assert len(found["source_sha256"]) == 64
    assert not inspect_symbol(source, "Other.update")["found"]
    assert not inspect_symbol(source, "Worker.missing")["found"]


def test_missing_interfaces_do_not_authorize_a_gpu_recipe(tmp_path):
    result = inspect_api(tmp_path)
    assert not result["inspection_passed"]
    assert not result["source_identity_matched"]
    assert not result["gpu_recipe_ready"]
    assert result["evidence_level"] == "source_interface_inspection"


def test_empty_history_is_not_a_pass():
    result = audit_history({})
    assert result["rollout_lifecycle"]["status"] == "incomplete"
    assert result["rl_system"]["status"] == "incomplete"


def test_complete_old_metrics_remain_partial_evidence():
    metrics = {"mean": {
        "rollout_corr/kl": 0.0, "rollout_corr/log_ppl_abs_diff": 0.003,
        "rollout_corr/chi2_token": 0.0, "rollout_corr/rollout_rs_masked_fraction": 0.08,
        "timing_s/gen": 1.0, "timing_s/update_actor": 2.0, "timing_s/step": 4.0,
    }}
    candidate = {"status": "ok", "metrics": metrics, "gpu_memory": {"peak_used_mb": 1000}}
    record = {"candidates": {name: candidate for name in (
        "async_release_metrics", "async_release_reject", "async_retain_metrics",
    )}}
    result = audit_history(record)
    assert result["rollout_lifecycle"]["status"] == "partial"
    assert result["rl_system"]["status"] == "partial"
    assert "真实参数变化" in result["rollout_lifecycle"]["not_proven"]
    assert "逐阶段状态驻留" in result["rl_system"]["not_proven"]
