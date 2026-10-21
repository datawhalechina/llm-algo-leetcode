from pathlib import Path

from tools.verl_system_mechanism_runtime import (
    VERL_COMMIT,
    build_comparisons,
    build_decisions,
    build_verl_command,
    isolated_subprocess_env,
    inspect_runtime,
    parse_console_metric_line,
    summarize_rows,
)


def test_parse_console_metric_line_ignores_noise_and_keeps_namespaces():
    assert parse_console_metric_line("ray worker ready") == {}
    row = parse_console_metric_line(
        "step:2 - rollout_corr/kl:0.012 - timing_s/gen:1.25 - perf/throughput:44"
    )
    assert row == {
        "step": 2,
        "rollout_corr/kl": 0.012,
        "timing_s/gen": 1.25,
        "perf/throughput": 44.0,
    }


def test_summary_keeps_per_step_and_aggregates_required_families():
    rows = [
        {"step": 1, "timing_s/gen": 2.0, "actor/loss": 9.0},
        {"step": 2, "timing_s/gen": 4.0, "perf/throughput": 8.0},
    ]
    summary = summarize_rows(rows)
    assert summary["step_ids"] == [1, 2]
    assert summary["mean"]["timing_s/gen"] == 3.0
    assert "actor/loss" not in summary["mean"]


def test_command_pins_single_gpu_matrix_controls():
    command = build_verl_command(
        verl_root=Path("/tmp/verl"),
        model_path="model",
        train_file=Path("/tmp/train.parquet"),
        val_file=Path("/tmp/test.parquet"),
        run_label="test",
        candidate="async_release_reject",
        total_steps=3,
    )
    assert "trainer.n_gpus_per_node=1" in command
    assert "trainer.total_training_steps=3" in command
    assert "actor_rollout_ref.rollout.mode=async" in command
    assert "actor_rollout_ref.rollout.free_cache_engine=True" in command
    assert "algorithm.rollout_correction.rollout_rs=token_k1" in command
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in command
    # Preload the base and avoid layered collection's unsupported NO_SHARD fallback.
    assert "actor_rollout_ref.rollout.load_format=safetensors" in command
    assert "actor_rollout_ref.rollout.layered_summon=False" in command
    assert "transfer_queue.backend.SimpleStorage.num_data_storage_units=1" in command


def test_subprocess_environment_keeps_http_proxy_but_drops_socks_fallback(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7891")
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:7891")

    environment = isolated_subprocess_env()

    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert "ALL_PROXY" not in environment
    assert "all_proxy" not in environment


def test_decisions_require_scoped_metric_completeness():
    base_mean = {
        "rollout_corr/kl": 0.0,
        "rollout_corr/log_ppl_abs_diff": 0.0,
        "rollout_corr/chi2_token": 0.0,
        "timing_s/gen": 1.0,
        "timing_s/update_actor": 2.0,
        "timing_s/step": 3.0,
        "perf/throughput": 4.0,
    }
    candidates = {
        "async_release_metrics": {
            "status": "ok",
            "metrics": {"mean": base_mean},
            "gpu_memory": {"peak_used_mb": 1000.0},
        },
        "async_release_reject": {
            "status": "ok",
            "metrics": {"mean": {"rollout_corr/rollout_rs_masked_fraction": 0.1}},
        },
        "async_retain_metrics": {
            "status": "ok",
            "metrics": {"mean": base_mean},
            "gpu_memory": {"peak_used_mb": 1200.0},
        },
    }
    decisions = build_decisions(candidates)
    assert decisions["rollout_lifecycle"]["decision"] == "accept_mechanism"
    assert decisions["rl_system"]["decision"] == "accept_mechanism"
    comparisons = build_comparisons(candidates)
    assert comparisons["rl_system"]["peak_used_mb"]["delta"] == 200.0


def test_embedded_probe_can_skip_dataset_files_without_weakening_default(monkeypatch, tmp_path):
    from tools import verl_system_mechanism_runtime as runtime

    monkeypatch.setattr(runtime.shutil, "which", lambda command: None)
    embedded = inspect_runtime(tmp_path, None, None, require_data_files=False)
    assert "train_file" not in embedded["required_paths"]
    assert "val_file" not in embedded["required_paths"]
    normal = inspect_runtime(tmp_path, None, None)
    assert "train_file_missing" in normal["reasons"]
    assert "val_file_missing" in normal["reasons"]
