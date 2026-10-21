"""Run the shared single-GPU verl experiment used by two Part 02 mechanism pages.

The runner deliberately uses verl's committed uv lock instead of importing verl into
the notebook kernel.  It records the exact command, raw console log, parsed metrics,
and two scoped decisions: rollout lifecycle evidence and colocated-system evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


VERL_TAG = "v0.9.1"
VERL_COMMIT = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
UV_EXTRAS = ("vllm",)
METRIC_PREFIXES = ("rollout_corr/", "timing_s/", "perf/", "actor/perf/")


def isolated_subprocess_env() -> dict[str, str]:
    """Keep unrelated Python paths and an optional SOCKS fallback out of workers."""

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = ""
    # The workstation exposes both HTTP(S) and SOCKS proxies.  verl's locked
    # environment supports the HTTP(S) route, while HTTPX requires the optional
    # ``socksio`` package as soon as ALL_PROXY points at SOCKS.  Keep the usable
    # HTTP(S) proxy and remove only the unsupported fallback.
    environment.pop("ALL_PROXY", None)
    environment.pop("all_proxy", None)
    return environment


def parse_console_metric_line(line: str) -> dict[str, float | int]:
    """Parse one official ``step:N - key:value`` console line."""

    match = re.search(r"(?:^|\s)step:(\d+)(?:\s+-\s+|$)", line)
    if not match:
        return {}
    row: dict[str, float | int] = {"step": int(match.group(1))}
    suffix = line[match.end() :]
    for field in suffix.split(" - "):
        if ":" not in field:
            continue
        key, raw_value = field.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            row[key] = value
    return row


def parse_console_metrics(lines: Iterable[str]) -> list[dict[str, float | int]]:
    """Return metric rows while ignoring Ray and dependency log noise."""

    return [row for line in lines if (row := parse_console_metric_line(line))]


def summarize_rows(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    """Aggregate stable metric families without hiding per-step evidence."""

    keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key != "step" and key.startswith(METRIC_PREFIXES)
        }
    )
    averages = {
        key: mean(float(row[key]) for row in rows if key in row)
        for key in keys
    }
    return {
        "steps_observed": len(rows),
        "step_ids": [int(row["step"]) for row in rows],
        "mean": averages,
        "last": rows[-1] if rows else {},
    }


def _uv_python_command() -> list[str]:
    command = ["uv", "run", "--frozen", "--all-packages"]
    for extra in UV_EXTRAS:
        command.extend(["--extra", extra])
    command.append("python3")
    return command


def build_verl_command(
    *,
    verl_root: Path,
    model_path: str,
    train_file: Path,
    val_file: Path,
    run_label: str,
    candidate: str,
    total_steps: int,
) -> list[str]:
    """Build the pinned v0.9.1 one-GPU GRPO command."""

    candidate_overrides = {
        "async_release_metrics": [
            "actor_rollout_ref.rollout.free_cache_engine=True",
            "algorithm.rollout_correction.rollout_rs=null",
            "algorithm.rollout_correction.rollout_rs_threshold=null",
        ],
        "async_release_reject": [
            "actor_rollout_ref.rollout.free_cache_engine=True",
            "algorithm.rollout_correction.rollout_rs=token_k1",
            "algorithm.rollout_correction.rollout_rs_threshold=0.95_1.05",
        ],
        "async_retain_metrics": [
            "actor_rollout_ref.rollout.free_cache_engine=False",
            "algorithm.rollout_correction.rollout_rs=null",
            "algorithm.rollout_correction.rollout_rs_threshold=null",
        ],
    }
    if candidate not in candidate_overrides:
        raise ValueError(f"unknown candidate: {candidate}")

    worker_python = "uv -v run --frozen --all-packages --extra vllm"
    return [
        *_uv_python_command(),
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.train_batch_size=4",
        "data.max_prompt_length=256",
        "data.max_response_length=128",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.dataloader_num_workers=0",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        "algorithm.rollout_correction.rollout_is_batch_normalize=False",
        "algorithm.rollout_correction.bypass_mode=False",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.lora_rank=16",
        "actor_rollout_ref.model.lora_alpha=16",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        # Preload the frozen base so subsequent updates contain only adapters.
        # FSDP1 on one GPU uses NO_SHARD; layered collection can fall back to
        # summon_full_params(offload_to_cpu=True), unsupported in that mode.
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=False",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        # Four prompts with four responses need at most sixteen live sequences.
        "actor_rollout_ref.rollout.max_model_len=512",
        "actor_rollout_ref.rollout.max_num_seqs=16",
        "actor_rollout_ref.rollout.max_num_batched_tokens=512",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "critic.enable=False",
        "reward.num_workers=1",
        # V1 enables TransferQueue internally; its eight-unit default is for
        # distributed workloads and launches unnecessary actors on this laptop.
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
        "trainer.logger=console",
        "trainer.project_name=llm_algo_mechanism",
        f"trainer.experiment_name={run_label}_{candidate}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
        f"trainer.total_training_steps={total_steps}",
        "+ray_kwargs.ray_init.object_store_memory=536870912",
        f"ray_kwargs.ray_init.runtime_env.py_executable={worker_python}",
        *candidate_overrides[candidate],
    ]


def inspect_runtime(
    verl_root: Path, train_file: Path | None, val_file: Path | None,
    *, require_data_files: bool = True,
) -> dict[str, Any]:
    """Check immutable source identity and local prerequisites without installing."""

    reasons: list[str] = []
    host_memory = {}
    if Path("/proc/meminfo").is_file():
        fields = dict(
            line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
        )
        host_memory = {
            name: round(int(fields[key].split()[0]) / 1024**2, 3)
            for name, key in (("total_gib", "MemTotal"), ("available_gib", "MemAvailable"))
        }
        # An admission guard for this small single-card experiment, not a
        # prediction of its peak RSS. Keep Ray's runtime memory monitor enabled.
        # v7 consumed roughly 18 GiB beyond its starting host usage. Reserve
        # additional headroom at launch; this remains a guard, not a peak bound.
        host_memory["minimum_available_gib"] = 20.0
        if host_memory["available_gib"] < host_memory["minimum_available_gib"]:
            reasons.append("insufficient_available_host_memory")
    uv = shutil.which("uv")
    if uv is None:
        reasons.append("uv_not_found")
    external_python = verl_root / ".venv/bin/python"
    required = {
        "verl_root": verl_root.is_dir(),
        "pyproject": (verl_root / "pyproject.toml").is_file(),
        "uv_lock": (verl_root / "uv.lock").is_file(),
        "uv_environment": external_python.is_file(),
    }
    if require_data_files:
        required.update({
            "train_file": train_file is not None and train_file.is_file(),
            "val_file": val_file is not None and val_file.is_file(),
        })
    reasons.extend(name + "_missing" for name, ok in required.items() if not ok)
    commit = None
    if (verl_root / ".git").exists():
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=verl_root, capture_output=True, text=True
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip()
    if commit != VERL_COMMIT:
        reasons.append("verl_commit_mismatch")
    packages: dict[str, Any] | None = None
    if external_python.is_file():
        probe = subprocess.run(
            [
                str(external_python),
                "-c",
                (
                    "import importlib.metadata as m, json, ray, torch, verl, vllm; "
                    "print(json.dumps({'torch': torch.__version__, "
                    "'torch_cuda': torch.version.cuda, "
                    "'cuda_available': torch.cuda.is_available(), "
                    "'vllm': m.version('vllm'), 'ray': m.version('ray')}))"
                ),
            ],
            cwd=verl_root,
            env=isolated_subprocess_env(),
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            packages = json.loads(probe.stdout.strip().splitlines()[-1])
            if packages.get("vllm") != "0.24.0":
                reasons.append("vllm_version_mismatch")
            if packages.get("torch_cuda") != "13.0":
                reasons.append("torch_cuda_version_mismatch")
            if not packages.get("cuda_available"):
                reasons.append("torch_cuda_unavailable")
        else:
            reasons.append("uv_environment_import_failed")
        proxy_probe = subprocess.run(
            [
                str(external_python),
                "-c",
                "import httpx; client = httpx.Client(); client.close()",
            ],
            cwd=verl_root,
            env=isolated_subprocess_env(),
            capture_output=True,
            text=True,
        )
        if proxy_probe.returncode != 0:
            reasons.append("httpx_proxy_initialization_failed")
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    if gpu is None or gpu.returncode != 0:
        reasons.append("nvidia_smi_unavailable")
    return {
        "status": "ok" if not reasons else "blocked",
        "ready": not reasons,
        "verl_tag": VERL_TAG,
        "expected_commit": VERL_COMMIT,
        "resolved_commit": commit,
        "uv": uv,
        "packages": packages,
        "host_memory": host_memory,
        "gpu_inventory": gpu.stdout.strip() if gpu and gpu.returncode == 0 else None,
        "required_paths": required,
        "reasons": reasons,
    }


def _sample_gpu_memory(
    stop: threading.Event, samples: list[float], host_samples: list[dict[str, float]]
) -> None:
    """Poll GPU allocations and host headroom across initialization and training."""

    while not stop.is_set():
        if Path("/proc/meminfo").is_file():
            fields = dict(
                line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
            host_samples.append({
                "total_mb": int(fields["MemTotal"].split()[0]) / 1024,
                "available_mb": int(fields["MemAvailable"].split()[0]) / 1024,
            })
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            for raw in completed.stdout.splitlines():
                try:
                    samples.append(float(raw.strip()))
                except ValueError:
                    pass
        stop.wait(0.5)


def run_candidate(command: list[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    """Stream one candidate and retain complete output for audit."""

    started = time.monotonic()
    lines: list[str] = []
    gpu_memory_samples: list[float] = []
    host_memory_samples: list[dict[str, float]] = []
    sampler_stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_gpu_memory,
        args=(sampler_stop, gpu_memory_samples, host_memory_samples),
        daemon=True,
    )
    sampler.start()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=isolated_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            lines.append(line)
        returncode = process.wait()
    sampler_stop.set()
    sampler.join(timeout=2)
    rows = parse_console_metrics(lines)
    return {
        "status": "ok" if returncode == 0 and rows else "failed",
        "returncode": returncode,
        "duration_s": round(time.monotonic() - started, 3),
        "command": command,
        "log_path": str(log_path),
        "host_memory": {
            "minimum_available_mb": min(
                (sample["available_mb"] for sample in host_memory_samples), default=None
            ),
            "peak_used_mb": max(
                (sample["total_mb"] - sample["available_mb"] for sample in host_memory_samples),
                default=None,
            ),
            "samples": len(host_memory_samples),
            "scope": "whole_host_memtotal_minus_memavailable",
        },
        "gpu_memory": {
            "peak_used_mb": max(gpu_memory_samples) if gpu_memory_samples else None,
            "mean_used_mb": mean(gpu_memory_samples) if gpu_memory_samples else None,
            "samples": len(gpu_memory_samples),
            "scope": "whole_device",
        },
        "metrics": summarize_rows(rows),
    }


def build_decisions(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Make evidence-completeness decisions without claiming production scale."""

    baseline = candidates.get("async_release_metrics", {})
    rejection = candidates.get("async_release_reject", {})
    retained = candidates.get("async_retain_metrics", {})
    baseline_mean = baseline.get("metrics", {}).get("mean", {})
    rejection_mean = rejection.get("metrics", {}).get("mean", {})
    retained_mean = retained.get("metrics", {}).get("mean", {})

    lifecycle_fields = (
        "rollout_corr/kl",
        "rollout_corr/log_ppl_abs_diff",
        "rollout_corr/chi2_token",
    )
    lifecycle_ok = (
        baseline.get("status") == "ok"
        and rejection.get("status") == "ok"
        and all(key in baseline_mean for key in lifecycle_fields)
        and "rollout_corr/rollout_rs_masked_fraction" in rejection_mean
    )
    system_fields = ("timing_s/gen", "timing_s/update_actor", "timing_s/step", "perf/throughput")
    system_ok = (
        baseline.get("status") == "ok"
        and retained.get("status") == "ok"
        and all(key in baseline_mean and key in retained_mean for key in system_fields)
        and baseline.get("gpu_memory", {}).get("peak_used_mb") is not None
        and retained.get("gpu_memory", {}).get("peak_used_mb") is not None
    )
    return {
        "rollout_lifecycle": {
            "decision": "accept_mechanism" if lifecycle_ok else "tune",
            "reason": "policy_gap_and_rejection_metrics_complete" if lifecycle_ok else "lifecycle_evidence_incomplete",
            "scope": "single_gpu_asynchronous_rollout_admission",
        },
        "rl_system": {
            "decision": "accept_mechanism" if system_ok else "tune",
            "reason": "phase_timing_and_cache_residency_pair_complete" if system_ok else "system_evidence_incomplete",
            "scope": "single_gpu_colocated_generate_update_loop",
        },
    }


def build_comparisons(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Expose compact paired evidence used by the two notebooks."""

    def metric(candidate: str, key: str) -> float | None:
        value = candidates.get(candidate, {}).get("metrics", {}).get("mean", {}).get(key)
        return float(value) if value is not None else None

    def memory(candidate: str) -> float | None:
        value = candidates.get(candidate, {}).get("gpu_memory", {}).get("peak_used_mb")
        return float(value) if value is not None else None

    def pair(key: str, baseline: str, candidate: str) -> dict[str, float | None]:
        left, right = metric(baseline, key), metric(candidate, key)
        return {
            "baseline": left,
            "candidate": right,
            "delta": right - left if left is not None and right is not None else None,
        }

    release = "async_release_metrics"
    reject = "async_release_reject"
    retain = "async_retain_metrics"
    release_memory, retain_memory = memory(release), memory(retain)
    return {
        "rollout_lifecycle": {
            key: pair(key, release, reject)
            for key in (
                "rollout_corr/kl",
                "rollout_corr/log_ppl_abs_diff",
                "rollout_corr/chi2_token",
                "rollout_corr/rollout_rs_masked_fraction",
            )
        },
        "rl_system": {
            **{
                key: pair(key, release, retain)
                for key in ("timing_s/gen", "timing_s/update_actor", "timing_s/update_weights", "timing_s/step", "perf/throughput")
            },
            "peak_used_mb": {
                "baseline": release_memory,
                "candidate": retain_memory,
                "delta": (
                    retain_memory - release_memory
                    if release_memory is not None and retain_memory is not None
                    else None
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 verl 单卡轨迹生命周期与系统机制矩阵")
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=None)
    parser.add_argument("--total-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verl_root = args.verl_root.expanduser().resolve()
    train_file = (args.train_file or verl_root / "data/gsm8k/train.parquet").expanduser().resolve()
    val_file = (args.val_file or verl_root / "data/gsm8k/test.parquet").expanduser().resolve()
    preflight = inspect_runtime(verl_root, train_file, val_file)
    candidates = ("async_release_metrics", "async_release_reject", "async_retain_metrics")
    commands = {
        name: build_verl_command(
            verl_root=verl_root,
            model_path=args.model_path,
            train_file=train_file,
            val_file=val_file,
            run_label=args.run_label,
            candidate=name,
            total_steps=args.total_steps,
        )
        for name in candidates
    }
    plan = {
        "schema_version": "verl-system-mechanism/v1",
        "run_label": args.run_label,
        "preflight": preflight,
        "config": {
            "verl_tag": VERL_TAG,
            "verl_commit": VERL_COMMIT,
            "model_path": args.model_path,
            "train_file": str(train_file),
            "val_file": str(val_file),
            "total_training_steps": args.total_steps,
            "candidates": list(candidates),
        },
        "commands": commands,
    }
    if args.mode == "preflight":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0 if preflight["ready"] else 2
    if not preflight["ready"]:
        raise RuntimeError("preflight blocked: " + ", ".join(preflight["reasons"]))
    if args.output.exists():
        raise FileExistsError(f"result already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(candidates):
        log_path = args.output.with_name(f"{args.output.stem}_{name}.log")
        results[name] = run_candidate(commands[name], cwd=verl_root, log_path=log_path)
        if index == 0 and results[name]["status"] != "ok":
            for skipped in candidates[index + 1 :]:
                results[skipped] = {
                    "status": "skipped",
                    "reason": "baseline_candidate_failed",
                    "command": commands[skipped],
                }
            break
    record = {
        **plan,
        "evidence_level": "real_backend_single_gpu_mechanism",
        "candidates": results,
        "comparisons": build_comparisons(results),
        "decisions": build_decisions(results),
        "limitations": [
            "不覆盖多工作进程队列、网络传输、跨节点容错或真实异步策略滞后",
            "拒绝采样候选只核对行为偏离指标与准入路径，不用于选择生产阈值",
        ],
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "ok" for item in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
