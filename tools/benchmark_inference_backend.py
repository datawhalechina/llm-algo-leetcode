#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible inference backend with a fixed workload.

Start a backend separately (for example, ``vllm serve ...``), then run this
client against its ``/v1/chat/completions`` endpoint. The implementation uses
only the Python standard library so it remains an optional GPU layer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from inference_result_schema import from_backend_report
except ImportError:  # pragma: no cover - supports ``python tools/...``
    from tools.inference_result_schema import from_backend_report


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _read_workload(path: str | None, prompt: str, num_prompts: int) -> list[str]:
    if not path:
        return [prompt] * num_prompts
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, str):
            prompts.append(item)
        elif "prompt" in item:
            prompts.append(str(item["prompt"]))
        elif "messages" in item:
            prompts.append(json.dumps(item["messages"], ensure_ascii=False))
        else:
            raise ValueError("Each workload JSONL row needs prompt or messages")
    if not prompts:
        raise ValueError(f"Workload is empty: {path}")
    if num_prompts <= 0:
        return prompts
    # 固定顺序循环保留请求内容与到达顺序，允许小型冻结 workload 形成指定规模；
    # 结果中的 requests 是实际请求数，不表示存在同样数量的不同样本。
    return [prompts[index % len(prompts)] for index in range(num_prompts)]


def _read_request_specs(path: str | None, prompt: str, num_prompts: int) -> list[dict[str, Any]]:
    """Read prompts plus optional request ids and priorities from JSONL."""
    if not path:
        rows = [{"prompt": prompt, "request_id": "request-0", "priority": 0}]
    else:
        rows = []
        for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str):
                item = {"prompt": item}
            if "prompt" in item:
                prompt_text = str(item["prompt"])
            elif "messages" in item:
                prompt_text = json.dumps(item["messages"], ensure_ascii=False)
            else:
                raise ValueError("Each workload JSONL row needs prompt or messages")
            rows.append({
                "prompt": prompt_text,
                "request_id": str(item.get("request_id", f"request-{index}")),
                "priority": int(item.get("priority", 0)),
                **({"expected_answer": str(item["expected_answer"])} if "expected_answer" in item else {}),
                **({"chat_template_kwargs": item["chat_template_kwargs"]} if "chat_template_kwargs" in item else {}),
            })
    if not rows:
        raise ValueError(f"Workload is empty: {path}")
    if num_prompts <= 0:
        return rows
    return [dict(rows[index % len(rows)], cycle=index // len(rows)) for index in range(num_prompts)]


def _parse_sse_line(line: bytes) -> dict[str, Any] | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:") or text[5:].strip() == "[DONE]":
        return None
    try:
        return json.loads(text[5:].strip())
    except json.JSONDecodeError:
        return None


def _run_request(base_url: str, model: str, request_spec: dict[str, Any], max_tokens: int,
                 temperature: float, timeout_s: float, send_priority: bool = False,
                 start_barrier: threading.Barrier | None = None,
                 clock_origin_s: float | None = None) -> dict[str, Any]:
    prompt = request_spec["prompt"]
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request_spec.get("chat_template_kwargs") is not None:
        body["chat_template_kwargs"] = request_spec["chat_template_kwargs"]
    if send_priority:
        body["priority"] = int(request_spec.get("priority", 0))
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if start_barrier is not None:
        start_barrier.wait(timeout=timeout_s)
    started = time.perf_counter()
    clock_origin_s = started if clock_origin_s is None else clock_origin_s
    first_token_s: float | None = None
    last_content_s: float | None = None
    stream_done_received = False
    chunks = 0
    output_text: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                if raw_line.strip() == b"data: [DONE]":
                    stream_done_received = True
                payload = _parse_sse_line(raw_line)
                if payload is None:
                    continue
                if payload.get("usage"):
                    usage = payload["usage"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                finish_reason = choices[0].get("finish_reason") or finish_reason
                content = (choices[0].get("delta") or {}).get("content") or ""
                if content:
                    content_received_s = time.perf_counter()
                    if first_token_s is None:
                        first_token_s = content_received_s
                    last_content_s = content_received_s
                    chunks += 1
                    output_text.append(content)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False, "error": str(exc), "prompt": prompt,
            "request_id": request_spec.get("request_id"),
            "priority": request_spec.get("priority", 0),
            **({"expected_answer": request_spec["expected_answer"]} if "expected_answer" in request_spec else {}),
            "submit_offset_ms": round((started - clock_origin_s) * 1000, 3),
        }

    finished = time.perf_counter()
    ttft_ms = ((first_token_s or finished) - started) * 1000
    e2e_ms = (finished - started) * 1000
    output_tokens = (usage or {}).get("completion_tokens")
    estimated_tokens = output_tokens is None
    if output_tokens is None:
        output_tokens = max(1, round(len("".join(output_text)) / 4))
    tpot_ms = (e2e_ms - ttft_ms) / max(1, output_tokens - 1)
    return {
        "ok": True,
        "prompt": prompt,
        "request_id": request_spec.get("request_id"),
        "priority": request_spec.get("priority", 0),
        **({"expected_answer": request_spec["expected_answer"]} if "expected_answer" in request_spec else {}),
        "output_text": "".join(output_text),
        "finish_reason": finish_reason,
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "submit_offset_ms": round((started - clock_origin_s) * 1000, 3),
        "first_token_offset_ms": round(((first_token_s or finished) - clock_origin_s) * 1000, 3),
        "completion_offset_ms": round((finished - clock_origin_s) * 1000, 3),
        "first_content_offset_ms": (
            round((first_token_s - clock_origin_s) * 1000, 3)
            if first_token_s is not None else None
        ),
        "last_content_offset_ms": (
            round((last_content_s - clock_origin_s) * 1000, 3)
            if last_content_s is not None else None
        ),
        "stream_done_received": stream_done_received,
        "content_tpot_proxy_ms": (
            round((last_content_s - first_token_s) * 1000 / (output_tokens - 1), 3)
            if output_tokens > 1 and first_token_s is not None and last_content_s is not None
            else None
        ),
        "output_tokens": output_tokens,
        "estimated_output_tokens": estimated_tokens,
        "stream_chunks": chunks,
        "usage": usage,
    }


def _parse_answer_marker(output_text: str) -> str | None:
    """Parse the frozen workload's explicit ``Answer: <integer>`` format."""
    matches = re.findall(r"(?im)^\s*Answer\s*:\s*(-?\d+)\s*$", output_text)
    return matches[-1] if matches else None


def _attach_task_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score objective answer, format, and stop-vs-truncation evidence when available."""
    scored = [item for item in results if item.get("ok") and "expected_answer" in item]
    if not scored:
        return {
            "answer_scored_requests": 0,
            "unique_answer_tasks": 0,
            "expected_answer_accuracy": None,
            "valid_answer_format_rate": None,
            "response_completion_rate": None,
            "complete_correct_rate": None,
            "truncated_response_rate": None,
            "answer_repeat_consistency": None,
        }
    for item in scored:
        parsed = _parse_answer_marker(str(item.get("output_text", "")))
        item["parsed_answer"] = parsed
        item["answer_format_valid"] = parsed is not None
        item["answer_correct"] = parsed == str(item["expected_answer"])
        item["response_complete"] = item.get("finish_reason") == "stop"
        item["complete_and_correct"] = item["answer_correct"] and item["response_complete"]
    total = len(scored)
    by_request: dict[str, list[str | None]] = {}
    for item in scored:
        by_request.setdefault(str(item.get("request_id", "")), []).append(item["parsed_answer"])
    stable_tasks = sum(
        bool(values) and all(value is not None for value in values) and len(set(values)) == 1
        for values in by_request.values()
    )
    return {
        "answer_scored_requests": total,
        "unique_answer_tasks": len(by_request),
        "expected_answer_accuracy": round(sum(item["answer_correct"] for item in scored) / total, 4),
        "valid_answer_format_rate": round(sum(item["answer_format_valid"] for item in scored) / total, 4),
        "response_completion_rate": round(sum(item["response_complete"] for item in scored) / total, 4),
        "complete_correct_rate": round(sum(item["complete_and_correct"] for item in scored) / total, 4),
        "truncated_response_rate": round(sum(item.get("finish_reason") == "length" for item in scored) / total, 4),
        "answer_repeat_consistency": round(stable_tasks / len(by_request), 4) if by_request else None,
    }


def _priority_pair_score(results: list[dict[str, Any]], rank_key: str) -> float | None:
    """Return the share of cross-priority pairs ordered high priority first."""

    ranked = [
        item for item in results
        if isinstance(item.get("priority"), int)
        and isinstance(item.get(rank_key), (int, float))
    ]
    ranked.sort(key=lambda item: (item[rank_key], str(item.get("request_id", ""))))
    comparable_pairs = preferred_pairs = 0
    for left_index, left in enumerate(ranked):
        for right in ranked[left_index + 1:]:
            if left["priority"] == right["priority"]:
                continue
            comparable_pairs += 1
            preferred_pairs += int(left["priority"] < right["priority"])
    return preferred_pairs / comparable_pairs if comparable_pairs else None


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    request_specs = _read_request_specs(args.workload, args.prompt, args.num_prompts)
    if args.synchronize_start and args.concurrency < len(request_specs):
        raise ValueError("--synchronize-start 要求 concurrency 不小于实际请求数")
    for _ in range(args.warmup):
        warmup_spec = dict(request_specs[0], priority=0)
        result = _run_request(args.base_url, args.model, warmup_spec, args.max_tokens,
                              args.temperature, args.timeout, False)
        if not result["ok"]:
            raise RuntimeError(f"Warm-up request failed: {result['error']}")

    started = time.perf_counter()
    start_barrier = threading.Barrier(len(request_specs)) if args.synchronize_start else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(
            _run_request, args.base_url, args.model, request_spec,
            args.max_tokens, args.temperature, args.timeout,
            args.send_workload_priority,
            start_barrier, started,
        ) for request_spec in request_specs]
        results = []
        for completion_rank, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            result["completion_rank"] = completion_rank
            results.append(result)
    duration_s = time.perf_counter() - started

    successful = [item for item in results if item["ok"]]
    task_quality = _attach_task_quality(results)
    for first_token_rank, item in enumerate(
        sorted(successful, key=lambda row: (
            row["first_token_offset_ms"], str(row.get("request_id", ""))
        )),
        start=1,
    ):
        item["first_token_rank"] = first_token_rank
    ttft = [item["ttft_ms"] for item in successful]
    tpot = [item["tpot_ms"] for item in successful]
    e2e = [item["e2e_ms"] for item in successful]
    output_tokens = sum(item["output_tokens"] for item in successful)
    prompt_token_values = [
        item["usage"].get("prompt_tokens")
        for item in successful
        if isinstance(item.get("usage"), dict)
        and isinstance(item["usage"].get("prompt_tokens"), (int, float))
    ]
    priority_order_score = _priority_pair_score(successful, "completion_rank")
    priority_first_token_order_score = _priority_pair_score(successful, "first_token_rank")
    return {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "workload": {
            "source": args.workload or "inline_prompt",
            "requests": len(request_specs),
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "repeats": 1,
            "workload_priority_sent": args.send_workload_priority,
            "synchronized_start": args.synchronize_start,
        },
        "metrics": {
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
            "request_throughput_per_s": round(len(successful) / duration_s, 4) if duration_s else 0.0,
            "output_token_throughput_per_s": round(output_tokens / duration_s, 4) if duration_s else 0.0,
            "ttft_ms": {"mean": round(statistics.mean(ttft), 3) if ttft else None,
                        "p50": _percentile(ttft, 50), "p95": _percentile(ttft, 95),
                        "p99": _percentile(ttft, 99)},
            "tpot_ms": {"mean": round(statistics.mean(tpot), 3) if tpot else None,
                        "p50": _percentile(tpot, 50), "p95": _percentile(tpot, 95),
                        "p99": _percentile(tpot, 99)},
            "e2e_ms": {"mean": round(statistics.mean(e2e), 3) if e2e else None,
                       "p50": _percentile(e2e, 50), "p95": _percentile(e2e, 95),
                       "p99": _percentile(e2e, 99)},
            "output_tokens": output_tokens,
            "prompt_tokens_mean": round(statistics.mean(prompt_token_values), 3)
            if prompt_token_values else None,
            "duration_s": round(duration_s, 4),
            "priority_order_score": round(priority_order_score, 4)
            if priority_order_score is not None else None,
            "priority_first_token_order_score": round(priority_first_token_order_score, 4)
            if priority_first_token_order_score is not None else None,
            **task_quality,
        },
        "results": results if args.include_requests else None,
    }


def aggregate_repeated_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated fixed-workload runs without hiding per-run evidence."""
    if not reports:
        raise ValueError("至少需要一轮 benchmark 结果")
    if len(reports) == 1:
        return reports[0]
    aggregate = json.loads(json.dumps(reports[0]))
    aggregate["runs"] = reports
    aggregate["workload"] = dict(aggregate.get("workload", {}), repeats=len(reports))
    aggregate["metrics_aggregation"] = "mean_of_run_metrics"
    aggregate["results_scope"] = "first_run; use runs[*].results for repeated request-level metrics"
    metrics = dict(aggregate.get("metrics", {}))
    for key, value in list(metrics.items()):
        if isinstance(value, (int, float)):
            values = [run.get("metrics", {}).get(key) for run in reports]
            values = [item for item in values if isinstance(item, (int, float))]
            if values:
                metrics[key] = round(statistics.mean(values), 4)
        elif isinstance(value, dict):
            nested = dict(value)
            for nested_key, nested_value in value.items():
                values = [run.get("metrics", {}).get(key, {}).get(nested_key) for run in reports]
                values = [item for item in values if isinstance(item, (int, float))]
                if values:
                    nested[nested_key] = round(statistics.mean(values), 4)
            metrics[key] = nested
    aggregate["metrics"] = metrics
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="backend")
    parser.add_argument("--project", default="66", help="Project number using the shared result schema")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--dtype", default="unknown")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cache-policy", default="default")
    parser.add_argument("--role", choices=("baseline", "candidate"))
    parser.add_argument("--peak-memory-mb", type=float)
    parser.add_argument("--workload", help="JSONL file with prompt or messages fields")
    parser.add_argument("--prompt", default="Explain KV cache in one sentence.")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--include-requests", action="store_true")
    parser.add_argument(
        "--send-workload-priority", action="store_true",
        help="Send each JSONL row's priority field to a priority-enabled backend",
    )
    parser.add_argument(
        "--synchronize-start", action="store_true",
        help="Release one full-concurrency request burst together; requires concurrency >= requests",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats 必须为正数")
    report = aggregate_repeated_reports([benchmark(args) for _ in range(args.repeats)])
    normalized = from_backend_report(
        report,
        project=args.project,
        strategy=args.label,
        backend=args.backend,
        dtype=args.dtype,
        batch=args.batch,
        cache_policy=args.cache_policy,
        role=args.role,
        peak_memory_mb=args.peak_memory_mb,
    )
    # Keep the raw report for compatibility with earlier result files while
    # adding a stable, cross-project contract under ``normalized_result``.
    report["schema_version"] = normalized["schema_version"]
    report["normalized_result"] = normalized
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
