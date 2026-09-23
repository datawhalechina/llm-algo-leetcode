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
import statistics
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
    return prompts[:num_prompts] if num_prompts > 0 else prompts


def _parse_sse_line(line: bytes) -> dict[str, Any] | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:") or text[5:].strip() == "[DONE]":
        return None
    try:
        return json.loads(text[5:].strip())
    except json.JSONDecodeError:
        return None


def _run_request(base_url: str, model: str, prompt: str, max_tokens: int,
                 temperature: float, timeout_s: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_s: float | None = None
    chunks = 0
    output_text: list[str] = []
    usage: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                payload = _parse_sse_line(raw_line)
                if payload is None:
                    continue
                if payload.get("usage"):
                    usage = payload["usage"]
                choices = payload.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content") or ""
                if content:
                    first_token_s = first_token_s or time.perf_counter()
                    chunks += 1
                    output_text.append(content)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc), "prompt": prompt}

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
        "ttft_ms": round(ttft_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "output_tokens": output_tokens,
        "estimated_output_tokens": estimated_tokens,
        "stream_chunks": chunks,
        "usage": usage,
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _read_workload(args.workload, args.prompt, args.num_prompts)
    for _ in range(args.warmup):
        result = _run_request(args.base_url, args.model, prompts[0], args.max_tokens,
                              args.temperature, args.timeout)
        if not result["ok"]:
            raise RuntimeError(f"Warm-up request failed: {result['error']}")

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_run_request, args.base_url, args.model, prompt,
                               args.max_tokens, args.temperature, args.timeout)
                   for prompt in prompts]
        results = [future.result() for future in futures]
    duration_s = time.perf_counter() - started

    successful = [item for item in results if item["ok"]]
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
    return {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "workload": {
            "source": args.workload or "inline_prompt",
            "requests": len(prompts),
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "repeats": 1,
        },
        "metrics": {
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
            "request_throughput_per_s": round(len(successful) / duration_s, 4) if duration_s else 0.0,
            "output_token_throughput_per_s": round(output_tokens / duration_s, 4) if duration_s else 0.0,
            "ttft_ms": {"mean": round(statistics.mean(ttft), 3) if ttft else None,
                        "p50": _percentile(ttft, 50), "p99": _percentile(ttft, 99)},
            "tpot_ms": {"mean": round(statistics.mean(tpot), 3) if tpot else None,
                        "p50": _percentile(tpot, 50), "p99": _percentile(tpot, 99)},
            "e2e_ms": {"mean": round(statistics.mean(e2e), 3) if e2e else None,
                       "p50": _percentile(e2e, 50), "p99": _percentile(e2e, 99)},
            "output_tokens": output_tokens,
            "prompt_tokens_mean": round(statistics.mean(prompt_token_values), 3)
            if prompt_token_values else None,
            "duration_s": round(duration_s, 4),
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
