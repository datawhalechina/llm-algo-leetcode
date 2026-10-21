"""Screen bounded rollout groups before testing a real lifecycle update.

The driver uses the existing verl environment and vLLM. Screening uses a
fixed arithmetic fixture; it does not train or claim policy activation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, median, pstdev
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.verl_system_mechanism_runtime import (
    VERL_COMMIT, VERL_TAG, inspect_runtime, isolated_subprocess_env,
)

SCHEMA = "verl-lifecycle-probe/v1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
POOL_SIZE = 8
GROUP_SIZE = 4
MAX_TOKENS = 128
SEED = 42
PROMPT_REVISION = "arithmetic_integer_only_v1"
SYSTEM_PROMPT = (
    "Calculate the expression. Reply with only the integer result. "
    "Do not include an explanation, units, or an answer marker."
)
# Historical grammar constants are retained for interpreting v9_4/v9_5 JSON.
ANSWER_REGEX = r"#### -?(0|[1-9][0-9]*)"
# Allow the short calculation requested by the unchanged system prompt.
# Excluding '#' from calculation lines guarantees a single answer marker.
RESPONSE_REGEX = r"(?:[^#\r\n]{1,96}\n){0,2}" + ANSWER_REGEX
DECODE_REVISION = "unconstrained_arithmetic_v1"
TASK_REVISION = "single_operation_arithmetic_v1"
INTEGER_REPLY_REGEX = r"\s*-?(0|[1-9][0-9]*)\s*"
ARITHMETIC_TASKS = (
    ("add", 7, 5), ("subtract", 23, 8),
    ("multiply", 6, 7), ("divide", 48, 6),
    ("add", 37, 58), ("subtract", 81, 47),
    ("multiply", 17, 6), ("divide", 144, 12),
)


def build_arithmetic_tasks() -> list[dict]:
    """Freeze operands before sampling; compute labels without model output."""
    tasks = []
    symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
    for index, (operation, left, right) in enumerate(ARITHMETIC_TASKS):
        if operation == "add":
            result = left + right
        elif operation == "subtract":
            result = left - right
        elif operation == "multiply":
            result = left * right
        else:
            if not right or left % right:
                raise ValueError("The fixed division task must have an integer result")
            result = left // right
        tasks.append({
            "sample_id": f"arithmetic/v1/{index}",
            "question": f"{left} {symbols[operation]} {right} = ?",
            "operation": operation, "operands": [left, right],
            "ground_truth": str(result), "data_source": TASK_REVISION,
            "split": "mechanism_probe",
        })
    return tasks


def score_arithmetic_reply(text: str, ground_truth: str) -> tuple[str | None, float]:
    """Score the whole integer reply; never recover an answer from reasoning."""
    if re.fullmatch(INTEGER_REPLY_REGEX, text) is None:
        return None, 0.0
    answer = text.strip()
    return answer, float(int(answer) == int(ground_truth))


def digest_json(value: object) -> str:
    """Hash canonical identities, not incidental dictionary formatting."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def assess_groups(groups: list[dict]) -> dict:
    """Require complete, parseable groups with real correctness differences."""
    eligible = []
    total = parsed = clipped = 0
    for group in groups:
        replies = group["responses"]
        total += len(replies)
        parsed += sum(reply["parsed_answer"] is not None for reply in replies)
        clipped += sum(reply["finish_reason"] == "length" for reply in replies)
        complete = len(replies) == GROUP_SIZE and all(
            reply["finish_reason"] == "stop"
            and reply["parsed_answer"] is not None
            and math.isfinite(reply["reward"])
            and reply["reward"] in (0.0, 1.0)
            for reply in replies
        )
        if complete and len({reply["reward"] for reply in replies}) > 1:
            eligible.append(group["sample_id"])
    return {
        "passed": bool(eligible),
        "eligible_group_ids": eligible,
        "groups_observed": len(groups),
        "responses_observed": total,
        "answer_parse_ratio": parsed / total if total else None,
        "length_limit_ratio": clipped / total if total else None,
        "next_stage_allowed": bool(eligible),
        "reason": "complete_mixed_reward_group_found" if eligible else "no_complete_mixed_reward_group",
    }


def diagnostic_number(text: str | None) -> Decimal | None:
    """Read a complete numeric string; never guess from a reasoning paragraph."""
    if text is None:
        return None
    value = text.strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)", value):
        return None
    try:
        parsed = Decimal(value.replace(",", ""))
        return parsed if parsed.is_finite() else None
    except InvalidOperation:
        return None


def summarize_diagnostics(groups: list[dict]) -> dict:
    """Separate format, arithmetic and group variation without changing rewards."""
    totals = Counter()
    errors = Counter()
    lengths = []
    details = []
    for group in groups:
        truth = diagnostic_number(group["ground_truth"])
        replies = group["responses"]
        rewards = [reply["reward"] for reply in replies]
        valid_rewards = bool(rewards) and all(math.isfinite(r) for r in rewards)
        response_details = []
        for index, reply in enumerate(replies):
            strict = reply["parsed_answer"] is not None
            numeric = diagnostic_number(reply["parsed_answer"] if strict else reply.get("text", ""))
            correct = numeric == truth if numeric is not None and truth is not None else None
            clipped = reply["finish_reason"] == "length"
            positive = math.isfinite(reply["reward"]) and reply["reward"] > 0
            if clipped:
                error = "length_limit"
            elif numeric is None or truth is None:
                error = "unparseable"
            elif not strict:
                error = "missing_format"
            elif correct:
                error = "correct" if positive else "numeric_representation_mismatch"
            else:
                error = "numeric_wrong"
            errors[error] += 1
            totals.update({
                "responses": 1, "strict_parsed": int(strict),
                "positive_reward": int(positive), "length_limited": int(clipped),
                "numeric_parsed": int(numeric is not None),
                "numeric_correct": int(correct is True),
                "numeric_wrong": int(correct is False),
                "numeric_unknown": int(correct is None),
                "missing_strict_format": int(not strict),
            })
            lengths.append(len(reply.get("token_ids", [])))
            response_details.append({
                "response_index": index, "strict_parsed": strict,
                "numeric_source": "strict_answer" if strict else "whole_reply",
                "numeric_answer": str(numeric) if numeric is not None else None,
                "numeric_correct": correct, "original_reward": reply["reward"],
                "primary_error": error,
            })
        details.append({
            "sample_id": group["sample_id"], "rewards": rewards,
            "reward_mean": mean(rewards) if valid_rewards else None,
            "reward_std": pstdev(rewards) if valid_rewards else None,
            "positive_reward_count": sum(r > 0 for r in rewards if math.isfinite(r)),
            "numeric_correct_count": sum(item["numeric_correct"] is True for item in response_details),
            "unique_reply_count": len({reply.get("text", "") for reply in replies}),
            "eligible": assess_groups([group])["passed"],
            "responses": response_details,
        })
    total = totals["responses"]
    return {
        "schema_version": "rollout-reward-diagnostics/v1",
        "summary": {
            "counts": dict(totals), "primary_error_counts": dict(errors),
            "positive_reward_ratio": totals["positive_reward"] / total if total else None,
            "numeric_correct_ratio": totals["numeric_correct"] / total if total else None,
            "mixed_reward_group_count": sum(
                item["reward_std"] is not None and item["reward_std"] > 0 for item in details
            ),
            "eligible_group_ratio": sum(item["eligible"] for item in details) / len(details) if details else None,
            "response_tokens": {"min": min(lengths) if lengths else None,
                                "median": median(lengths) if lengths else None,
                                "max": max(lengths) if lengths else None},
        },
        "groups": details,
        "scope": "只诊断格式和数值等价，不修改严格奖励或下一阶段准入条件",
    }


def summarize_screening(
    groups: list[dict], *, constrained: bool = False, response_regex: str | None = None,
) -> dict:
    """Explain screening blockers; never turn missing evidence into a pass."""
    states = Counter({name: 0 for name in (
        "incomplete_or_invalid", "all_wrong", "all_correct", "mixed_reward",
    )})
    blockers = Counter()
    trace_checks = Counter()
    valid_trace_tokens = total_tokens = 0
    complete_positive = complete_parsed = 0
    canonical_answers = response_count = 0
    contract_matches = calculation_responses = 0
    # Historical callers used answer-only decoding. Do not reinterpret old JSON
    # using the latest grammar: callers should pass its recorded constraint.
    contract_regex = response_regex if response_regex is not None else ANSWER_REGEX
    group_details = []
    for group in groups:
        replies = group["responses"]
        response_count += len(replies)
        canonical_answers += sum(bool(re.fullmatch(ANSWER_REGEX, r.get("text", ""))) for r in replies)
        contract_matches += sum(bool(re.fullmatch(contract_regex, r.get("text", ""))) for r in replies)
        calculation_responses += sum(
            "\n#### " in r.get("text", "") for r in replies
        )
        reasons = []
        if len(replies) != GROUP_SIZE:
            reasons.append("wrong_group_size")
        if any(reply["finish_reason"] != "stop" for reply in replies):
            reasons.append("not_normally_terminated")
        if any(reply["parsed_answer"] is None for reply in replies):
            reasons.append("answer_parse_failed")
        if constrained and any(not re.fullmatch(contract_regex, r.get("text", "")) for r in replies):
            reasons.append("constraint_not_satisfied")
        rewards = [reply["reward"] for reply in replies]
        if any(not math.isfinite(r) or r not in (0, 1) for r in rewards):
            reasons.append("invalid_reward")
        state = "incomplete_or_invalid"
        if not reasons:
            state = "mixed_reward" if len(set(rewards)) > 1 else (
                "all_correct" if rewards[0] == 1 else "all_wrong"
            )
            complete_parsed += len(replies)
            complete_positive += sum(r == 1 for r in rewards)
        states[state] += 1
        blockers.update(reasons)
        trace_valid = True
        for reply in replies:
            tokens = reply.get("token_ids", [])
            probabilities = reply.get("behavior_log_probs")
            mask = reply.get("action_mask")
            total_tokens += len(tokens)
            if probabilities is None or mask is None or not tokens:
                trace_checks["missing_or_empty"] += 1
                trace_valid = False
            elif not (len(tokens) == len(probabilities) == len(mask)):
                trace_checks["length_mismatch"] += 1
                trace_valid = False
            elif any(not math.isfinite(p) or p > 1e-5 for p in probabilities):
                trace_checks["invalid_log_probability"] += 1
                trace_valid = False
            elif any(m != 1 for m in mask):
                trace_checks["unexpected_action_mask"] += 1
                trace_valid = False
            else:
                trace_checks["valid_responses"] += 1
                valid_trace_tokens += len(tokens)
        group_details.append({
            "sample_id": group["sample_id"], "state": state,
            "blockers": reasons, "trace_valid": trace_valid,
            "centered_reward_nonzero_count": sum(r != mean(rewards) for r in rewards)
            if not reasons else None,
        })
    mixed = states["mixed_reward"]
    usable = sum(g["state"] == "mixed_reward" and g["trace_valid"] for g in group_details)
    if not groups:
        action = "collect_screening_data"
    elif any(trace_checks[k] for k in ("missing_or_empty", "length_mismatch",
                                       "invalid_log_probability", "unexpected_action_mask")):
        action = "repair_trajectory_evidence"
    elif mixed:
        action = "design_matching_policy_update" if constrained else "prepare_bounded_update"
    elif blockers:
        action = "inspect_format_and_termination"
    else:
        action = "review_question_difficulty_before_more_training"
    return {
        "schema_version": "rollout-screening-decision/v1",
        "group_count": len(groups), "group_states": dict(states),
        "response_count": response_count, "canonical_integer_answer_count": canonical_answers,
        "canonical_integer_answer_ratio": canonical_answers / response_count if response_count else None,
        "output_contract_regex": contract_regex,
        "output_contract_match_count": contract_matches,
        "output_contract_match_ratio": contract_matches / response_count if response_count else None,
        "calculation_response_count": calculation_responses,
        "calculation_response_ratio": calculation_responses / response_count if response_count else None,
        "group_blocker_counts": dict(blockers),  # Reasons can overlap.
        "complete_group_response_count": complete_parsed,
        "complete_group_positive_reward_ratio": complete_positive / complete_parsed
        if complete_parsed else None,
        "trace_checks": dict(trace_checks), "observed_token_count": total_tokens,
        "valid_trace_token_count": valid_trace_tokens,
        "mixed_group_count": mixed, "usable_mixed_group_count": usable,
        "usable_response_count": usable * GROUP_SIZE,
        "sampling_distribution": "grammar_constrained" if constrained else "unconstrained",
        "direct_training_reuse_allowed": False,
        "recommended_action": action, "groups": group_details,
        "scope": "筛查不是训练；受约束概率不能与未使用相同约束的学习器概率直接求 PPO 比率",
    }


def build_sampling_params():
    """Use ordinary sampling: no grammar, forced prefix or injected answers."""
    from vllm import SamplingParams

    return SamplingParams(
        n=GROUP_SIZE, temperature=1.0, top_p=1.0, top_k=-1,
        max_tokens=MAX_TOKENS, logprobs=0, seed=SEED,
    )


def run_probe(args: argparse.Namespace) -> dict:
    """Generate naturally from the frozen base and score fixed arithmetic."""
    # The standalone path lacks Ray's automatic spawn override. Set this
    # before importing vLLM: platform imports may already initialize CUDA.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import importlib.metadata as metadata
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM

    # Local-only resolution preserves the workstation's existing model identity.
    snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True)).resolve()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rows = build_arithmetic_tasks()
    if len(rows) != POOL_SIZE:
        raise ValueError("The frozen arithmetic pool has an unexpected size")
    groups, prompts = [], []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > 256:
            raise ValueError("Selected question exceeds the frozen prompt budget")
        groups.append({
            **row,
            "messages": messages,
            "prompt_sha256": digest_json(messages),
        })
        prompts.append(prompt)

    llm = LLM(
        model=str(snapshot), dtype="bfloat16", max_model_len=512,
        max_num_seqs=16, max_num_batched_tokens=512,
        gpu_memory_utilization=0.35, enforce_eager=True, seed=SEED,
        # Match verl's rollout mode. vLLM uses its native sampler to return
        # processed log probabilities, avoiding the optional FlashInfer JIT sampler.
        logprobs_mode="processed_logprobs",
    )
    outputs = llm.generate(prompts, build_sampling_params())
    if len(outputs) != len(groups):
        raise RuntimeError("vLLM returned an incomplete question pool")
    for group, output in zip(groups, outputs):
        group["responses"] = []
        for reply in output.outputs:
            token_ids = list(reply.token_ids)
            if reply.logprobs is None or len(reply.logprobs) != len(token_ids):
                raise RuntimeError("Missing sampled-token behavior log probabilities")
            log_probs = [float(probs[token].logprob) for token, probs in zip(token_ids, reply.logprobs)]
            if not all(math.isfinite(value) for value in log_probs):
                raise RuntimeError("Non-finite behavior log probabilities")
            parsed_answer, reward = score_arithmetic_reply(reply.text, group["ground_truth"])
            group["responses"].append({
                "text": reply.text, "token_ids": token_ids,
                "behavior_log_probs": log_probs,
                "action_mask": [1] * len(token_ids),
                "finish_reason": reply.finish_reason,
                "stop_reason": reply.stop_reason,
                "parsed_answer": parsed_answer, "reward": reward,
            })
    config = {
        "verl_tag": VERL_TAG, "verl_commit": VERL_COMMIT,
        "torch": metadata.version("torch"),
        "model_id": MODEL_ID, "resolved_snapshot": str(snapshot),
        "model_revision": snapshot.name,
        "task_revision": TASK_REVISION, "task_pool_sha256": digest_json(rows),
        "data_origin": "repository_fixed_arithmetic_fixture",
        "split": "mechanism_probe",
        "pool_size": POOL_SIZE, "group_size": GROUP_SIZE,
        "max_response_tokens": MAX_TOKENS, "seed": SEED,
        "prompt_revision": PROMPT_REVISION,
        "decode_revision": DECODE_REVISION,
        "structured_outputs": None,
        "verifier_reply_regex": INTEGER_REPLY_REGEX,
        "sampling_distribution": "unconstrained",
        "behavior_log_probs_semantics": "unconstrained_processed_sampling_log_probs",
        "temperature": 1.0, "top_p": 1.0, "top_k": -1,
        "reward_function": "tools.verl_lifecycle_probe.score_arithmetic_reply",
        "reward_method": "whole_integer_exact_v1", "vllm": metadata.version("vllm"),
        "worker_start_method": os.environ["VLLM_WORKER_MULTIPROC_METHOD"],
        "logprobs_mode": "processed_logprobs",
    }
    return {
        "schema_version": SCHEMA, "run_label": args.run_label,
        "stage": "reward_probe", "status": "ok", "config": config,
        "behavior_policy_id": digest_json({
            "revision": snapshot.name, "adapters": [],
            "decode_revision": DECODE_REVISION, "structured_outputs": None,
            "temperature": 1.0, "top_p": 1.0, "top_k": -1,
        }),
        "groups": groups, "gate": assess_groups(groups),
        "diagnostics": summarize_diagnostics(groups),
        "screening": summarize_screening(groups, response_regex=INTEGER_REPLY_REGEX),
        "limitations": [
            "题目为预先固定的合成算术任务，只验证机制，不估计 GSM8K 准确率或泛化能力",
            "当前仅运行冻结基座采样，未执行 LoRA 更新、策略准入或 worker 激活验证",
            "后续训练必须重新检查实际生成组的奖励差异；筛查通过不保证下一批优势非零",
            "普通采样无语法约束；训练前仍需核对学习器与行为侧概率、Token 和实际动作掩码",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="轨迹生命周期的短答案奖励筛查")
    parser.add_argument("--mode", choices=("preflight", "run", "worker"), default="preflight")
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.verl_root = args.verl_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"保留已有结果，请读取它或更换标签：{args.output}")
    if args.mode == "worker":
        record = run_probe(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"output": str(args.output), "gate": record["gate"]}, ensure_ascii=False))
        return 0
    preflight = inspect_runtime(args.verl_root, None, None, require_data_files=False)
    plan = {
        "schema_version": SCHEMA, "run_label": args.run_label,
        "stage": "reward_probe", "preflight": preflight,
        "config": {"model": MODEL_ID, "question_pool": POOL_SIZE, "responses_per_question": GROUP_SIZE,
                   "max_response_tokens": MAX_TOKENS, "training_steps": 0,
                   "prompt_revision": PROMPT_REVISION, "system_prompt": SYSTEM_PROMPT,
                   "decode_revision": DECODE_REVISION,
                   "structured_outputs": None, "sampling_distribution": "unconstrained",
                   "task_revision": TASK_REVISION,
                   "task_pool_sha256": digest_json(build_arithmetic_tasks()),
                   "tasks": build_arithmetic_tasks(),
                   "reward_method": "whole_integer_exact_v1",
                   "direct_training_reuse_allowed": False},
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not preflight["ready"]:
        return 2
    if args.mode == "preflight":
        return 0
    if args.output.with_suffix(".log").exists():
        raise FileExistsError("已有同标签筛查日志，请检查失败原因或更换标签")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.verl_root / ".venv/bin/python"), str(Path(__file__).resolve()),
               "--mode", "worker", "--verl-root", str(args.verl_root),
               "--run-label", args.run_label,
               "--output", str(args.output)]
    log = args.output.with_suffix(".log")
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=args.verl_root, env=isolated_subprocess_env(),
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"奖励筛查失败：{log}", file=sys.stderr)
    else:
        record = json.loads(args.output.read_text())
        print(json.dumps(record["gate"], ensure_ascii=False, indent=2))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
