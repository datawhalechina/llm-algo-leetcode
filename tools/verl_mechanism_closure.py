"""Bounded verl-native update, frozen-trajectory rescoring and state evidence.

The notebook's driver stays dependency-light. GPU work only runs in explicit
worker mode in the existing PT-VERL-CUDA environment, one candidate at a time.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from itertools import accumulate
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verl_lifecycle_probe import MODEL_ID, SYSTEM_PROMPT, digest_json, build_arithmetic_tasks
from tools.verl_mechanism_audit import inspect_api, check_sft_loss_cpu
from tools.verl_system_mechanism_runtime import inspect_runtime, isolated_subprocess_env, VERL_COMMIT

SCHEMA = "verl-mechanism-closure/v1"
SOURCE = ROOT / "benchmarks/results/pt_verl_lifecycle/rtx5070ti_rollout_arithmetic_v1.json"
SEED = 42
TOLERANCES = {"same_version_max_abs": 0.15, "same_version_mean_abs": 0.03,
              "change_floor": 1e-5, "delta_relative_rmse": 0.5, "admission_mean_abs": 0.2}
TRAIN_TASKS = (("11 + 9 = ?", "20"), ("32 - 14 = ?", "18"),
               ("9 * 8 = ?", "72"), ("96 / 8 = ?", "12"))


def encode_prompt(tokenizer, messages) -> list[int]:
    # Match the historical vLLM path and avoid version-dependent BatchEncoding
    # returns from Transformers 5's tokenize=True chat-template API.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def write_once(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_frozen_source(path: Path) -> tuple[dict, list[dict]]:
    source = json.loads(path.read_text())
    config = source.get("config", {})
    if (source.get("status") != "ok" or config.get("model_id") != MODEL_ID
            or config.get("sampling_distribution") != "unconstrained"
            or config.get("task_pool_sha256") != digest_json(build_arithmetic_tasks())
            or any(config.get(k) != v for k, v in {"temperature": 1.0, "top_p": 1.0, "top_k": -1}.items())):
        raise ValueError("Frozen source does not match the ordinary arithmetic probe")
    if Path(config.get('resolved_snapshot', '')).name != config.get('model_revision'):
        raise ValueError('Snapshot path and recorded model revision disagree')
    groups = source.get("groups", [])
    expected = build_arithmetic_tasks()
    if [g["sample_id"] for g in groups] != [g["sample_id"] for g in expected]:
        raise ValueError("Frozen question identities are missing, repeated or reordered")
    trajectories = []
    for group, task in zip(groups, expected):
        if group["question"] != task["question"] or group["ground_truth"] != task["ground_truth"]:
            raise ValueError("Frozen task payload mismatch")
        if digest_json(group["messages"]) != group["prompt_sha256"]:
            raise ValueError("Frozen prompt hash mismatch")
        replies = group.get("responses", [])
        if len(replies) != 4:
            raise ValueError("Incomplete source group")
        reply = replies[0]  # Predeclared selection, independent of its reward.
        tokens, probs, mask = reply["token_ids"], reply["behavior_log_probs"], reply["action_mask"]
        if (reply["finish_reason"] != "stop" or not tokens or len(tokens) != len(probs)
                or len(mask) != len(tokens) or any(x != 1 for x in mask)
                or not all(math.isfinite(x) and x <= 1e-5 for x in probs)):
            raise ValueError("First frozen response is not a complete aligned trajectory")
        trajectories.append({"sample_id": group["sample_id"], "messages": group["messages"],
                             "response_ids": tokens, "action_mask": mask,
                             "response_text": reply['text'],
                             "behavior_log_probs": probs, "source_response_index": 0,
                             "behavior_policy_id": source["behavior_policy_id"]})
    if {q for q, _ in TRAIN_TASKS} & {g["question"] for g in groups}:
        raise ValueError("Training examples overlap frozen question probes")
    return source, trajectories


def compare_scores(left: list[list[float]], right: list[list[float]]) -> dict:
    if not left or len(left) != len(right):
        raise ValueError("Empty or mismatched score batch")
    errors = []
    for a, b in zip(left, right):
        if not a or len(a) != len(b):
            raise ValueError("Mismatched action positions")
        if not all(math.isfinite(x) and x <= 1e-5 for x in a + b):
            raise ValueError("Nonfinite score")
        errors.extend(abs(x - y) for x, y in zip(a, b))
    return {"tokens": len(errors), "mean_abs": sum(errors) / len(errors), "max_abs": max(errors)}


def decide_candidate(record: dict) -> dict:
    """Require real changes and differential sync evidence, not call counters."""
    s = record["scores"]
    if len(record['trajectories']) != len(s['behavior']):
        raise ValueError('Trajectory identity count and score count disagree')
    comparisons = {name: compare_scores(s[a], s[b]) for name, a, b in (
        ("source_vs_actor_before", "behavior", "actor_before"),
        ("same_version_before", "actor_before", "rollout_before"),
        ("same_version_after", "actor_after", "rollout_after"),
        ("actor_change", "actor_before", "actor_after"),
        ("rollout_change", "rollout_before", "rollout_after"),
    )}
    actor_delta = [y-x for a, b in zip(s["actor_before"], s["actor_after"]) for x, y in zip(a, b)]
    rollout_delta = [y-x for a, b in zip(s["rollout_before"], s["rollout_after"]) for x, y in zip(a, b)]
    delta_error = math.sqrt(sum((a-b)**2 for a, b in zip(actor_delta, rollout_delta)) / len(actor_delta))
    signal = math.sqrt(sum(a*a for a in actor_delta) / len(actor_delta))
    relative = delta_error / max(signal, TOLERANCES["change_floor"])
    update = record["update"]
    comparisons["delta_alignment"] = {"rmse": delta_error, "actor_delta_rms": signal, "relative_rmse": relative}
    checks = {
        "parameters_changed": update["optimizer_steps"] == 1 and update["same_names_shapes_dtype"]
        and math.isfinite(update["loss"]) and math.isfinite(update["grad_norm"]) and update["grad_norm"] > 0
        and math.isfinite(update["delta_l2"]) and update["delta_l2"] > 0 and update["changed_tensors"] > 0
        and update["before_sha256"] != update["after_sha256"],
        "same_version_probabilities": all(
            comparisons[k]["max_abs"] <= TOLERANCES["same_version_max_abs"]
            and comparisons[k]["mean_abs"] <= TOLERANCES["same_version_mean_abs"]
            for k in ("source_vs_actor_before", "same_version_before", "same_version_after")),
        "frozen_scores_changed": comparisons["actor_change"]["max_abs"] > TOLERANCES["change_floor"],
        "synchronized_change_visible": comparisons["rollout_change"]["max_abs"] > TOLERANCES["change_floor"]
        and relative <= TOLERANCES["delta_relative_rmse"],
        "sync_rpc_and_version": record["sync"]["rpc_completed"]
        and record['initial_sync']['rpc_completed'] and record['initial_sync']['version'] == 0
        and record['initial_sync']['sender_adapter_sha256'] == update['before_sha256']
        and record['sync']['version'] == 1
        and record["sync"]["sender_adapter_sha256"] == update["after_sha256"]
        and record["rollout_versions_before"] == [0] * len(s["behavior"])
        and record["rollout_versions_after"] == [1] * len(s["behavior"]),
        "real_generation_records": all(
            len(record[key]) == len(s['behavior']) and all(
                item['version'] == version and item['token_ids']
                and item['log_probs'] is not None
                and len(item['token_ids']) == len(item['log_probs'])
                and all(math.isfinite(x) and x <= 1e-5 for x in item['log_probs'])
                for item in record[key])
            for key, version in (('generation_before', 0), ('generation_after', 1))),
    }
    admission = []
    for trajectory, before, after in zip(record["trajectories"], s["behavior"], s["actor_after"]):
        gap = sum(abs(a-b) for a, b in zip(before, after)) / len(before)
        admission.append({"sample_id": trajectory["sample_id"], "version_lag": 1,
                          "mean_abs_log_prob_gap": gap,
                          "admitted": gap <= TOLERANCES["admission_mean_abs"]})
    return {"passed": all(checks.values()), "checks": checks, "comparisons": comparisons,
            "admission": admission, "tolerances": TOLERANCES,
            "scope": "single_rank_verl_supervised_update_and_probability_sync_not_grpo_or_atomic_activation"}


def build_config(verl_root: Path, snapshot: str, release: bool):
    from hydra import compose, initialize_config_dir
    overrides = [
        f"actor_rollout_ref.model.path={snapshot}", "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.lora_rank=16", "actor_rollout_ref.model.lora_alpha=16",
        "actor_rollout_ref.model.target_modules=[q_proj,v_proj]",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4", "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.optim.lr=0.001", "actor_rollout_ref.actor.optim.total_training_steps=1",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0", "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True", "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async", "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=False", "actor_rollout_ref.rollout.checkpoint_engine.backend=naive",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1", "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
        "actor_rollout_ref.rollout.max_model_len=512", "actor_rollout_ref.rollout.max_num_seqs=8",
        "actor_rollout_ref.rollout.max_num_batched_tokens=512", "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=False", "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1", "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.free_cache_engine={release}", "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1", "trainer.total_training_steps=1", "critic.enable=False", "algorithm.use_kl_in_reward=False",
    ]
    with initialize_config_dir(config_dir=str(verl_root / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


def make_batch(sequences: list[dict], pad_token_id: int, *, train: bool):
    """Full-sequence nested tensors; masks refer to target token positions."""
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    ids, positions, masks = [], [], []
    for item in sequences:
        prompt, reply = item["prompt_ids"], item["response_ids"]
        if not prompt or not reply or len(prompt) + len(reply) >= 512:
            raise ValueError("Invalid bounded sequence")
        ids.append(torch.tensor(prompt + reply, dtype=torch.long))
        positions.append(torch.arange(len(prompt) + len(reply)))
        masks.append(torch.tensor([0] * len(prompt) + [1] * len(reply)))
    offsets = torch.tensor([0] + list(accumulate(len(x) for x in ids)))
    nested = lambda rows: torch.nested.nested_tensor_from_jagged(torch.cat(rows), offsets)
    data = TensorDict({"input_ids": nested(ids), "position_ids": nested(positions), "loss_mask": nested(masks)}, [len(ids)])
    tu.assign_non_tensor(data, temperature=1.0, calculate_entropy=False, compute_loss=train,
                         pad_mode=DatasetPadMode.NO_PADDING, pad_token_id=pad_token_id,
                         global_token_num=[len(x) for x in ids])
    if train:
        tu.assign_non_tensor(data, mini_batch_size=len(ids), epochs=1, seed=SEED,
                             dataloader_kwargs={"shuffle": False})
    return data


def cpu_data_contract(args) -> dict:
    """Exercise the actual token, nested-batch and upstream score codecs on CPU."""
    import inspect
    import torch
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from tools.verl_mechanism_worker import MechanismWorker, response_scores, seeded_initialization
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs
    source, trajectories = load_frozen_source(args.source)
    snapshot = source['config']['resolved_snapshot']
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    for trajectory in trajectories:
        trajectory['prompt_ids'] = encode_prompt(tokenizer, trajectory['messages'])
        if tokenizer.decode(trajectory['response_ids'], skip_special_tokens=True) != trajectory['response_text']:
            raise ValueError('Frozen reply Token IDs do not decode to the saved text')
        if trajectory['response_ids'][-1] != tokenizer.eos_token_id:
            raise ValueError('The source response must preserve its terminating EOS')
    batch = make_batch(trajectories, tokenizer.pad_token_id, train=False)
    lengths = [len(t['prompt_ids']) for t in trajectories]
    replies = [len(t['response_ids']) for t in trajectories]
    rows = [-(torch.arange(p+r).float()+1) / 100 for p, r in zip(lengths, replies)]
    offsets = torch.tensor([0] + list(accumulate(len(row) for row in rows)))
    scores = response_scores(torch.nested.nested_tensor_from_jagged(torch.cat(rows), offsets), lengths, replies)
    assert scores == [row[p-1:p+r-1].tolist() for row, p, r in zip(rows, lengths, replies)]
    for trajectory, row in zip(trajectories, rows):
        full = trajectory['prompt_ids'] + trajectory['response_ids']
        output = SimpleNamespace(prompt_logprobs=[None]+[
            {token: SimpleNamespace(logprob=float(row[i]), rank=1)} for i, token in enumerate(full[1:])])
        extra = {}
        extract_prompt_logprobs(output, 0, extra)
        start = len(trajectory['prompt_ids'])-1
        assert [x[0] for x in extra['prompt_ids'][start:start+len(trajectory['response_ids'])]] == trajectory['response_ids']
    training = [{'prompt_ids':encode_prompt(tokenizer, [{'role':'system','content':SYSTEM_PROMPT},
                                                        {'role':'user','content':q}]),
                 'response_ids':tokenizer.encode(a, add_special_tokens=False)+[tokenizer.eos_token_id]}
                for q, a in TRAIN_TASKS]
    train = make_batch(training, tokenizer.pad_token_id, train=True)
    assert int(train['loss_mask'].sum()) == sum(len(t['response_ids']) for t in training)
    for release in (True, False):
        config = build_config(args.verl_root, snapshot, release)
        model = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
        assert model.lora_rank == 16 and not model.use_remove_padding
        assert actor.optim.total_training_steps == 1
        assert rollout.free_cache_engine == release and rollout.checkpoint_engine.backend == 'naive'
    assert inspect.iscoroutinefunction(MechanismWorker.closure_sync)
    from peft import LoraConfig, get_peft_model
    parameters = {}

    def build_cpu_adapter():
        module = torch.nn.Module()
        module.q_proj = torch.nn.Linear(8, 8, bias=False)
        module.v_proj = torch.nn.Linear(8, 8, bias=False)
        model = get_peft_model(module, LoraConfig(r=16, lora_alpha=16,
                                                 target_modules=['q_proj', 'v_proj'], bias='none'))
        parameters.clear()
        parameters.update({name: value.detach().clone() for name, value in model.named_parameters()
                           if 'lora_' in name})
        assert len(parameters) == 4

    initial = seeded_initialization(build_cpu_adapter, lambda: parameters, SEED)
    torch.rand(127)  # Different prior RNG history must not change initialization.
    paired = seeded_initialization(build_cpu_adapter, lambda: parameters, SEED, initial['adapter_sha256'])
    assert paired['paired_hash_matched']
    try:
        seeded_initialization(build_cpu_adapter, lambda: parameters, SEED, 'wrong-hash')
    except RuntimeError as error:
        assert 'training not started' in str(error)
    else:
        raise AssertionError('Initial parameter mismatch must stop the paired run')
    return {'cpu_loss':check_sft_loss_cpu(), 'batch_layout':'nested_full_sequence',
            'probe_count':len(trajectories), 'supervised_steps':1,
            'checks':['actual_model_tokenizer_and_eos', 'actor_score_shift', 'verl_rollout_prompt_score_shift',
                      'training_action_mask', 'both_candidates_dataclass_config', 'worker_async_rpc_signature',
                      'seeded_adapter_reproducibility', 'initial_hash_mismatch_blocks_training'],
            'gpu_validated':False}


class ResourceTrace:
    def __init__(self):
        self.events, self.samples, self.errors = [], [], []
        self.stop = threading.Event()

    def sample(self):
        try:
            info = dict(line.split(":", 1) for line in Path('/proc/meminfo').read_text().splitlines())
            memory = {k: int(v.split()[0]) / 1024 for k, v in info.items() if k in ('MemTotal', 'MemAvailable')}
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
                                 capture_output=True, text=True, timeout=3)
            self.samples.append({"monotonic_ns": time.monotonic_ns(),
                                 "gpu_used_mb": float(gpu.stdout.strip()) if gpu.returncode == 0 else None,
                                 "host_used_mb": memory['MemTotal'] - memory['MemAvailable']})
        except Exception as error:
            self.errors.append(str(error))

    def loop(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(0.25)

    @contextmanager
    def phase(self, name):
        event = {"name": name, "begin_ns": time.monotonic_ns()}
        try:
            yield
        finally:
            event['end_ns'] = time.monotonic_ns()
            event['duration_s'] = (event['end_ns'] - event['begin_ns']) / 1e9
            self.events.append(event)

    def report(self):
        for e in self.events:
            samples = [s for s in self.samples if e['begin_ns'] <= s['monotonic_ns'] <= e['end_ns']]
            values = [s['gpu_used_mb'] for s in samples if s['gpu_used_mb'] is not None]
            e.update(sample_count=len(samples), sampled_gpu_peak_mb=max(values) if values else None)
        return {"clock": "single_host_monotonic_ns", "interval_s": 0.25,
                "scope": "whole_gpu_and_whole_host_including_other_apps",
                "events": self.events, "samples": self.samples, "errors": self.errors,
                "limitation": "短阶段可能漏采峰值；边界事件不是持续采样峰值，未测到记为 null"}


def one(value):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("Expected exactly one verl worker result")
    return value[0]


def system_decision(candidates: dict, resources: dict) -> dict:
    """Check stage evidence, without requiring memory savings or speedups."""
    checks, measurements = {}, {}
    for name in ('release', 'retain'):
        record, trace = candidates.get(name, {}), resources.get(name, {})
        events = {e['name']: e for e in trace.get('events', [])}
        required = ('generation_before', 'actor_score_before', 'supervised_update',
                    'actor_score_after', 'weight_sync', 'generation_after', 'sleep_before_actor')
        checks[name + '_update_and_sync'] = bool(record.get('decision', {}).get('passed'))
        checks[name + '_stage_events'] = all(
            key in events and events[key]['end_ns'] >= events[key]['begin_ns'] for key in required)
        worker_events = record.get('worker_events', [])
        checks[name + '_actual_sync_events'] = all(any(
            e['name'] == event and e['edge'] == edge for e in worker_events)
            for event in ('verl_weight_sync', 'rollout_weight_transfer')
            for edge in ('begin', 'end'))
        resume_events = [e for e in worker_events if e['name'] == 'rollout_resume']
        # Retain mode intentionally does not call resume: absence is expected,
        # not missing evidence of a restoration that should have happened.
        checks[name + '_restore_matches_mode'] = (all(any(
            e['edge'] == edge and e.get('tags') == [tag] for e in resume_events)
            for edge in ('begin', 'end') for tag in ('weights', 'kv_cache'))
            if name == 'release' else not resume_events)
        samples = trace.get('samples', [])
        valid = [s for s in samples if s.get('gpu_used_mb') is not None]
        checks[name + '_resource_samples'] = bool(valid) and all(
            math.isfinite(s['gpu_used_mb']) and math.isfinite(s['host_used_mb']) for s in valid)
        measurements[name] = {
            'sampled_gpu_peak_mb': max((s['gpu_used_mb'] for s in valid), default=None),
            'sampled_host_peak_mb': max((s['host_used_mb'] for s in samples), default=None),
            'phases': [{k: e.get(k) for k in ('name', 'duration_s', 'sample_count', 'sampled_gpu_peak_mb')}
                       for e in trace.get('events', [])],
        }
    configs = [candidates.get(n, {}).get('resolved_config') for n in ('release', 'retain')]
    if all(configs):
        # Exclude only the declared treatment; all remaining model/update settings must match.
        normalized = json.loads(json.dumps(configs))
        treatments = [c['actor_rollout_ref']['rollout'].pop('free_cache_engine') for c in normalized]
        checks['paired_config'] = treatments == [True, False] and normalized[0] == normalized[1]
        checks['same_frozen_source'] = (candidates['release']['source_sha256']
                                        == candidates['retain']['source_sha256'])
    else:
        checks['paired_config'] = checks['same_frozen_source'] = False
    initial = [candidates.get(n, {}).get('initialization', {}) for n in ('release', 'retain')]
    updates = [candidates.get(n, {}).get('update', {}) for n in ('release', 'retain')]
    hashes = [r.get('adapter_sha256') for r in initial]
    checks['paired_initial_parameters'] = (bool(hashes[0]) and hashes[0] == hashes[1]
        and initial[0].get('seed') == initial[1].get('seed') == SEED
        and initial[1].get('expected_sha256') == hashes[0]
        and initial[1].get('paired_hash_matched') is True
        and all(r.get('before_sha256') == h for r, h in zip(updates, hashes)))
    return {'passed': all(checks.values()), 'checks': checks, 'measurements': measurements,
            'scope': 'single_gpu_stage_and_state_observation', 'steady_state_speedup_proven': False,
            'limitation': '整卡与整机采样包含其他应用；短阶段峰值可能漏采，差值不等于 KV Cache 大小'}


def read_summary(path: Path, scope: str) -> dict:
    """Read existing results without relying on notebook execution state."""
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('schema_version') != SCHEMA or scope not in ('lifecycle', 'system'):
        raise ValueError('Unexpected result schema or page scope')
    if scope == 'system':
        return report['system_decision']
    return {name: {'status': r['status'], 'decision': r.get('decision'),
                   'update': r.get('update'), 'reason': r.get('reason'), 'error': r.get('error')}
            for name, r in report['candidates'].items()}


def run_worker(args) -> dict:
    if args.candidate == 'retain' and not args.expected_initial_sha256:
        raise ValueError('Retain confirmation requires the release initial adapter hash')
    import ray
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from tools.verl_mechanism_worker import MechanismWorker
    from verl.single_controller.ray import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
    from verl.workers.rollout.llm_server import LLMServerManager
    from verl.checkpoint_engine.base import CheckpointEngineManager
    from verl.utils.config import omega_conf_to_dataclass
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    source, trajectories = load_frozen_source(args.source)
    snapshot = source['config']['resolved_snapshot']
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    for trajectory in trajectories:
        trajectory['prompt_ids'] = encode_prompt(tokenizer, trajectory['messages'])
        trajectory['trajectory_sha256'] = digest_json({k: trajectory[k] for k in ('prompt_ids', 'response_ids', 'action_mask')})
    training = []
    for question, answer in TRAIN_TASKS:
        prompt = encode_prompt(tokenizer, [{'role':'system','content':SYSTEM_PROMPT},
                                           {'role':'user','content':question}])
        training.append({'prompt_ids':prompt, 'response_ids':tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]})
    config = build_config(args.verl_root, snapshot, args.candidate == 'release')
    runtime_env = get_ppo_ray_runtime_env(config)
    runtime_env['py_executable'] = str(args.verl_root / '.venv/bin/python')
    runtime_env['env_vars'].update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    resources = ResourceTrace()
    sampler = threading.Thread(target=resources.loop, daemon=True)
    sampler.start()
    try:
        with resources.phase('initialization'):
            ray.init(address='local', namespace='mechanism_' + args.run_label + '_' + args.candidate,
                     runtime_env=runtime_env, object_store_memory=536870912)
            pool = RayResourcePool([1], use_gpu=True, max_colocate_count=1, name_prefix='mechanism_' + args.candidate)
            group = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=RayClassWithInitArgs(
                cls=ray.remote(MechanismWorker), config=config.actor_rollout_ref, role='actor_rollout'))
            initialization = one(group.closure_initialize(SEED, args.expected_initial_sha256))
            servers = LLMServerManager.create(config=config, worker_group=group, rollout_resource_pool=pool)
            checkpoint = CheckpointEngineManager(
                config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
                actor_wg=group, replicas=servers.get_replicas())
        with resources.phase('sleep_before_initial_sync'):
            checkpoint.sleep_replicas()
        with resources.phase('initial_sync'):
            initial_sync = one(group.closure_sync(0))

        def rollout_scores(version):
            values, versions = [], []
            server = servers.server_handles[0]
            for index, trajectory in enumerate(trajectories):
                full = trajectory['prompt_ids'] + trajectory['response_ids']
                output = ray.get(server.generate.remote(
                    prompt_ids=full, sampling_params={'max_tokens':1,'temperature':1.0,'top_p':1.0,
                                                      'top_k':-1,'prompt_logprobs':0,'logprobs':False},
                    request_id=f'fixed-{version}-{index}'))
                extra = output.extra_fields
                start = len(trajectory['prompt_ids']) - 1
                rows = extra['prompt_ids'][start:start+len(trajectory['response_ids'])]
                if [r[0] for r in rows] != trajectory['response_ids']:
                    raise ValueError('verl rollout prompt probability positions do not match frozen response')
                values.append([r[0] for r in extra['prompt_logprobs'][start:start+len(rows)]])
                versions.append(extra.get('global_steps'))
            return values, versions

        def generate(version):
            replies = []
            for i, trajectory in enumerate(trajectories):
                result = ray.get(servers.server_handles[0].generate.remote(
                    prompt_ids=trajectory['prompt_ids'],
                    sampling_params={'max_tokens':16,'temperature':1.0,'top_p':1.0,'top_k':-1,'seed':SEED+i,'logprobs':True},
                    request_id=f'generation-{version}-{i}'))
                replies.append({'sample_id':trajectory['sample_id'], 'token_ids':result.token_ids,
                                'log_probs':result.log_probs, 'stop_reason':result.stop_reason,
                                'version':result.extra_fields.get('global_steps')})
            return replies

        with resources.phase('generation_before'):
            generation_before = generate(0)
        with resources.phase('rollout_score_before'):
            rollout_before, versions_before = rollout_scores(0)
        with resources.phase('sleep_before_actor'):
            checkpoint.sleep_replicas()
        prompts = [len(t['prompt_ids']) for t in trajectories]
        replies = [len(t['response_ids']) for t in trajectories]

        def actor_scores():
            return one(group.closure_score(make_batch(trajectories, tokenizer.pad_token_id, train=False), prompts, replies))['log_probs']

        with resources.phase('actor_score_before'):
            actor_before = actor_scores()
        with resources.phase('supervised_update'):
            update = one(group.closure_update(make_batch(training, tokenizer.pad_token_id, train=True)))
        with resources.phase('actor_score_after'):
            actor_after = actor_scores()
        with resources.phase('weight_sync'):
            sync = one(group.closure_sync(1))
        with resources.phase('rollout_score_after'):
            rollout_after, versions_after = rollout_scores(1)
        with resources.phase('generation_after'):
            generation_after = generate(1)
        events = one(group.closure_events())
        record = {
            'schema_version':SCHEMA, 'run_label':args.run_label, 'candidate':args.candidate, 'status':'measured',
            'verl_commit':VERL_COMMIT, 'source_path':str(args.source),
            'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
            'resolved_config':OmegaConf.to_container(config, resolve=True),
            'trajectories':trajectories, 'training_examples':training, 'training_task_sha256':digest_json(TRAIN_TASKS),
            'initialization':initialization, 'update':update, 'initial_sync':initial_sync, 'sync':sync,
            'scores':{'behavior':[t['behavior_log_probs'] for t in trajectories],
                      'actor_before':actor_before, 'actor_after':actor_after,
                      'rollout_before':rollout_before, 'rollout_after':rollout_after},
            'rollout_versions_before':versions_before, 'rollout_versions_after':versions_after,
            'generation_before':generation_before, 'generation_after':generation_after,
            'worker_events':events,
            'resource_path':str(args.output.with_suffix('.resources.json')),
            'limitations':['监督更新用于机制，不证明 GRPO 或质量提升',
                           '接收侧身份以实际 RPC、版本与概率差分核对；没有直接导出接收侧权重哈希',
                           '单卡不验证原子激活、并发消费、网络队列或分布式恢复'],
        }
        record['decision'] = decide_candidate(record)
        return record
    finally:
        if ray.is_initialized():
            ray.shutdown()
        resources.stop.set()
        sampler.join(timeout=4)
        # Kept in a sidecar even if a worker fails before a main record is made.
        write_once(args.output.with_suffix('.resources.json'), resources.report())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight','run','worker','cpu_check'], default='preflight')
    parser.add_argument('--verl-root', type=Path, required=True)
    parser.add_argument('--run-label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--candidate', choices=['release','retain'], default='release')
    parser.add_argument('--expected-initial-sha256', help='Internal paired-run gate; not a learner setting')
    args = parser.parse_args()
    for name in ('verl_root','output','source'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.mode == 'cpu_check':
        print(json.dumps(cpu_data_contract(args),ensure_ascii=False))
        return 0
    if args.mode == 'worker':
        if args.output.exists() or args.output.with_suffix('.resources.json').exists():
            raise FileExistsError(f'保留已有子实验结果：{args.output}')
        try:
            result = run_worker(args)
        except Exception as error:
            import traceback
            traceback.print_exc()
            write_once(args.output, {'schema_version':SCHEMA,'run_label':args.run_label,
                                    'candidate':args.candidate,'status':'failed','error':repr(error)})
            return 1
        write_once(args.output, result)
        return 0
    source, trajectories = load_frozen_source(args.source)
    api = inspect_api(args.verl_root)
    preflight = inspect_runtime(args.verl_root, None, None, require_data_files=False)
    environment = isolated_subprocess_env()
    environment['PYTHONPATH'] = str(ROOT)  # Only this local package, not unrelated Python sites.
    command = [str(args.verl_root / '.venv/bin/python'), '-m', 'tools.verl_mechanism_closure',
               '--verl-root',str(args.verl_root),'--run-label',args.run_label,'--output',str(args.output),
               '--source',str(args.source)]
    cpu = subprocess.run(command+['--mode','cpu_check'], cwd=args.verl_root,
                         env=environment, capture_output=True, text=True)
    ready = preflight['ready'] and api['inspection_passed'] and cpu.returncode == 0
    preparation = {'schema_version':SCHEMA,'run_label':args.run_label,'preflight':preflight,
                      'interfaces_found':api['inspection_passed'], 'cpu_check_returncode':cpu.returncode,
                      'cpu_check_output':cpu.stdout[-3000:], 'cpu_check_error':cpu.stderr[-3000:] if cpu.returncode else None,
                      'ready_for_first_gpu_run':ready, 'gpu_validated':False,
                      'source':str(args.source),'probe_count':len(trajectories),'tolerances':TOLERANCES}
    print(json.dumps(preparation, ensure_ascii=False,indent=2),flush=True)
    if not ready:
        return 2
    if args.mode == 'preflight':
        return 0
    paths = {c:args.output.with_name(args.output.stem+'_'+c+'.json') for c in ('release','retain')}
    for path in [args.output, *paths.values()]:
        if path.exists() or path.with_suffix('.log').exists() or path.with_suffix('.resources.json').exists():
            raise FileExistsError(f'保留已有结果，请读取它或更换标签：{path}')
    results, resource_records = {}, {}
    for candidate, path in paths.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        worker_command = command.copy()
        worker_command[worker_command.index('--output')+1] = str(path)
        if candidate == 'retain':
            worker_command += ['--expected-initial-sha256', results['release']['initialization']['adapter_sha256']]
        with path.with_suffix('.log').open('x') as log:
            completed = subprocess.run(worker_command+['--mode','worker','--candidate',candidate],
                                       cwd=args.verl_root,env=environment,stdout=log,stderr=subprocess.STDOUT)
        results[candidate] = json.loads(path.read_text()) if path.is_file() else {'status':'failed','returncode':completed.returncode}
        resource_path = path.with_suffix('.resources.json')
        if resource_path.is_file():
            resource_records[candidate] = json.loads(resource_path.read_text())
        if candidate == 'release' and (completed.returncode or not results[candidate].get('decision',{}).get('passed')):
            results['retain'] = {'status':'skipped','reason':'release_update_or_sync_not_validated'}
            break
    paired_decision = system_decision(results, resource_records)
    write_once(args.output, {'schema_version':SCHEMA,'run_label':args.run_label,'candidates':results,
                            'preparation':preparation,
                            'code_sha256':{name:hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()
                                           for name in ('verl_mechanism_closure.py','verl_mechanism_worker.py')},
                            'system_decision':paired_decision,
                            'scope':'single_gpu_verl_native_supervised_mechanisms_not_grpo',
                            'steady_state_speedup_proven':False})
    print(f'结果已保存：{args.output}')
    print(json.dumps({'candidates': {name: {'status': r.get('status'), 'error': r.get('error'),
                       'reason': r.get('reason'), 'checks': r.get('decision', {}).get('checks')}
                       for name, r in results.items()},
                      'paired_initial_parameters': paired_decision['checks']['paired_initial_parameters'],
                      'system_passed': paired_decision['passed']}, ensure_ascii=False, indent=2))
    return 0 if all(r.get('status') in ('measured', 'skipped') for r in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
