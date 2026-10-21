"""Instrumented single-rank verl worker; imported only in PT-VERL-CUDA.

The inherited verl methods perform inference, training and weight transfer.
The added RPCs preserve exact token positions and collect bounded evidence.
"""

from contextlib import contextmanager
from functools import partial
import hashlib
import math
import os
import time

import torch

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.losses import sft_loss


def parameter_fingerprint(parameters: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def seeded_initialization(build, parameters, seed: int, expected_sha256: str | None = None) -> dict:
    """Seed before LoRA creation; a paired mismatch stops before any training."""
    from transformers import set_seed

    set_seed(seed)
    build()
    actual = parameter_fingerprint(parameters())
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"Initial LoRA mismatch: expected={expected_sha256}, actual={actual}; training not started")
    return {'seed': seed, 'adapter_sha256': actual, 'expected_sha256': expected_sha256,
            'paired_hash_matched': actual == expected_sha256 if expected_sha256 is not None else None,
            'method': 'seed_before_verl_init_and_verify_exported_adapter'}


def scalar_metric(value) -> float:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Expected one finite metric from exactly one optimizer step")
    return float(value)


def response_scores(log_probs, prompt_lengths: list[int], response_lengths: list[int]) -> list[list[float]]:
    """verl log_probs[i] predicts input_ids[i+1], including response EOS."""
    rows = log_probs.unbind()
    if len(rows) != len(prompt_lengths) or len(rows) != len(response_lengths):
        raise ValueError("Scored batch size mismatch")
    result = []
    for row, prompt, response in zip(rows, prompt_lengths, response_lengths):
        if prompt < 1 or response < 1 or row.numel() != prompt + response:
            raise ValueError("Unexpected full-sequence score shape")
        scores = row[prompt - 1:prompt + response - 1].float().tolist()
        if len(scores) != response or not all(math.isfinite(x) and x <= 1e-5 for x in scores):
            raise ValueError("Invalid response log probabilities")
        result.append(scores)
    return result


class MechanismWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def closure_initialize(self, seed: int, expected_sha256: str | None = None):
        return seeded_initialization(super().init_model, self._parameters, seed, expected_sha256)

    def _mark(self, name: str, edge: str, **extra):
        torch.cuda.synchronize()
        if not hasattr(self, "_closure_events"):
            self._closure_events = []
        self._closure_events.append({
            "name": name, "edge": edge, "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(), "actor_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "actor_reserved_mb": torch.cuda.memory_reserved() / 1024**2, **extra,
        })

    @contextmanager
    def _phase(self, name: str):
        self._mark(name, "begin")
        try:
            yield
        finally:
            self._mark(name, "end")

    def _parameters(self) -> dict:
        if torch.distributed.get_world_size() != 1:
            raise RuntimeError("This probe only supports the pinned single-rank FSDP path")
        iterator, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=False, base_sync_done=True,
        )
        if peft_config is None:
            raise RuntimeError("Expected unmerged LoRA parameters, not full base weights")
        parameters = {}
        for name, tensor in iterator:
            if "lora_" not in name:
                raise RuntimeError(f"Unexpected non-adapter tensor: {name}")
            parameters[name] = tensor.detach().cpu().clone()
        if not parameters or not all(torch.isfinite(t).all().item() for t in parameters.values()):
            raise RuntimeError("Missing or nonfinite adapter parameters")
        return parameters

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def closure_score(self, data, prompt_lengths, response_lengths):
        with self._phase("actor_score"):
            output = self.compute_log_prob(data)
        return {"log_probs": response_scores(tu.get(output, "log_probs"), prompt_lengths, response_lengths)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def closure_update(self, data):
        before = self._parameters()
        self.set_loss_fn(partial(sft_loss, config=None))
        with self._phase("supervised_update"):
            output = self.update_actor(data)
        after = self._parameters()
        if before.keys() != after.keys() or any(
            before[k].dtype != after[k].dtype or before[k].shape != after[k].shape for k in before
        ):
            raise RuntimeError("Parameter names, dtype or shapes changed across the step")
        delta_squared = sum((after[k].double() - before[k].double()).square().sum().item() for k in before)
        metrics = tu.get(output, "metrics")
        return {
            "optimizer_steps": 1, "loss_kind": "verl_builtin_sft_loss",
            "training_examples": int(data.batch_size[0]),
            "supervised_tokens": int(tu.get(data, "loss_mask").values().sum().item()),
            "loss": scalar_metric(metrics["loss"]),
            "grad_norm": scalar_metric(metrics["grad_norm"]),
            "before_sha256": parameter_fingerprint(before),
            "after_sha256": parameter_fingerprint(after),
            "delta_l2": math.sqrt(delta_squared),
            "changed_tensors": sum(not torch.equal(before[k], after[k]) for k in before),
            "parameter_tensors": len(before), "parameter_numel": sum(t.numel() for t in before.values()),
            "same_names_shapes_dtype": True,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def closure_sync(self, version: int):
        # Observe actual inherited calls, without replacing their implementation.
        original_resume = self.rollout.resume
        original_update = self.rollout.update_weights

        async def resume(*args, **kwargs):
            self._mark("rollout_resume", "begin", tags=kwargs.get("tags"))
            try:
                return await original_resume(*args, **kwargs)
            finally:
                self._mark("rollout_resume", "end", tags=kwargs.get("tags"))

        async def update(*args, **kwargs):
            self._mark("rollout_weight_transfer", "begin")
            try:
                return await original_update(*args, **kwargs)
            finally:
                self._mark("rollout_weight_transfer", "end")

        self.rollout.resume, self.rollout.update_weights = resume, update
        self._mark("verl_weight_sync", "begin", version=version)
        try:
            await self.update_weights(global_steps=version, mode="naive")
        finally:
            self.rollout.resume, self.rollout.update_weights = original_resume, original_update
            self._mark("verl_weight_sync", "end", version=version)
        return {"rpc_completed": True, "version": version,
                "sender_adapter_sha256": parameter_fingerprint(self._parameters())}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def closure_events(self):
        return getattr(self, "_closure_events", [])
