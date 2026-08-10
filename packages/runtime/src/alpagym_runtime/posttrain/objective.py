# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpaGym's selected-action replay PPO surrogate, as a posttrain Objective.

posttrain's rule is ``trainer = engine, objective = algorithm``: ``LLMTrainer`` never inspects the
model's outputs, it just calls ``objective(model, batch)``. So AlpaGym's forward contract --

    model(**model_inputs) -> {"log_probs": Tensor[R], "kl_div": Tensor[R] | None}

lives here, not in an engine. The surrogate itself is ``alpagym_runtime.cosmos.replay_objective``,
reused unmodified: it is already pure torch and already normalizes over valid rows, which is
exactly what ``Reduction.MEAN`` declares.

Advantages are READ, never derived: posttrain writes ``transition.advantage`` upstream at reward
time (``algorithm/annotators/advantage.py::group_relative``), broadcasting one episode-level
advantage across all of that episode's transitions.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from projects.cosmos3.posttrain.algorithm.annotators.advantage import require_advantages
from projects.cosmos3.posttrain.algorithm.objective import Objective, Reduction
from projects.cosmos3.posttrain.schema import Trajectory

from alpagym_runtime.cosmos.replay_objective import (
    assert_replay_shapes,
    compute_kl_penalty,
    compute_ppo_surrogate,
)


def make_alpagym_objective(
    build_model_inputs: Callable[[Any], tuple[dict[str, Any], torch.Tensor]],
    *,
    ratio_clip_low: float,
    ratio_clip_high: float,
    kl_beta: float,
    reference_reset_interval: int,
) -> Objective:
    """Bind the replay PPO surrogate to one policy family's model-input dialect.

    Args:
        build_model_inputs: the policy bundle's hook, mapping one replay payload to
            ``(model_forward_kwargs, old_logprob)``. Each policy owns its own dialect; this
            objective stays policy-agnostic by receiving it.
        ratio_clip_low: PPO epsilon below 1.0.
        ratio_clip_high: PPO epsilon above 1.0.
        kl_beta: KL penalty weight. Must be 0.0 in milestone 1 (no reference model).
        reference_reset_interval: steps between reference resets. Must be 0 in milestone 1.

    Returns:
        An ``Objective`` declaring ``Reduction.MEAN``.

    Raises:
        NotImplementedError: ``kl_beta > 0`` or ``reference_reset_interval > 0``. The reference
            model is not ported yet, and computing a KL-free loss under a config that asked for KL
            would be a silent wrong answer.
    """
    if kl_beta > 0.0 or reference_reset_interval > 0:
        raise NotImplementedError(
            "the reference model is not ported to posttrain yet (milestone 1): "
            f"kl_beta={kl_beta}, reference_reset_interval={reference_reset_interval}. "
            "Set both to 0, or port the reference model first."
        )

    def score(model: Any, batch: list[Trajectory]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Rescore every recorded action in ``batch`` and apply the clipped PPO surrogate."""
        require_advantages(batch)

        rows = [transition for trajectory in batch for transition in trajectory.transitions]
        if not rows:
            raise ValueError("empty batch reached the AlpaGym objective")

        model_inputs: list[dict[str, Any]] = []
        old_logprobs: list[torch.Tensor] = []
        for transition in rows:
            inputs, old_logprob = build_model_inputs(transition.algo_extra["payload"])
            model_inputs.append(inputs)
            old_logprobs.append(old_logprob)

        device = next(model.parameters()).device
        collated = {
            key: torch.stack([torch.as_tensor(inputs[key]) for inputs in model_inputs]).to(device)
            for key in model_inputs[0]
        }
        old = torch.stack([torch.as_tensor(v) for v in old_logprobs]).to(device).float()
        advantages = torch.tensor(
            [float(transition.advantage) for transition in rows],
            device=device,
            dtype=torch.float32,
        )
        # Every row here is a real recorded step. Padding is a cosmos-rl minibatching artifact that
        # posttrain's batching does not produce, so the mask is all-False rather than absent -- the
        # surrogate and the KL term both reduce over it.
        is_padding = torch.zeros_like(old, dtype=torch.bool)

        result = model(**collated)
        new = result["log_probs"]
        kl_div = result.get("kl_div")
        assert_replay_shapes(new, old, advantages, kl_div)

        policy_loss, ratio = compute_ppo_surrogate(
            new,
            old,
            advantages,
            ratio_clip_low=ratio_clip_low,
            ratio_clip_high=ratio_clip_high,
            is_padding=is_padding,
        )
        kl_loss = compute_kl_penalty(kl_div, is_padding, kl_beta=kl_beta, device=device)
        loss = policy_loss + kl_loss

        with torch.no_grad():
            clipped = (ratio < 1.0 - ratio_clip_low) | (ratio > 1.0 + ratio_clip_high)
            metrics = {
                "policy_loss": float(policy_loss.detach()),
                "kl_loss": float(kl_loss.detach()),
                "ratio_min": float(ratio.min()),
                "ratio_max": float(ratio.max()),
                "clip_fraction": float(clipped.float().mean()),
                "advantage_mean": float(advantages.mean()),
                "rows": len(rows),
            }
        return loss, metrics

    return Objective(fn=score, reduction=Reduction.MEAN)
