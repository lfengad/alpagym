# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AlpaGym's selected-action replay PPO surrogate as a posttrain Objective.

The objective READS ``transition.advantage`` -- written upstream at reward time by
``algorithm/annotators/advantage.py`` -- and does not derive it. It declares ``Reduction.MEAN``
because ``compute_ppo_surrogate`` already normalizes over valid rows and returns a scalar.

Milestone 1 does not port the reference model, so a config asking for KL must fail loud rather
than silently training KL-free.
"""

from types import SimpleNamespace

import pytest
import torch
from alpagym_runtime.posttrain.objective import make_alpagym_objective
from projects.cosmos3.posttrain.algorithm.primitives.objective import Reduction
from projects.cosmos3.posttrain.schema import ObsBundle, Trajectory, Transition

_CLIPS = {"ratio_clip_low": 0.2, "ratio_clip_high": 0.2}
_OFF = {"kl_beta": 0.0, "reference_reset_interval": 0}


def _build_model_inputs(replay_data):
    """Stand-in for the policy bundle's hook, which takes the whole replay ENVELOPE."""
    return {"row": replay_data.payload["row"]}, replay_data.old_logprob


class _ShiftModel(torch.nn.Module):
    """Returns log_probs shifted by a trainable scalar, so the loss carries a real gradient."""

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(shift))

    def forward(self, row, **kwargs):
        return {"log_probs": row + self.shift, "kl_div": None}


def _trajectory(n_steps: int, advantage: float, old_logprob: float = 0.0) -> Trajectory:
    """One episode's worth of replay rows, all carrying the same episode-level advantage."""
    obs = ObsBundle(representations={"scene_id": "s"}, id="p")
    transitions = []
    for i in range(n_steps):
        transition = Transition(
            obs=obs,
            policy_info={"logprob": torch.tensor(old_logprob)},
            algo_extra={
                "replay_data": SimpleNamespace(
                    payload={"row": torch.tensor(float(old_logprob))},
                    old_logprob=torch.tensor(old_logprob),
                )
            },
            step_idx=i,
        )
        transition.advantage = advantage
        transitions.append(transition)
    return Trajectory(transitions=transitions, id="t", group_id="p")


@pytest.mark.parametrize(
    "override", [{"kl_beta": 0.1}, {"reference_reset_interval": 3}], ids=["kl", "ref_reset"]
)
def test_kl_and_reference_reset_are_refused_in_milestone_1(override) -> None:
    """A config that asks for KL must not silently get a KL-free loss."""
    with pytest.raises(NotImplementedError, match="reference model"):
        make_alpagym_objective(_build_model_inputs, **_CLIPS, **{**_OFF, **override})


def test_declares_mean_reduction() -> None:
    """compute_ppo_surrogate already divides by the valid-row count, which IS the MEAN contract."""
    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    assert objective.reduction is Reduction.MEAN


def test_zero_advantage_gives_zero_loss() -> None:
    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    loss, _ = objective(_ShiftModel(), [_trajectory(4, advantage=0.0)])
    assert torch.allclose(loss, torch.zeros(()))


def test_loss_is_finite_and_differentiable() -> None:
    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    model = _ShiftModel(shift=0.1)
    loss, metrics = objective(model, [_trajectory(4, advantage=1.0)])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.shift.grad is not None and torch.isfinite(model.shift.grad)
    # Namespaced: `schema/metrics.py::validate_public_metric_keys` rejects a bare key, and the
    # reporter validates every frame -- a rename here fails the run at the first report, not at
    # the first metric read.
    assert set(metrics) >= {"train/ratio_min", "train/ratio_max", "train/clip_fraction", "train/rows"}


def test_positive_advantage_pushes_logprob_up() -> None:
    """Sign check: with a positive advantage the surrogate must reward raising the logprob.

    Without this the clipping and the negation could both be wrong and still produce a finite,
    differentiable loss -- which every other test here would accept.
    """
    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    model = _ShiftModel(shift=0.0)
    loss, _ = objective(model, [_trajectory(4, advantage=1.0)])
    loss.backward()
    # d(loss)/d(shift) < 0 means gradient DESCENT increases shift, i.e. raises log_probs.
    assert model.shift.grad.item() < 0


def test_rows_are_flattened_across_trajectories() -> None:
    """The batch is a list of trajectories; every transition of every one is a scored row."""
    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    _, metrics = objective(
        _ShiftModel(), [_trajectory(4, advantage=1.0), _trajectory(3, advantage=1.0)]
    )
    assert metrics["train/rows"] == 7


def test_missing_advantage_raises_rather_than_defaulting_to_zero() -> None:
    """A missing advantage is a wiring bug; treating it as 0.0 would silently drop the row."""
    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    trajectory = _trajectory(2, advantage=0.0)
    trajectory.transitions[0].advantage = None
    with pytest.raises(Exception):
        objective(_ShiftModel(), [trajectory])


def test_empty_batch_raises() -> None:
    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    with pytest.raises(ValueError, match="empty batch"):
        objective(_ShiftModel(), [])


def test_reduction_is_the_one_the_trainer_dispatches_on() -> None:
    """The objective's `Reduction` must be the enum `backward` compares against, BY IDENTITY.

    `algorithm/primitives/objective.py::backward` dispatches with `is`, and posttrain carries a
    SECOND, unrelated `Reduction` -- `schema/metrics.py` aliases `MetricReduction` under that name
    for metric roll-up. When the objective module moved under `primitives/`, this file's old import
    path resolved to that other enum: the import succeeded, every type check passed, and the run
    died three steps in with `AssertionError: unhandled reduction Reduction.MEAN`. Identity is the
    only assertion that catches it.
    """
    from projects.cosmos3.posttrain.algorithm.primitives.objective import Reduction as TrainerReduction

    objective = make_alpagym_objective(_build_model_inputs, **_CLIPS, **_OFF)
    assert objective.reduction is TrainerReduction.MEAN
