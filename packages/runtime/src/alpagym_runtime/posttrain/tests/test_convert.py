# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for converting a completed AlpaSim episode into a posttrain Trajectory.

One episode becomes one Trajectory of N Transitions (one per drive tick). The per-step replay
payload stays OPAQUE in ``algo_extra["payload"]``: posttrain's codec walks nested dicts and
externalizes the large tensors inside to BulkRefs, so opacity costs nothing on the bulk plane,
while splitting the payload across obs/action would force changes to alpamayo_r1's
``from_payload`` contract.

``group_id`` comes from ``obs.id`` (the uuid the env stamps) so completions of distinct scenes
never share a GRPO group.
"""

import pytest
import torch
from alpagym_runtime.posttrain.convert import episode_to_trajectory
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.types import EpisodeOutput, PolicyOutput, RewardResult
from projects.cosmos3.posttrain.schema import FINAL_REWARD_KEY, ObsBundle, Status


def _policy_output(step: int, with_replay: bool = True) -> PolicyOutput:
    """One drive tick, carrying a minimal but structurally complete replay envelope."""
    replay_data = None
    if with_replay:
        replay_data = PolicyReplayData(
            replay_schema_version=1,
            payload_schema="alpamayo_r1.trajectory.v1",
            payload_schema_version=1,
            model_family="alpamayo_r1",
            action_selection=ActionSelection(set_ix=0, sample_ix=step),
            old_logprob=torch.tensor(float(-step)),
            payload={
                "model_input": {"image_frames": torch.zeros(2, 2)},
                "timesteps": torch.zeros(3),
            },
        )
    return PolicyOutput(
        chosen_xyz=torch.zeros(3),
        chosen_quat=torch.zeros(4),
        chosen_dt_us=torch.zeros(()),
        replay_data=replay_data,
    )


def _episode(
    num_steps: int = 3,
    reward: float | None = -9.35,
    with_replay: bool = True,
) -> EpisodeOutput:
    """A completed episode with `num_steps` drive ticks."""
    return EpisodeOutput(
        scene_id="scene-a",
        session_uuid="sess-1",
        num_steps=num_steps,
        policy_outputs=tuple(_policy_output(i, with_replay) for i in range(num_steps)),
        reward=None if reward is None else RewardResult(total=reward),
    )


def _obs(obs_id: str = "prompt-uuid") -> ObsBundle:
    """The scene prompt an episode was rolled out from."""
    return ObsBundle(representations={"scene_id": "scene-a"}, id=obs_id)


def test_one_transition_per_drive_tick() -> None:
    """A 22-step episode -- the configured episode length -- yields 22 ordered transitions."""
    trajectory = episode_to_trajectory(_episode(num_steps=22), _obs(), traj_id="prompt-uuid_k0")
    assert len(trajectory.transitions) == 22
    assert [t.step_idx for t in trajectory.transitions] == list(range(22))


def test_group_id_comes_from_obs_id_not_scene_id() -> None:
    """Two episodes of the SAME scene must land in the same group only when they share a prompt.

    `scene_id` is shared by every rollout of a scene across steps; `obs.id` is a per-prompt uuid.
    Keying the group on scene_id would merge completions the advantage estimator must keep apart.
    """
    trajectory = episode_to_trajectory(_episode(), _obs("prompt-uuid"), traj_id="t")
    assert trajectory.group_id == "prompt-uuid"
    assert trajectory.group_id != trajectory.transitions[0].obs.representations["scene_id"]


def test_identity_and_status_are_carried() -> None:
    trajectory = episode_to_trajectory(_episode(), _obs(), traj_id="prompt-uuid_k0")
    assert trajectory.id == "prompt-uuid_k0"
    assert trajectory.session == "sess-1"
    assert trajectory.status is Status.COMPLETED


def test_terminal_reward_only_on_the_last_transition() -> None:
    """`_terminal_reward` reads transitions[-1]; a reward on every step would double-count."""
    trajectory = episode_to_trajectory(_episode(num_steps=3, reward=-9.35), _obs(), traj_id="t")
    assert [t.reward for t in trajectory.transitions[:-1]] == [None, None]
    assert trajectory.transitions[-1].reward == {FINAL_REWARD_KEY: -9.35}
    assert trajectory.final_reward == -9.35


def test_logprob_action_and_opaque_payload_land_in_their_fields() -> None:
    trajectory = episode_to_trajectory(_episode(num_steps=2), _obs(), traj_id="t")
    second = trajectory.transitions[1]
    assert float(second.policy_info["logprob"]) == -1.0
    assert second.action == ActionSelection(set_ix=0, sample_ix=1)
    assert "image_frames" in second.algo_extra["payload"]["model_input"]


def test_done_is_set_only_on_the_last_transition() -> None:
    trajectory = episode_to_trajectory(_episode(num_steps=3), _obs(), traj_id="t")
    assert [t.done for t in trajectory.transitions] == [False, False, True]


def test_missing_replay_data_raises() -> None:
    """Without replay_data the trainer cannot rescore the recorded action -- fail at the boundary."""
    with pytest.raises(ValueError, match="replay_data"):
        episode_to_trajectory(_episode(num_steps=1, with_replay=False), _obs(), traj_id="t")


def test_missing_reward_raises() -> None:
    """A missing reward must not become 0.0: that is a legitimate score for a bad attempt."""
    with pytest.raises(ValueError, match="reward"):
        episode_to_trajectory(_episode(reward=None), _obs(), traj_id="t")


def test_empty_obs_id_raises() -> None:
    """An empty id would collapse every scene into one GRPO group and corrupt the advantage."""
    with pytest.raises(ValueError, match="obs.id"):
        episode_to_trajectory(_episode(), _obs(obs_id=""), traj_id="t")


def test_no_policy_outputs_raises() -> None:
    with pytest.raises(ValueError, match="policy_outputs"):
        episode_to_trajectory(_episode(num_steps=0), _obs(), traj_id="t")
