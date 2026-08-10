# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AlpaGym rollout body's posttrain-facing contract.

The 22-step closed loop -- ``simulate()`` out to AlpaSim, ``drive()`` callbacks back in -- stays
INSIDE the body and is invisible to the framework, so these tests fake the streaming worker (the
one seam to the simulator) and assert only the surface posttrain sees: P prompts x G completions
come back as a flat list of Trajectories, grouped and id'd per
``roles/rollout/base.py::group_ids_for_prompt``.
"""

import pytest
from alpagym_runtime.posttrain.rollout import AlpagymRollout
from projects.cosmos3.posttrain.roles.rollout.base import Rollout
from projects.cosmos3.posttrain.schema import ObsBundle


def _prompts(*scene_ids: str) -> list[ObsBundle]:
    """Scene prompts as the env role emits them: a scene id plus a per-prompt uuid."""
    return [
        ObsBundle(representations={"scene_id": scene_id}, id=f"uuid-{scene_id}")
        for scene_id in scene_ids
    ]


def test_satisfies_the_rollout_protocol() -> None:
    """Structural conformance is what bind() and weight_sync rely on; check it explicitly."""
    assert isinstance(object.__new__(AlpagymRollout), Rollout)


def test_generate_returns_p_times_g_trajectories(rollout_full) -> None:
    trajectories = rollout_full.generate(_prompts("a", "b"), group=3)
    assert len(trajectories) == 6


def test_group_ids_are_per_prompt_and_traj_ids_unique(rollout_full) -> None:
    """Distinct prompts must never share a group, or the advantage estimator pools them."""
    trajectories = rollout_full.generate(_prompts("a", "b"), group=3)
    assert {t.group_id for t in trajectories} == {"uuid-a", "uuid-b"}
    assert len({t.id for t in trajectories}) == 6


def test_scene_id_comes_from_the_prompt_not_a_dataset_index(rollout_full) -> None:
    """The env carries the scene id explicitly, which retires cosmos-rl's prompt_idx lookup."""
    rollout_full.generate(_prompts("scene-x", "scene-y"), group=3)
    assert [p.scene_id for p in rollout_full._worker.submitted] == ["scene-x", "scene-y"]


def test_every_prompt_gets_a_distinct_payload_id(rollout_full) -> None:
    """The streaming worker dedups on payload id; a repeat would collapse two prompts into one."""
    rollout_full.generate(_prompts("a", "b"), group=3)
    ids = [p.prompt_idx for p in rollout_full._worker.submitted]
    assert len(set(ids)) == len(ids)


def test_group_mismatch_raises(rollout_full) -> None:
    """`rollouts_per_payload` is fixed when the worker is built; a differing `group` is a config
    bug, and silently honouring the constructed value would train on the wrong group size."""
    with pytest.raises(ValueError, match="group"):
        rollout_full.generate(_prompts("a"), group=5)


def test_short_episode_count_raises(rollout_short) -> None:
    """The worker resolves permanently-failed payloads with fewer episodes than n_target. Surface
    it here rather than letting a short group reach the advantage computation."""
    with pytest.raises(RuntimeError, match="exhausted retries"):
        rollout_short.generate(_prompts("a"), group=3)
