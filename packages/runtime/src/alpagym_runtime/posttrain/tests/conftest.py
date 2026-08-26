# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the posttrain-facing AlpaGym bodies.

The rollout body's 22-step closed loop runs against a live AlpaSim runtime over gRPC, which no
unit test can stand up. These fixtures replace the streaming worker -- the single seam between the
body and the simulator -- with a stub that resolves pre-baked episodes, leaving the
framework-facing surface (`generate` -> list[Trajectory]) under test.
"""

from concurrent.futures import Future

import pytest
import torch
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.types import EpisodeOutput, PolicyOutput, RewardResult


def _episode(scene_id: str, steps: int = 3, reward: float = -9.35) -> EpisodeOutput:
    """A completed episode shaped like what StreamingRolloutWorker resolves."""
    outputs = tuple(
        PolicyOutput(
            chosen_xyz=torch.zeros(3),
            chosen_quat=torch.zeros(4),
            chosen_dt_us=torch.zeros(()),
            replay_data=PolicyReplayData(
                replay_schema_version=1,
                payload_schema="alpamayo_r1.trajectory.v1",
                payload_schema_version=1,
                model_family="alpamayo_r1",
                action_selection=ActionSelection(set_ix=0, sample_ix=i),
                old_logprob=torch.tensor(float(-i)),
                payload={"model_input": {"image_frames": torch.zeros(2, 2)}},
            ),
        )
        for i in range(steps)
    )
    return EpisodeOutput(
        scene_id=scene_id,
        session_uuid=f"sess-{scene_id}",
        num_steps=steps,
        policy_outputs=outputs,
        reward=RewardResult(total=reward),
    )


class _StubPayloadState:
    """Mirrors SharedPayloadState's read surface: an already-resolved future and n_target."""

    def __init__(self, episodes: list[EpisodeOutput], n_target: int) -> None:
        self.n_target = n_target
        self.future: Future = Future()
        self.future.set_result(episodes)


class _StubWorker:
    """Stands in for StreamingRolloutWorker: records submissions, resolves `episodes_per_prompt`."""

    def __init__(self, episodes_per_prompt: int, n_target: int) -> None:
        self._episodes_per_prompt = episodes_per_prompt
        self._n_target = n_target
        self.submitted: list = []

    def submit_payload(self, payload) -> _StubPayloadState:
        self.submitted.append(payload)
        episodes = [_episode(payload.scene_id) for _ in range(self._episodes_per_prompt)]
        return _StubPayloadState(episodes, self._n_target)

    def shutdown(self) -> None:
        pass


def _rollout_with(worker) -> object:
    """Build an AlpagymRollout around `worker`, bypassing the AlpaSim-dependent constructor."""
    from alpagym_runtime.posttrain.rollout import AlpagymRollout

    rollout = object.__new__(AlpagymRollout)
    rollout._worker = worker
    rollout._group_size = worker._n_target
    rollout._payload_seq = 0
    rollout._driver_server = None
    rollout._inference_engine = None
    rollout._engine_thread = None
    rollout._shutdown_done = False
    return rollout


@pytest.fixture
def rollout_full():
    """A rollout whose every prompt yields the full group of episodes."""
    return _rollout_with(_StubWorker(episodes_per_prompt=3, n_target=3))


@pytest.fixture
def rollout_short():
    """A rollout whose prompts come back one episode short -- retries exhausted, sim-side."""
    return _rollout_with(_StubWorker(episodes_per_prompt=1, n_target=3))
