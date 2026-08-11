# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a completed AlpaSim episode into posttrain's unified Trajectory.

One ``EpisodeOutput`` becomes one ``Trajectory`` whose transitions are the episode's drive ticks.
The per-step replay envelope -- the ``PolicyReplayData`` ``alpamayo_r1`` needs to rescore the
action it took -- stays opaque under ``algo_extra["replay_data"]``. The WHOLE envelope is kept, not
just its ``payload`` dict: the model family's own
``AlpamayoR1InferenceModel.build_trainer_model_inputs`` reads ``model_family``, ``payload_schema``
and ``old_logprob`` off it as well, and handing it a bare payload would strip the very fields it
validates against. posttrain's codec walks dataclasses and nested dicts alike, so the large tensors
inside (camera frames above all) are still externalized to ``BulkRef``s and never travel on the
control plane.

NOTE: ``alpagym_runtime.types.Trajectory`` is the ego vehicle's PHYSICAL path and is unrelated to
posttrain's RL ``Trajectory`` imported here. Only the posttrain one is used in this module.
"""

from __future__ import annotations

from projects.cosmos3.posttrain.schema import (
    FINAL_REWARD_KEY,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
)

from alpagym_runtime.types import EpisodeOutput


def episode_to_trajectory(episode: EpisodeOutput, obs: ObsBundle, traj_id: str) -> Trajectory:
    """Build one posttrain ``Trajectory`` from one completed AlpaSim episode.

    Args:
        episode: the completed session. Every ``policy_outputs`` entry must carry ``replay_data``,
            and ``reward`` must be present -- the trainer rescores recorded actions and the GRPO
            advantage estimator needs a terminal reward, so neither is defaultable.
        obs: the scene prompt this episode was rolled out from. ``obs.id`` is the GRPO group_id
            namespace and must be non-empty (see ``roles/rollout/base.py::group_ids_for_prompt``).
        traj_id: globally unique id for this completion, conventionally ``f"{obs.id}_k{ki}"``.

    Returns:
        A ``Trajectory`` of one ``Transition`` per drive tick, terminal reward on the last one.

    Raises:
        ValueError: ``obs.id`` is empty, ``episode.reward`` is missing, the episode has no ticks,
            or a tick is missing ``replay_data``.
    """
    if not obs.id:
        raise ValueError(
            "obs.id is empty; it is the GRPO group_id namespace and an empty id collapses every "
            "scene into one group (see roles/rollout/base.py::group_ids_for_prompt)"
        )
    if episode.reward is None:
        raise ValueError(
            f"episode {episode.session_uuid} has no reward; GRPO needs a terminal reward"
        )
    if not episode.policy_outputs:
        raise ValueError(f"episode {episode.session_uuid} has no policy_outputs")

    last = len(episode.policy_outputs) - 1
    transitions: list[Transition] = []
    for step_idx, policy_output in enumerate(episode.policy_outputs):
        replay_data = policy_output.replay_data
        if replay_data is None:
            raise ValueError(
                f"episode {episode.session_uuid} step {step_idx} has no replay_data; the trainer "
                "rescores the recorded action and cannot reconstruct it"
            )
        transitions.append(
            Transition(
                obs=obs,
                action=replay_data.action_selection,
                policy_info={"logprob": replay_data.old_logprob},
                algo_extra={"replay_data": replay_data},
                step_idx=step_idx,
                done=step_idx == last,
            )
        )

    transitions[-1].reward = {FINAL_REWARD_KEY: float(episode.reward.total)}
    return Trajectory(
        transitions=transitions,
        id=traj_id,
        session=episode.session_uuid,
        status=Status.COMPLETED,
        group_id=obs.id,
    )
