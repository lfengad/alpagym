# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""posttrain rollout body driving AlpaSim simulator sessions.

The episode loop is inverted: this body calls ``simulate()`` on the AlpaSim runtime over gRPC, and
AlpaSim then calls BACK into ``EgodriverServer.drive()`` once per tick -- 22 times -- before
``simulate()`` returns. That whole micro-loop stays inside the body and is invisible to the
framework, which sees only ``generate(prompts, group) -> list[Trajectory]``. This is the shape
posttrain's DESIGN §2.7 sketches for VLA ("the obs->act->step micro-loop lives inside").

Ported from ``alpagym_runtime.cosmos.rollout_backend``. The simulator-facing half -- runtime stub,
driver server, streaming worker, and the shutdown ordering -- is carried over unchanged; only the
framework-facing surface differs (``generate``/``prepare_recv``/``apply_bucket`` instead of
``rollout_generation``/``model_param_map``/``set_underlying_model``).

Nothing on this module's import graph reaches ``cosmos_rl`` -- verified by importing it and finding
no ``cosmos_rl`` entry in ``sys.modules``, which is the claim that matters and is not the same as
the module's own import list being clean. It was not: ``streaming_worker`` imported ``RLPayload``
for a type annotation, and that one line loaded 185 cosmos-rl modules into the rollout actor.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import torch
import yaml
from alpagym_host.config import ExecutionBackend, RunConfig, load_run_config
from alpagym_host.endpoint_registry import (
    FileTopologyRegistry,
    TopologyEndpoint,
    rollout_worker_capacity,
)
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceStub
from projects.cosmos3.posttrain.roles.core.registry import register_rollout
from projects.cosmos3.posttrain.roles.rollout.base import (
    ShardSpec,
    WeightReceiverBase,
    group_ids_for_prompt,
)
from projects.cosmos3.posttrain.schema import ObsBundle, Trajectory

from alpagym_runtime.alpasim.driver_server import EgodriverServer
from alpagym_runtime.episode_runner.streaming_worker import StreamingRolloutWorker
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.perf.instrument.lifecycle import initialize_perf
from alpagym_runtime.perf.instrument.marker import record_perf_marker
from alpagym_runtime.perf.instrument.scope import measure_perf
from alpagym_runtime.policies.factory import build_inference_engine, build_policy_factory
from alpagym_runtime.posttrain.convert import episode_to_trajectory

logger = logging.getLogger(__name__)

_MAX_GRPC_MSG_SIZE = 256 * 1024 * 1024  # 256 MiB; matches AlpaSim runtime defaults.


@dataclass(frozen=True)
class _ScenePayload:
    """One prompt's work item, satisfying `StreamingRolloutWorker`'s `ScenePayload` protocol.

    `scene_id` is read by the resolver this module hands the worker; `prompt_idx` is the worker's
    dedup key. Nothing else on cosmos-rl's `RLPayload` was ever used.
    """

    prompt_idx: int
    scene_id: str


class AlpagymRollout(WeightReceiverBase):
    """posttrain rollout body that executes simulator sessions through AlpaSim."""

    def __init__(self, run_config: RunConfig) -> None:
        """Build the inference engine, driver server, AlpaSim stub, and streaming worker.

        cosmos-rl split this across `post_init_hook` and `init_engine` because it constructed
        bodies in two phases; posttrain constructs them once, so the split is gone.
        """
        self._run_config = run_config
        initialize_perf(run_config)
        self._topology_registry = FileTopologyRegistry(run_config.artifact_paths.topology_registry_dir)
        self._weight_backend: Any = None
        self._weight_version = 0
        # Verify the first sync only (see `apply_bucket`); cleared once that stream lands.
        self._verify_sync = True
        self._payload_seq = 0
        self._shutdown_done = False
        # Built below; None sentinels so `shutdown` no-ops cleanly on a partial init failure.
        self._inference_engine: InferenceEngine | None = None
        self._driver_server: EgodriverServer | None = None
        self._alpasim_runtime_stub: RuntimeServiceStub | None = None
        self._worker: StreamingRolloutWorker | None = None
        self._engine_thread: threading.Thread | None = None

        self._group_size = int(run_config.cosmos.rollout.n_generation)

        self._inference_engine = build_inference_engine(run_config)
        # PUBLIC `model`, not `_model`: `_Host.set_weight_backend` injects the weight-sync backend
        # into every body that `hasattr(body, "model")`. Under a private name the injection silently
        # skips this rollout, every pushed bucket is dropped, and the failure only surfaces at
        # `stamp_weight_version`. The cosmos-rl port used `_model`; posttrain's convention does not.
        self.model = self._inference_engine.get_model()
        record_perf_marker("rollout/model_ready", cpu_snapshot=True, gpu_snapshot=True)
        policy_factory = build_policy_factory(run_config, self._inference_engine)
        distributed = ExecutionBackend(run_config.execution.backend).is_slurm_run

        driver_id = f"driver-{socket.gethostname()}-pid-{os.getpid()}"
        alpasim_runtime_endpoint: TopologyEndpoint = self._topology_registry.acquire_alpasim_runtime(
            driver_id=driver_id
        )
        max_concurrent_rollouts = rollout_worker_capacity(
            runtime_capacity=int(alpasim_runtime_endpoint.capacity),
            rollout_replicas=int(run_config.cosmos.launch.rollout_replicas),
            alpasim_runtime_count=len(self._topology_registry.list_alpasim_runtimes()),
        )
        self._driver_server = EgodriverServer(
            name=driver_id,
            max_concurrent_rollouts=max_concurrent_rollouts,
            policy_factory=policy_factory,
            publish_host=socket.gethostname() if distributed else "localhost",
        )
        self._driver_server.start()
        self._topology_registry.publish_driver(self._driver_server.topology_endpoint)

        channel = grpc.insecure_channel(
            alpasim_runtime_endpoint.to_grpc_target(),
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MSG_SIZE),
                ("grpc.max_receive_message_length", _MAX_GRPC_MSG_SIZE),
            ],
        )
        grpc.channel_ready_future(channel).result(timeout=5.0)
        self._alpasim_runtime_stub = RuntimeServiceStub(channel)

        self._worker = StreamingRolloutWorker(
            alpasim_runtime_stub=self._alpasim_runtime_stub,
            driver_server=self._driver_server,
            simulation_timeout_s=float(run_config.alpasim.simulation_timeout_s),
            reward_config=run_config.reward,
            max_concurrent_rollouts=max_concurrent_rollouts,
            rollouts_per_payload=self._group_size,
            # The env carries the scene id on the prompt, so the payload already has it. This
            # retires cosmos-rl's `scene_ids[payload.prompt_idx]` lookup and the TODO beside it.
            scene_id_resolver=lambda payload: payload.scene_id,
        )

        # daemon=True so the engine thread does not block Python exit on a clean shutdown.
        self._engine_thread = threading.Thread(
            target=self._inference_engine.run_loop, name="alpagym-infer", daemon=True
        )
        self._engine_thread.start()
        atexit.register(self.shutdown)

        record_perf_marker("rollout/backend_ready", cpu_snapshot=True, gpu_snapshot=True)
        logger.info(
            "[alpagym] Streaming rollout backend ready: runtime=%s driver=%s max_concurrent_rollouts=%d",
            alpasim_runtime_endpoint,
            self._driver_server.topology_endpoint,
            max_concurrent_rollouts,
        )

    @measure_perf("rollout/generate", category="orchestration", cpu_snapshot=True, gpu_snapshot=True)
    def generate(self, prompts: list[ObsBundle], **knobs: Any) -> list[Trajectory]:
        """Run one AlpaSim session per prompt per group member; return flat Trajectories.

        Args:
            prompts: scene prompts from the env role. Each carries `representations["scene_id"]`
                and the uuid `obs.id` that becomes the GRPO group_id.
            **knobs: posttrain's per-call knobs. `group` must equal the group size this body was
                built with -- `rollouts_per_payload` is fixed when the streaming worker is
                constructed, so a differing value cannot be honoured.

        Returns:
            `len(prompts) * group` trajectories, prompt-major.

        Raises:
            ValueError: `group` disagrees with the constructed group size.
            RuntimeError: called before construction finished, or a payload exhausted its
                simulator retries and came back short.
        """
        group = int(knobs.get("group", self._group_size))
        if group != self._group_size:
            raise ValueError(
                f"group={group} but this rollout was built for group={self._group_size}; "
                "rollouts_per_payload is fixed when the streaming worker is constructed, so "
                "honouring the constructed value would silently train on the wrong group size"
            )
        if self._worker is None:
            raise RuntimeError("generate called before the engine was initialized")

        # Submit all before awaiting so simulate jobs run in parallel; the in-order await only
        # keeps the return deterministic.
        states = []
        for obs in prompts:
            self._payload_seq += 1
            states.append(
                self._worker.submit_payload(
                    _ScenePayload(
                        prompt_idx=self._payload_seq,
                        scene_id=str(obs.representations["scene_id"]),
                    )
                )
            )

        trajectories: list[Trajectory] = []
        for obs, state in zip(prompts, states):
            episodes = state.future.result()
            # The streaming worker resolves permanently-failed payloads with fewer episodes than
            # n_target instead of raising. Surface it here, or a short group reaches the advantage
            # computation and corrupts the group statistics.
            if len(episodes) < state.n_target:
                raise RuntimeError(
                    "Rollout payload exhausted retries: "
                    f"scene_id={obs.representations['scene_id']} "
                    f"collected={len(episodes)}/{state.n_target}"
                )
            group_id, traj_ids = group_ids_for_prompt(obs, group)
            trajectories.extend(
                episode_to_trajectory(episode, obs, traj_id)
                for episode, traj_id in zip(episodes, traj_ids)
            )
        return trajectories

    def apply_weights(self, source: Any | None = None) -> None:
        """Consumer-pull weight import. The training loop uses producer-driven `weight_sync`
        instead; this is the kept surface for manual / NCCL rigs."""
        if self._weight_backend is None:
            return
        from projects.cosmos3.posttrain.comm.weight_transfer import pull_stream_from_peer

        pull_stream_from_peer(self, source)

    def apply_bucket(self, payload: Any, i: int) -> None:
        """Load one pushed weight bucket in place. No version semantics: the producer stamps the
        version once, after the whole stream lands, so a stream that fails partway leaves the old
        version rather than advertising weights it never loaded."""
        if self._weight_backend is None:
            return
        if not self._verify_sync:
            self._weight_backend.apply_bucket(self.model, i, payload)
            return
        # Startup self-check, first sync only. Before any training the trainer holds the same
        # checkpoint this model loaded, so the transfer has to be numerically identity. Split the
        # fetch from the load (posttrain exposes both halves) and compare on the way through: a
        # transfer that corrupts a few tensors is otherwise invisible until the ratio drifts several
        # steps later, which reads as a training problem rather than a sync one. The bar sits above
        # bf16's 2**-13 round-trip.
        state = self._weight_backend.fetch_bucket_state(payload, i)
        if state is not None:
            current = dict(self.model.named_parameters())
            for name, incoming in state.items():
                have = current.get(name)
                if have is None:
                    continue
                diff = (incoming.detach().float() - have.detach().float()).abs().max().item()
                if diff > 1e-3:
                    raise RuntimeError(
                        f"weight sync is not identity at startup: {name} differs by {diff:.3e} "
                        f"before any training, when both sides still hold the same checkpoint"
                    )
        self._weight_backend.apply_bucket_state(self.model, i, state)

    def mark_weight_version_ready(self, version: int) -> None:
        """Record the version, and retire the startup self-check once the first stream lands.

        Hooked to `mark_weight_version_ready`, not `stamp_weight_version`: posttrain publishes a
        version in two phases now (mark ready, then commit through a rollout-wide barrier), and
        the old single stamp is no longer called. Left on the dead method, the self-check never
        retires and re-runs after every later sync -- where the weights legitimately differ,
        because a step trained them. It read as a transfer fault: `differs by 2.5e-03 before any
        training` on a stream that was in fact correct.
        """
        super().mark_weight_version_ready(version)
        self._verify_sync = False

    def prepare_recv(self) -> dict[str, ShardSpec]:
        """Report this rank's receive layout per parameter. The inference engine holds the whole
        model (no tensor parallelism), so every tensor is reported whole."""
        return {
            name: ShardSpec(offset=0, length=param.shape[0] if param.dim() else 1, dim=0)
            for name, param in self.model.named_parameters()
        }

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Every non-weight generation semantic, as stable finite JSON (`roles/rollout/base.py`).

        A resumed run must produce the same rollouts from the same weights, and this is what a
        validation check compares to prove it. What decides an AlpaGym rollout is NOT an HF
        generation config -- the policy runs a 22-tick closed loop against a simulator -- so the
        fields are the ones that would change the trajectory: which policy, which simulator scene
        set, and the closed-loop timing the wizard was brought up with. Read from the resolved run
        config rather than from live objects, so the identity is the same before and after the
        engine thread starts.
        """
        from projects.cosmos3.posttrain.utils.identity import normalize_typed_json_dict

        sim = self._run_config.alpasim.wizard_args
        # Through `normalize_typed_json_dict`, the same projection `config/build.py` puts the vLLM
        # rollout's identity through: it is what ENFORCES the protocol's "stable finite JSON" --
        # canonical key order, no NaN/Inf, no Python bool/number coercion. Hand-built `int()`/`str()`
        # calls merely happen to satisfy it today.
        return normalize_typed_json_dict(
            {
                "policy_kind": self._run_config.policy.model.kind,
                "policy_path": self._run_config.policy.model.path,
                "n_sim_steps": sim.n_sim_steps,
                "control_timestep_us": sim.control_timestep_us,
                "force_gt_duration_us": sim.force_gt_duration_us,
                "expected_valid_steps": self._run_config.expected_valid_steps,
            },
            context="AlpaGym rollout validation semantics",
        )

    def weight_version(self) -> int:
        """The training iteration whose weights this rollout currently holds.

        A method, not a property, to match `roles/rollout/vllm.py` and because the loop reads it
        through a worker handle -- which invokes it remotely as a method, so a property would
        resolve to an int and then fail to be called.
        """
        return self._weight_version

    def shutdown(self) -> None:
        """Stop accepting payloads and tear down worker, driver, and engine.

        Ordering matters and is carried over verbatim: worker first so simulate-pool threads stop
        issuing new `simulate()` calls; driver next so AlpaSim's pending `drive()` callbacks fail
        and let the runtime release in-flight sessions; engine sentinel last so `drive()`-issued
        inference futures finish draining before the engine thread exits.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if self._worker is not None:
            self._worker.shutdown()
        if self._driver_server is not None:
            self._driver_server.stop()
        if self._inference_engine is not None:
            self._inference_engine.shutdown()
        if self._engine_thread is not None:
            self._engine_thread.join(timeout=30.0)


@register_rollout("alpagym_rollout")
def build_alpagym_rollout(config: Any) -> AlpagymRollout:
    """Construct the AlpaGym rollout body from the entry's config.

    `config` is the entry's `AlpagymSection`, whose `resolved_config_path` points at the run config
    AlpaGym's host CLI already resolved -- the same file `cosmos/entrypoint.py` reads today. It is
    that section rather than the whole `EntryConfig` because the loop's own fields (gpus, steps,
    batch shape) mean nothing to a role body. Milestone 1 deliberately does not route this through
    i4's config root; see the design doc's §8.2.
    """
    return AlpagymRollout(load_run_config(Path(config.resolved_config_path)))
