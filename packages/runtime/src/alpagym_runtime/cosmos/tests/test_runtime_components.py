# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import importlib
import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint
from PIL import Image


def _deep_merge_policy_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """In-place deep merge `overrides` into `base`."""
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge_policy_overrides(existing, value)
        else:
            base[key] = value


def _write_resolved_config(
    tmp_path: Path,
    scene_ids: list[str],
    policy_overrides: dict[str, Any],
    max_concurrent_rollouts: int,
    simulation_timeout_s: float,
    experiment_name: str,
    execution_backend: str = "local_process",
    runtime_scene_ids: list[str] | None = None,
    runtime_capacity: int | None = None,
    rollout_replicas: int = 1,
    runtime_count: int = 1,
) -> Path:
    """Write a resolved alpagym config YAML and return its path."""
    base_policy: dict[str, Any] = {
        "kind": "alpamayo",
        "model": {
            "kind": "alpamayo_r1",
            "path": "unused-by-test",
            "device": "cpu",
            "dtype": "float32",
            "use_cameras": [],
            "input_size": [320, 512],
            "num_context_frames": 2,
            "num_historical_waypoints": 1,
            "num_future_waypoints": 2,
            "step_dt_us": 500_000,
        },
        "inference": {
            "max_batch_size": 4,
            "return_trace_for_rl": True,
            "sampling": {
                "top_p": 1.0,
                "top_k": None,
                "temperature": 1.0,
                "num_traj_samples": 1,
                "num_traj_sets": 1,
                "max_generation_length": None,
            },
        },
        "trajectory_selector": "identity",
    }
    _deep_merge_policy_overrides(base_policy, policy_overrides)
    resolved = {
        "command": "run",
        "run_root": str(tmp_path),
        "logging_level": "DEBUG",
        "cache_root_dir": str(tmp_path / "cache"),
        "execution": {
            "backend": execution_backend,
            "resolved_config_path": None,
            "slurm": {
                "job_name": "alpagym",
                "partition": None,
                "account": None,
                "time": "02:00:00",
                "nodes": 1,
                "gpus_per_node": 8,
                "topology": {
                    "kind": "all_in_one",
                    "alpasim_gpus": 4,
                },
                "exclusive": True,
                "cpus_per_task": None,
                "container_image": None,
                "container_cache_root": None,
                "container_workdir": "/alpamayo/projects/alpagym",
                "uv_cache_dir": str(tmp_path / "uv-cache"),
                "container_mounts": [],
                "export_env": [],
            },
        },
        "dataset": {"scene_ids": scene_ids},
        "policy": base_policy,
        "reward": {"terms": [{"kind": "distance_to_gt", "scale": -0.01}]},
        "alpasim": {
            "repo_url": "ssh://git@example/alpasim.git",
            "repo_ref": "abc123",
            "startup_timeout_s": 600.0,
            "simulation_timeout_s": simulation_timeout_s,
            "wizard_args": {
                "deploy": "local",
                "topology": "1gpu",
                "driver": None,
                "driver_source": "external_dynamic",
                "force_gt_duration_us": 3_000_000,
                "control_timestep_us": 100_000,
                "n_sim_steps": 38,
                "extra_overrides": "",
            },
        },
        "cosmos": {
            "mode": "colocated",
            "placement": "colocated",
            "launch": {
                "policy_replicas": 1,
                "rollout_replicas": rollout_replicas,
                "controller_port": 29500,
            },
            "train": {
                "max_num_steps": 1,
                "num_epochs": 1,
                "train_batch_per_replica": 2,
                "optm_lr": 1.0e-6,
                "optm_warmup_steps": 20,
                "train_policy": {
                    "allowed_outdated_steps": 4,
                    "on_policy": False,
                    "mini_batch": 1,
                    "grpo_ratio_clip_low": 0.2,
                    "grpo_ratio_clip_high": 0.2,
                    "grpo_optimization_iterations": 1,
                    "kl_beta": 0.0,
                    "reference_reset_interval": 0,
                },
            },
            "policy": {
                "parallelism": {
                    "tp_size": 1,
                    "cp_size": 1,
                    "ep_size": 1,
                    "dp_shard_size": 1,
                    "pp_size": 1,
                    "pp_micro_batch_size": 1,
                    "dp_replicate_size": 1,
                },
            },
            "rollout": {
                "backend": "alpagym_rollout",
                "n_generation": 2,
                "batch_size": 1,
                "parallelism": {
                    "tp_size": 1,
                    "pp_size": 1,
                },
            },
            "logging": {
                "logger": ["console"],
                "log_training_metrics_every_n_steps": 1,
                "project_name": "alpagym",
                "experiment_name": experiment_name,
            },
        },
        "expected_valid_steps": 8,
        "artifact_paths": {
            "run_dir": str(tmp_path),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "policy_model_bundle_dir": str(tmp_path / "artifacts" / "policy_model_bundle"),
            "resolved_config_path": str(tmp_path / "resolved_config.yaml"),
            "cosmos_config_path": str(tmp_path / "cosmos_config.toml"),
            "submit_script_path": str(tmp_path / "submit.sbatch"),
            "log_dir": str(tmp_path / "logs"),
            "topology_registry_dir": str(tmp_path / "topology"),
            "alpasim_log_dir": str(tmp_path / "alpasim"),
            "alpasim_scene_ids_path": str(tmp_path / "alpasim_scene_ids.yaml"),
            "perf_dir": str(tmp_path / "perf"),
        },
        "perf": {
            "enabled": False,
            "sample_every_n": 1,
            "resource_sample_interval_s": 5.0,
            "max_samples_per_series": 1000,
            "flush_every_n_updates": 1000,
            "flush_interval_s": 60.0,
            "collect_cpu": True,
            "collect_gpu": True,
        },
    }
    resolved_config_path = tmp_path / "resolved_config.yaml"
    resolved_config_path.write_text(yaml.safe_dump(resolved))
    (tmp_path / "alpasim_scene_ids.yaml").write_text(
        yaml.safe_dump(
            {"scene_ids": runtime_scene_ids or scene_ids},
            sort_keys=False,
        )
    )
    registry = FileTopologyRegistry(tmp_path / "topology")
    for runtime_index in range(runtime_count):
        registry.publish_alpasim_runtime(
            TopologyEndpoint(
                id=f"alpasim-runtime-{runtime_index}",
                host="runtime.local",
                port=6101 + runtime_index,
                capacity=runtime_capacity or max_concurrent_rollouts,
            )
        )
    return resolved_config_path


_ALPAMAYO_TEST_CAMERA = "camera_front_wide_120fov"


def test_entrypoint_configures_logging_from_resolved_config_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmos_stubs: None,
) -> None:
    """Cosmos workers read AlpaGym log level from resolved YAML, not generated TOML."""
    del cosmos_stubs
    entrypoint_module = importlib.import_module("alpagym_runtime.cosmos.entrypoint")
    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["scene_000"],
        policy_overrides={},
        max_concurrent_rollouts=1,
        simulation_timeout_s=30.0,
        experiment_name="logging_from_yaml",
    )
    cosmos_config_path = tmp_path / "cosmos_config.toml"
    cosmos_config_path.write_text(
        f"[custom]\nresolved_config_path = {json.dumps(str(resolved_config_path))}\n",
        encoding="utf-8",
    )
    basic_config_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        entrypoint_module,
        "get_policy_bundle",
        lambda model_kind: SimpleNamespace(
            build_data_packer=lambda config, cosmos_role: SimpleNamespace(close=lambda: None)
        ),
    )
    monkeypatch.setattr(
        entrypoint_module.logging,
        "basicConfig",
        lambda **kwargs: basic_config_calls.append(kwargs),
    )
    monkeypatch.setattr(entrypoint_module, "launch_worker", lambda **kwargs: None)

    entrypoint_module.main(["--config", str(cosmos_config_path)])

    assert basic_config_calls[0]["level"] == logging.DEBUG


def test_entrypoint_dataset_uses_discovered_runtime_scenes(
    tmp_path: Path,
    cosmos_stubs: None,
) -> None:
    """Cosmos dataset uses AlpaSim-discovered scenes, not the authored selector list."""
    del cosmos_stubs
    entrypoint_module = importlib.import_module("alpagym_runtime.cosmos.entrypoint")
    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["authored_scene"],
        policy_overrides={},
        max_concurrent_rollouts=1,
        simulation_timeout_s=30.0,
        experiment_name="runtime_scenes",
        runtime_scene_ids=["runtime_scene_a", "runtime_scene_b"],
    )

    dataset = entrypoint_module._build_dataset(
        SimpleNamespace(custom={"resolved_config_path": str(resolved_config_path)})
    )

    assert len(dataset) == 2
    assert dataset[0] == "runtime_scene_a"
    assert dataset[1] == "runtime_scene_b"


def test_rollout_generation_declares_current_weight_version(
    cosmos_stubs: None,
) -> None:
    """`rollout_generation` declares cosmos-rl's `current_weight_version` kwarg.

    Cosmos-RL `_call_rollout_generation` forwards `current_weight_version`
    via kwargs. The signature declares it explicitly (Google-style) instead
    of absorbing it via `**kwargs`, so a future cosmos kwarg surfaces as a
    loud `TypeError` rather than being silently consumed.
    """
    del cosmos_stubs
    import inspect

    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    sig = inspect.signature(rollout_module.AlpagymRollout.rollout_generation)
    assert "current_weight_version" in sig.parameters, (
        "rollout_generation must declare `current_weight_version` so a "
        "new cosmos kwarg surfaces as TypeError instead of being absorbed."
    )


def test_rollout_init_acquires_runtime_for_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmos_stubs: None,
) -> None:
    """Distributed rollout workers acquire the runtime assigned to their driver."""
    del cosmos_stubs
    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    from cosmos_rl.rollout.rollout_base import RolloutRegistry

    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["scene_000"],
        policy_overrides={"kind": "fake"},
        max_concurrent_rollouts=1,
        simulation_timeout_s=30.0,
        experiment_name="cosmos_runtime_binding",
        execution_backend="slurm",
        rollout_replicas=4,
        runtime_count=2,
    )

    class FakeDriverServer:
        """Driver server stand-in that records publish host wiring."""

        instances: list["FakeDriverServer"] = []

        def __init__(
            self,
            name: str,
            max_concurrent_rollouts: int,
            policy_factory: Any,
            publish_host: str = "localhost",
        ) -> None:
            """Capture constructor arguments used by rollout init."""
            del policy_factory
            self.name = name
            self.max_concurrent_rollouts = max_concurrent_rollouts
            self.publish_host = publish_host
            self.topology_endpoint = TopologyEndpoint(
                id=name,
                host=publish_host,
                port=43292,
                capacity=max_concurrent_rollouts,
            )
            self.started = False
            self.instances.append(self)

        def start(self) -> None:
            """Record driver startup."""
            self.started = True

        def stop(self) -> None:
            """Accept driver shutdown."""

    class FakeRegistry:
        """Topology registry stand-in with two runtime endpoints."""

        acquired_driver_ids: list[str] = []
        published_drivers: list[TopologyEndpoint] = []

        def __init__(self, registry_dir: str) -> None:
            """Accept the configured registry directory."""
            self.registry_dir = registry_dir

        def acquire_alpasim_runtime(self, driver_id: str) -> TopologyEndpoint:
            """Return the endpoint assigned to the driver."""
            self.acquired_driver_ids.append(driver_id)
            endpoints = [
                TopologyEndpoint(
                    id="runtime-0",
                    host="runtime-0.local",
                    port=6101,
                    capacity=1,
                ),
                TopologyEndpoint(
                    id="runtime-1",
                    host="runtime-1.local",
                    port=6102,
                    capacity=5,
                ),
            ]
            return endpoints[1]

        def list_alpasim_runtimes(self) -> list[TopologyEndpoint]:
            """Return published runtime endpoints."""
            return [
                TopologyEndpoint(
                    id="runtime-0",
                    host="runtime-0.local",
                    port=6101,
                    capacity=1,
                ),
                TopologyEndpoint(
                    id="runtime-1",
                    host="runtime-1.local",
                    port=6102,
                    capacity=5,
                ),
            ]

        def publish_driver(self, endpoint: TopologyEndpoint) -> None:
            """Record the rollout worker's driver endpoint."""
            self.published_drivers.append(endpoint)

    class FakeStreamingWorker:
        """Streaming worker stand-in that captures the selected runtime stub."""

        instances: list["FakeStreamingWorker"] = []

        def __init__(
            self,
            *,
            alpasim_runtime_stub: Any,
            driver_server: FakeDriverServer,
            simulation_timeout_s: float,
            reward_config: Any,
            max_concurrent_rollouts: int,
            rollouts_per_payload: int,
            scene_id_resolver: Any,
        ) -> None:
            """Capture constructor arguments used by rollout init."""
            del simulation_timeout_s, reward_config
            del rollouts_per_payload, scene_id_resolver
            self.alpasim_runtime_stub = alpasim_runtime_stub
            self.driver_server = driver_server
            self.max_concurrent_rollouts = max_concurrent_rollouts
            self.instances.append(self)

        def shutdown(self) -> None:
            """Accept rollout shutdown."""

    class FakeInferenceEngine:
        """Inference engine stand-in exposing Jef's model-sync interface."""

        def __init__(self) -> None:
            """Create the fake model."""
            self.model = torch.nn.Linear(1, 1)

        def get_model(self) -> torch.nn.Module:
            """Return a model object for rollout weight sync."""
            return self.model

        def run_loop(self) -> None:
            """Return immediately for the test thread."""

        def shutdown(self) -> None:
            """Accept rollout shutdown."""

        def set_model(self, model: torch.nn.Module) -> None:
            """Update the fake model reference."""
            self.model = model

    class FakeChannel:
        """Stand-in gRPC channel."""

        def __init__(self, target: str, options: Any = None) -> None:
            """Capture the selected runtime target."""
            del options
            self.target = target

    def fake_channel_ready_future(channel: Any) -> Any:
        """Return an object whose `result(timeout=...)` is a no-op."""
        del channel
        return SimpleNamespace(result=lambda timeout: None)

    monkeypatch.setattr(
        rollout_module,
        "build_inference_engine",
        lambda config: FakeInferenceEngine(),
    )
    monkeypatch.setattr(rollout_module, "build_policy_factory", lambda config, engine: object())
    monkeypatch.setattr(rollout_module.socket, "gethostname", lambda: "worker-host")
    monkeypatch.setattr(rollout_module, "EgodriverServer", FakeDriverServer)
    monkeypatch.setattr(rollout_module, "FileTopologyRegistry", FakeRegistry)
    monkeypatch.setattr(rollout_module, "StreamingRolloutWorker", FakeStreamingWorker)
    monkeypatch.setattr(rollout_module, "RuntimeServiceStub", lambda channel: channel)
    monkeypatch.setattr(rollout_module.grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(rollout_module.grpc, "channel_ready_future", fake_channel_ready_future)

    config = SimpleNamespace(custom={"resolved_config_path": str(resolved_config_path)})
    rollout = RolloutRegistry.get_rollout_cls("alpagym_rollout")(config=config)
    try:
        rollout.init_engine(quantization="none", seed=0, load_format="auto")

        assert FakeRegistry.acquired_driver_ids == [FakeDriverServer.instances[0].name]
        assert FakeDriverServer.instances[0].name.startswith("driver-worker-host-pid-")
        assert FakeDriverServer.instances[0].publish_host == "worker-host"
        assert FakeDriverServer.instances[0].max_concurrent_rollouts == 3
        assert FakeStreamingWorker.instances[0].max_concurrent_rollouts == 3
        assert FakeRegistry.published_drivers == [FakeDriverServer.instances[0].topology_endpoint]
        assert (
            FakeStreamingWorker.instances[0].alpasim_runtime_stub.target == "runtime-1.local:6102"
        )
    finally:
        rollout.shutdown()


def _alpamayo_test_jpeg() -> bytes:
    """Encode a single-color RGB image as JPEG bytes for the Alpamayo rollout test."""
    image = Image.new("RGB", (8, 6), color=(123, 45, 67))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _alpamayo_test_calibration() -> Any:
    """Single-camera calibration matching the Alpamayo test policy config."""
    from alpagym_runtime.types import CameraCalibration, CameraIntrinsics, Pose

    return (
        CameraCalibration(
            name=_ALPAMAYO_TEST_CAMERA,
            logical_id=_ALPAMAYO_TEST_CAMERA,
            intrinsics=CameraIntrinsics(fx=400.0, fy=400.0, cx=320.0, cy=240.0),
            extrinsic_pose=Pose(),
        ),
    )


def _alpamayo_test_policy_input() -> Any:
    """Build a one-camera `PolicyInput` for the Alpamayo rollout test."""
    from alpagym_runtime.inference.types import NUM_ROUTE_WAYPOINTS
    from alpagym_runtime.types import (
        CameraImage,
        EgoPose,
        PolicyInput,
        Pose,
        RouteWaypoint,
        Trajectory,
    )

    nan = float("nan")
    live_waypoints = (RouteWaypoint(x=5.0, y=0.0), RouteWaypoint(x=10.0, y=1.0))
    route_waypoints = live_waypoints + tuple(
        RouteWaypoint(x=nan, y=nan, z=nan) for _ in range(NUM_ROUTE_WAYPOINTS - len(live_waypoints))
    )

    return PolicyInput(
        step_index=0,
        time_now_us=1_000_000,
        time_query_us=1_200_000,
        camera_images=(
            CameraImage(
                logical_id=_ALPAMAYO_TEST_CAMERA,
                image_bytes=_alpamayo_test_jpeg(),
                frame_end_us=1_000_000,
            ),
        ),
        ego_trajectory=Trajectory(
            poses=(
                EgoPose(timestamp_us=900_000, pose=Pose()),
                EgoPose(timestamp_us=1_000_000, pose=Pose()),
            )
        ),
        route_waypoints=route_waypoints,
        route_timestamp_us=1_000_000,
        calibration=_alpamayo_test_calibration(),
    )


class _FakeR1Model(torch.nn.Module):
    """Fake public R1 trajectory model exposing `sample_trajectories_from_data`."""

    def __init__(
        self,
        num_traj_sets: int,
        num_traj_samples: int,
        horizon: int,
    ) -> None:
        """Bind the fake to the per-call output shape it should emit."""
        super().__init__()
        self._num_traj_sets = num_traj_sets
        self._num_traj_samples = num_traj_samples
        self._horizon = horizon
        self.call_count = 0
        self.sample_call_kwargs: list[dict[str, Any]] = []
        self.config = SimpleNamespace(
            model_type="alpamayo_reasoning_vla_expert",
            legacy_inference_image_input_format=True,
        )

    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        return_extra: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return preset tensors with a leading batch axis matching `data`."""
        self.sample_call_kwargs.append(
            {
                "return_extra": return_extra,
                "diffusion_kwargs": kwargs.get("diffusion_kwargs"),
            }
        )
        del top_p, top_k, temperature, return_extra, kwargs
        del num_traj_samples, num_traj_sets
        self.call_count += 1
        batch_size = int(data["camera_indices"].shape[0])
        pred_xyz = torch.zeros(
            (batch_size, self._num_traj_sets, self._num_traj_samples, self._horizon, 3),
            dtype=torch.float32,
        )
        pred_rot = (
            torch.eye(3)
            .expand(
                batch_size,
                self._num_traj_sets,
                self._num_traj_samples,
                self._horizon,
                3,
                3,
            )
            .clone()
        )
        logprob = torch.zeros(
            (batch_size, self._num_traj_sets, self._num_traj_samples), dtype=torch.float32
        )
        num_candidates = self._num_traj_sets * self._num_traj_samples
        samples_list = torch.zeros(
            (batch_size * num_candidates, self._horizon + 1, self._horizon, 3),
            dtype=torch.float32,
        )
        timesteps = torch.linspace(0.0, 1.0, self._horizon + 1, dtype=torch.float32)
        return (
            pred_xyz,
            pred_rot,
            logprob,
            {
                "samples_list": samples_list,
                "timesteps": timesteps,
                "noise_level": 0.2,
            },
        )


def test_alpamayo_rollout_model_param_map_deduplicates_tied_parameters() -> None:
    """R2R sync should use the same unique parameter surface as P2R sync."""
    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    rollout = rollout_module.AlpagymRollout.__new__(rollout_module.AlpagymRollout)
    rollout._engine_initialized = True
    rollout._model_param_map = None

    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Linear(2, 2, bias=False)
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    model.lm_head.weight = model.embed_tokens.weight
    rollout._model = model

    class FakeWeightMapper:
        """Weight mapper stand-in for the Cosmos rollout parameter-map path."""

        @staticmethod
        def rollout_map_local_key_to_hf_key(name: str) -> str:
            """Leave fake parameter names unchanged."""
            return name

    assert set(model.state_dict()) == {"embed_tokens.weight", "lm_head.weight"}
    assert list(rollout.model_param_map(FakeWeightMapper())) == ["embed_tokens.weight"]


def test_alpamayo_rollout_with_monkeypatched_alpamayo_r1_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cosmos_stubs: None,
) -> None:
    """Drive one Alpamayo rollout through a monkeypatched Alpamayo R1 model."""
    del cosmos_stubs

    rollout_module = importlib.import_module("alpagym_runtime.cosmos.rollout_backend")
    importlib.import_module("alpagym_runtime.cosmos.trainer")
    packer_module = importlib.import_module("alpagym_runtime.cosmos.packer")

    import alpagym_alpamayo_r1.inference_model as r1_inference_model
    import alpamayo1_x_rl.models.expert_model.model
    from alpagym_alpamayo_r1.inference_model import AlpamayoR1InferenceModel
    from alpagym_runtime.replay import DataPackerConfig
    from cosmos_rl.rollout.rollout_base import RolloutRegistry

    bundle_dir = tmp_path / "alpamayo_r1_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text(
        json.dumps({"model_type": "alpamayo_reasoning_vla_expert"}), encoding="utf-8"
    )

    horizon = 4
    num_traj_sets = 1
    num_traj_samples = 2

    fake_r1_model = _FakeR1Model(
        num_traj_sets=num_traj_sets,
        num_traj_samples=num_traj_samples,
        horizon=horizon,
    )

    monkeypatch.setattr(
        alpamayo1_x_rl.models.expert_model.model.ExpertModelRL,
        "from_pretrained",
        lambda *args, **kwargs: fake_r1_model,
    )
    monkeypatch.setattr(
        r1_inference_model,
        "tokenize_for_generation",
        lambda *args, **kwargs: {"input_ids": torch.zeros((1, 3), dtype=torch.int64)},
    )

    resolved_config_path = _write_resolved_config(
        tmp_path,
        scene_ids=["scene_000", "scene_001"],
        policy_overrides={
            "kind": "alpamayo",
            "model": {
                "kind": "alpamayo_r1",
                "path": str(bundle_dir),
                "dtype": "bfloat16",
                "use_cameras": [_ALPAMAYO_TEST_CAMERA],
                "num_context_frames": 1,
                "num_historical_waypoints": 2,
                "num_future_waypoints": horizon,
            },
            "inference": {
                "sampling": {
                    "top_p": 0.98,
                    "top_k": None,
                    "temperature": 0.6,
                    "num_traj_samples": num_traj_samples,
                    "num_traj_sets": num_traj_sets,
                    "max_generation_length": None,
                },
            },
        },
        max_concurrent_rollouts=2,
        simulation_timeout_s=30.0,
        experiment_name="cosmos_alpamayo_r1_smoke",
    )

    class FakeAlpamayoDriverServer:
        """Driver server stand-in that exposes its policy factory to the runner."""

        def __init__(
            self,
            name: str,
            max_concurrent_rollouts: int,
            policy_factory: Any,
            publish_host: str = "localhost",
        ) -> None:
            """Capture construction kwargs and stash the captured policy factory."""
            self.name = name
            self.max_concurrent_rollouts = max_concurrent_rollouts
            self.policy_factory = policy_factory
            self.topology_endpoint = TopologyEndpoint(
                id=name,
                host=publish_host,
                port=43292,
                capacity=max_concurrent_rollouts,
            )

        def start(self) -> None:
            """Accept rollout-worker driver startup."""

        def stop(self) -> None:
            """Accept rollout-worker driver teardown."""

    class FakeRegistry:
        """Topology registry stand-in for the Alpamayo rollout test."""

        published_drivers: list[TopologyEndpoint] = []

        def __init__(self, registry_dir: str) -> None:
            """Accept the configured registry directory."""
            self.registry_dir = registry_dir

        def acquire_alpasim_runtime(self, driver_id: str) -> TopologyEndpoint:
            """Return one published AlpaSim RuntimeService endpoint."""
            del driver_id
            return TopologyEndpoint(
                id="alpasim-runtime-0",
                host="runtime.local",
                port=6101,
                capacity=1,
            )

        def list_alpasim_runtimes(self) -> list[TopologyEndpoint]:
            """Return published runtime endpoints."""
            return [
                TopologyEndpoint(
                    id="alpasim-runtime-0",
                    host="runtime.local",
                    port=6101,
                    capacity=1,
                )
            ]

        def publish_driver(self, endpoint: TopologyEndpoint) -> None:
            """Record the rollout worker's driver endpoint."""
            self.published_drivers.append(endpoint)

    class FakeStreamingWorker:
        """Streaming-worker stand-in that drives captured policies through the inference engine.

        Replaces the real `StreamingRolloutWorker` so the test can exercise
        `AlpagymRollout.init_engine` and `rollout_generation` without a real
        AlpaSim runtime or gRPC channel. Each `submit_payload` call
        synchronously walks the captured policy factory through the
        long-lived inference engine thread (already running by the time
        the test calls `rollout_generation`). Like the real worker it
        produces in-memory ``EpisodeOutput`` completions; egress to a
        transport happens later in the packer's ``get_rollout_output``.
        """

        def __init__(
            self,
            *,
            alpasim_runtime_stub: Any,
            driver_server: FakeAlpamayoDriverServer,
            simulation_timeout_s: float,
            reward_config: Any,
            max_concurrent_rollouts: int,
            rollouts_per_payload: int,
            scene_id_resolver: Any,
            max_scene_retries: int = 3,
        ) -> None:
            """Capture the wiring needed to drive policies for assertions."""
            del alpasim_runtime_stub, reward_config, max_concurrent_rollouts, max_scene_retries
            self._driver_server = driver_server
            self._simulation_timeout_s = simulation_timeout_s
            self._scene_id_resolver = scene_id_resolver
            self._rollouts_per_payload = rollouts_per_payload
            self.captured_outputs: dict[str, tuple[Any, ...]] = {}
            self.driver_step_calls = 0

        def submit_payload(self, payload: Any) -> Any:
            """Drive one payload synchronously and return a resolved-future state."""
            return self._drive_payload(payload)

        def shutdown(self) -> None:
            """No-op shutdown for the fake worker."""

        def _drive_payload(self, payload: Any) -> Any:
            """Walk one payload's `rollouts_per_payload` siblings through the policy factory."""
            from concurrent.futures import Future

            from alpagym_runtime.types import EpisodeOutput

            factory = self._driver_server.policy_factory
            scene_id = self._scene_id_resolver(payload)
            episodes: list[Any] = []
            for j in range(self._rollouts_per_payload):
                session_uuid = f"sess-{scene_id}-{j}"
                policy = factory(session_uuid, _alpamayo_test_calibration(), 0)
                output = policy.step(_alpamayo_test_policy_input())
                self.captured_outputs[session_uuid] = (output,)
                policy.close()
                self.driver_step_calls += 1
                episodes.append(
                    EpisodeOutput(
                        scene_id=scene_id,
                        session_uuid=session_uuid,
                        num_steps=1,
                        policy_outputs=(output,),
                    )
                )
            future: Future = Future()
            future.set_result(episodes)
            return SimpleNamespace(
                payload=payload,
                n_target=self._rollouts_per_payload,
                future=future,
            )

    class FakeChannel:
        """Stand-in gRPC channel; the streaming worker only needs the stub."""

        def __init__(self, target: str, options: Any = None) -> None:
            """Capture target for diagnostics."""
            del options
            self.target = target

    def fake_channel_ready_future(channel: Any) -> Any:
        """Return an object whose `result(timeout=...)` is a no-op."""
        del channel
        return SimpleNamespace(result=lambda timeout: None)

    monkeypatch.setattr(rollout_module, "StreamingRolloutWorker", FakeStreamingWorker)
    monkeypatch.setattr(rollout_module, "EgodriverServer", FakeAlpamayoDriverServer)
    monkeypatch.setattr(rollout_module, "FileTopologyRegistry", FakeRegistry)
    monkeypatch.setattr(rollout_module, "RuntimeServiceStub", lambda channel: SimpleNamespace())
    monkeypatch.setattr(rollout_module.grpc, "insecure_channel", FakeChannel)
    monkeypatch.setattr(rollout_module.grpc, "channel_ready_future", fake_channel_ready_future)
    FakeRegistry.published_drivers = []

    config = SimpleNamespace(
        custom={
            "resolved_config_path": str(resolved_config_path),
        }
    )
    rollout = RolloutRegistry.get_rollout_cls("alpagym_rollout")(config=config)
    rollout.init_engine(quantization="none", seed=0, load_format="auto")
    assert rollout.get_underlying_model() is fake_r1_model

    class FakeWeightMapper:
        """Weight mapper stand-in for the Cosmos rollout parameter-map path."""

        @staticmethod
        def rollout_map_local_key_to_hf_key(name: str) -> str:
            """Leave fake parameter names unchanged."""
            return name

    assert rollout.model_param_map(FakeWeightMapper()) == {}
    synced_r1_model = _FakeR1Model(
        num_traj_sets=num_traj_sets,
        num_traj_samples=num_traj_samples,
        horizon=horizon,
    )
    rollout._model_param_map = {"stale": torch.tensor(1.0)}
    rollout.set_underlying_model(synced_r1_model)
    assert rollout.get_underlying_model() is synced_r1_model
    assert rollout._model_param_map is None
    assert rollout._inference_engine.get_model() is synced_r1_model

    try:
        results = rollout.rollout_generation(
            payloads=[
                SimpleNamespace(prompt_idx=0),
                SimpleNamespace(prompt_idx=1),
            ],
            stream=None,
            data_packer=packer_module.AlpagymDataPacker(
                config=DataPackerConfig(expected_valid_steps=1),
                build_model_inputs=functools.partial(
                    AlpamayoR1InferenceModel.build_trainer_model_inputs,
                    num_context_frames=1,
                ),
            ),
        )

        worker = cast(FakeStreamingWorker, rollout._worker)
        expected_sessions = 2 * 2  # 2 scenes * n_generation=2

        # End-to-end: inference_engine.run_loop() drained one drive per session and the
        # fake R1 model was invoked once per drive.
        assert worker.driver_step_calls == expected_sessions
        assert fake_r1_model.call_count == 0
        assert synced_r1_model.call_count == expected_sessions
        assert len(synced_r1_model.sample_call_kwargs) == expected_sessions
        for call_kwargs in synced_r1_model.sample_call_kwargs:
            assert call_kwargs["return_extra"] is True
            diffusion_kwargs = call_kwargs["diffusion_kwargs"]
            assert isinstance(diffusion_kwargs, dict)
            assert diffusion_kwargs["return_info"] is True
        assert FakeRegistry.published_drivers[0].id.startswith("driver-")

        # The public R1 smoke path requests replay traces for trainer-side packing.
        assert len(worker.captured_outputs) == expected_sessions
        for session_outputs in worker.captured_outputs.values():
            assert len(session_outputs) == 1
            for output in session_outputs:
                assert output.replay_data is not None
                assert output.replay_data.model_family == "alpamayo_r1"
                assert output.replay_data.payload_schema == "alpamayo_r1.trajectory.v1"
                assert output.chosen_logprob is not None
                assert output.chosen_logprob.shape == (1,)
                assert output.all_pred_xyz is not None
                assert output.all_pred_xyz.shape == (
                    num_traj_sets,
                    num_traj_samples,
                    horizon,
                    3,
                )
                assert output.all_pred_quat is not None

        # Rollout payloads → in-memory EpisodeOutput completions in order.
        # Egress to a transport happens later in get_rollout_output, not here.
        assert len(results) == 2
        assert [len(result.completions) for result in results] == [2, 2]
        completion_scene_ids = [
            {completion.scene_id for completion in result.completions} for result in results
        ]
        assert completion_scene_ids == [{"scene_000"}, {"scene_001"}]
    finally:
        rollout.shutdown()
