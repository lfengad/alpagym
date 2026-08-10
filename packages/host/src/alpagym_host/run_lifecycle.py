# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import signal
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import FrameType
from typing import cast

import grpc
import yaml
from alpasim_grpc.v0.common_pb2 import Empty
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceStub

from alpagym_host.alpasim_dependency import resolve_alpasim_checkout
from alpagym_host.alpasim_wizard import (
    _build_wizard_command,
    ensure_process_terminated,
    start_wizard,
    wait_for_runtime_ready,
)
from alpagym_host.config import ExecutionBackend, RunConfig, alpagym_project_root
from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint
from alpagym_host.log_organizer import organize_role_logs, tee_role_logs
from alpagym_host.run_topology import (
    RunHostPlan,
    RunTopologyPlan,
    build_local_topology,
    build_slurm_topology,
)
from alpagym_host.slurm import (
    allocated_hostnames,
    build_cosmos_srun_command,
    build_wizard_srun_command,
    prepare_container_image,
)
from alpagym_host.transport_env import apply_transport_env_vars


def fetch_runtime_info(host: str, port: int, timeout_s: float) -> tuple[int, list[str]]:
    """Fetch capacity and resolved scenes from an AlpaSim RuntimeService.

    Args:
        host: RuntimeService host.
        port: RuntimeService port.
        timeout_s: Connection and request timeout in seconds.

    Returns:
        Tuple of maximum supported concurrent rollouts and resolved scene ids.
    """
    target = f"{host}:{port}"
    with grpc.insecure_channel(target) as channel:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        info = RuntimeServiceStub(channel).get_runtime_info(Empty(), timeout=timeout_s)
    return int(info.max_supported_concurrent_rollouts), [
        str(scene.scene_id) for scene in info.scenes
    ]


def validate_local_process_config(execution_backend: ExecutionBackend) -> None:
    """Validate local-process prerequisites before AlpaSim Wizard startup."""
    if execution_backend is not ExecutionBackend.local_process:
        return
    if shutil.which("docker") is None:
        raise ValueError(
            "execution.backend=local_process requires Docker in PATH because AlpaSim Wizard "
            "uses Docker Compose for local launches."
        )
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "execution.backend=local_process requires `docker compose` to be available."
        ) from exc


class _AutoresumeTimeout(Exception):
    """Raised from the pre-timeout SIGUSR1 handler to unwind the run for a requeue."""


def _raise_autoresume_timeout(signum: int, frame: FrameType | None) -> None:
    """Turn the pre-timeout SIGUSR1 into an exception so the run tears down cleanly."""
    raise _AutoresumeTimeout


def execute_run(config: RunConfig) -> None:
    """Execute a run through the shared local and Slurm lifecycle.

    With ``execution.slurm.autoresume``, a pre-timeout SIGUSR1 tears the run
    down and requeues the job, which then resumes from the latest checkpoint.
    """
    execution_backend = ExecutionBackend(config.execution.backend)
    validate_local_process_config(execution_backend)
    autoresume = execution_backend.is_slurm_run and config.execution.slurm.autoresume
    # Install the handler before the try so a SIGUSR1 arriving during Slurm setup
    # (allocation, image import) still unwinds into the requeue path below rather
    # than killing the process with the default action.
    if autoresume:
        signal.signal(signal.SIGUSR1, _raise_autoresume_timeout)
    wizard_processes: list[subprocess.Popen[str]] = []
    requeue_for_autoresume = False
    try:
        scene_selector = (
            f"{len(config.dataset.scene_ids)} scene ids"
            if config.dataset.scene_ids is not None
            else f"test suite {config.dataset.test_suite_id!r}"
        )
        logging.info(
            "Starting AlpaGym run: backend=%s run_dir=%s dataset=%s policy_replicas=%d "
            "rollout_replicas=%d",
            execution_backend.value,
            config.artifact_paths.run_dir,
            scene_selector,
            config.cosmos.launch.policy_replicas,
            config.cosmos.launch.rollout_replicas,
        )
        if execution_backend.is_slurm_run:
            hostnames = allocated_hostnames()
            logging.info(
                "Resolved Slurm allocation: hostnames=%s gpus_per_node=%d",
                ",".join(hostnames),
                config.execution.slurm.gpus_per_node,
            )
            topology = build_slurm_topology(
                backend=execution_backend,
                hostnames=hostnames,
                gpus_per_node=config.execution.slurm.gpus_per_node,
                topology=config.execution.slurm.topology,
            )
            logging.info(
                "Preparing Slurm container image: image=%s cache_root=%s",
                config.execution.slurm.container_image,
                config.execution.slurm.container_cache_root,
            )
            container_image = prepare_container_image(
                container_image=cast(str, config.execution.slurm.container_image),
                container_cache_root=config.execution.slurm.container_cache_root,
            )
            logging.info("Using Slurm container image: %s", container_image)
        else:
            topology = build_local_topology()
            container_image = None

        _log_topology(topology)
        alpasim_checkout_root = resolve_alpasim_checkout(config=config.alpasim)
        registry = FileTopologyRegistry(config.artifact_paths.topology_registry_dir)
        if autoresume:
            # A requeue reuses the run dir; reset the prior attempt's AlpaSim
            # rendezvous so this attempt rebuilds it. Normal launches keep the
            # fail-fast exclusive publish on an accidentally reused run dir.
            registry.reset_alpasim_topology()
        alpasim_hosts = topology.alpasim_host_plans
        logging.info("Starting %d AlpaSim Wizard process(es)", len(alpasim_hosts))
        for runtime_index, host in enumerate(alpasim_hosts):
            wizard_processes.append(
                _start_wizard_process(
                    config=config,
                    execution_backend=execution_backend,
                    host=host,
                    runtime_index=runtime_index,
                    alpasim_checkout_root=alpasim_checkout_root,
                )
            )

        logging.info("Waiting for %d AlpaSim runtime endpoint(s)", len(alpasim_hosts))
        runtime_scene_ids: list[str] | None = None
        for runtime_index, (host, process) in enumerate(
            zip(alpasim_hosts, wizard_processes, strict=True)
        ):
            _ensure_wizard_processes_running(wizard_processes)
            wizard_log_dir = _wizard_log_dir(config=config, runtime_index=runtime_index)
            runtime_host, runtime_port = wait_for_runtime_ready(
                wizard_process=process,
                runtime_server_path=wizard_log_dir / "generated-runtime-server.yaml",
                timeout_s=config.alpasim.startup_timeout_s,
                published_host=host.hostname,
            )
            runtime_capacity, scene_ids = fetch_runtime_info(
                runtime_host,
                runtime_port,
                timeout_s=config.alpasim.startup_timeout_s,
            )
            if not scene_ids:
                raise ValueError(f"AlpaSim runtime {runtime_index} reported no scenes")
            if runtime_scene_ids is None:
                runtime_scene_ids = scene_ids
            elif scene_ids != runtime_scene_ids:
                raise ValueError(
                    "AlpaSim runtimes reported different scene lists: "
                    f"{runtime_scene_ids!r} != {scene_ids!r}"
                )
            endpoint = TopologyEndpoint(
                id=f"alpasim-runtime-{runtime_index}",
                host=runtime_host,
                port=runtime_port,
                capacity=runtime_capacity,
            )
            registry.publish_alpasim_runtime(endpoint)
            logging.info(
                "Published AlpaSim runtime: id=alpasim-runtime-%d host=%s port=%d capacity=%d",
                runtime_index,
                runtime_host,
                runtime_port,
                runtime_capacity,
            )

        config.artifact_paths.alpasim_scene_ids_path.write_text(
            yaml.safe_dump(
                {"scene_ids": runtime_scene_ids or []},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        cosmos_command = _build_cosmos_command(
            config=config,
            execution_backend=execution_backend,
            topology=topology,
            container_image=container_image,
        )
        logging.info(
            "Starting Cosmos launcher: backend=%s cosmos_hosts=%s log_dir=%s",
            execution_backend.value,
            ",".join(topology.cosmos_hosts),
            config.artifact_paths.log_dir,
        )
        logging.info("Starting Cosmos launcher command: %s", cosmos_command)
        # Write the NCCL fabric env to the host os.environ right before the Cosmos
        # launch. Locally the subprocess inherits it; on Slurm `srun --export=ALL`
        # carries it to every Policy/Rollout/Controller worker (and `bash -lc` does
        # not strip NCCL_*). It governs both the AlpaGym data plane and cosmos-rl's
        # native weight-sync mesh. Disk runs leave nccl_env empty, so this is a no-op.
        apply_transport_env_vars(config.transport)
        if config.transport.nccl_env:
            logging.info(
                "Applied NCCL fabric env to os.environ: %s", dict(config.transport.nccl_env)
            )
        # Unbuffer worker stdout so cosmos-rl's import-time print() output (e.g. its
        # model auto-discovery) is interleaved in order instead of being flushed in a
        # block when the process exits.
        os.environ["PYTHONUNBUFFERED"] = "1"
        # Only local runs inherit this terminal; Slurm replicas write their own logs.
        tee_logs = (
            tee_role_logs(config.artifact_paths.log_dir)
            if not execution_backend.is_slurm_run
            else nullcontext()
        )
        with organize_role_logs(config.artifact_paths.log_dir), tee_logs:
            subprocess.run(
                cosmos_command,
                check=True,
                text=True,
            )
        logging.info("Cosmos launcher completed")
    except _AutoresumeTimeout:
        requeue_for_autoresume = True
        logging.info("Pre-timeout SIGUSR1 received; tearing down to requeue the Slurm job")
    finally:
        if wizard_processes:
            logging.info("Stopping %d AlpaSim Wizard process(es)", len(wizard_processes))
        for process in wizard_processes:
            ensure_process_terminated(process)

    if requeue_for_autoresume:
        job_id = os.environ["SLURM_JOB_ID"]
        logging.info("Requeuing Slurm job %s for autoresume", job_id)
        subprocess.run(["scontrol", "requeue", job_id], check=True, text=True)


def _start_wizard_process(
    config: RunConfig,
    execution_backend: ExecutionBackend,
    host: RunHostPlan,
    runtime_index: int,
    alpasim_checkout_root: Path,
) -> subprocess.Popen[str]:
    """Start one Wizard process for a topology host."""
    wizard_log_dir = _wizard_log_dir(config=config, runtime_index=runtime_index)
    wizard_log_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Starting AlpaSim Wizard: runtime_index=%d host=%s log_dir=%s",
        runtime_index,
        host.hostname,
        wizard_log_dir,
    )
    if not execution_backend.is_slurm_run:
        return start_wizard(
            config=config.alpasim,
            execution_backend=execution_backend,
            dataset=config.dataset,
            alpasim_run_dir=wizard_log_dir,
            cwd=alpasim_checkout_root,
        )

    wizard_command = _build_wizard_command(
        config=config.alpasim,
        execution_backend=execution_backend,
        dataset=config.dataset,
        alpasim_run_dir=wizard_log_dir,
        checkout_root=alpasim_checkout_root,
    )
    command = build_wizard_srun_command(
        host=host,
        slurm=config.execution.slurm,
        wizard_command=wizard_command,
        log_path=(config.artifact_paths.log_dir / f"wizard_{runtime_index}.log").resolve(),
    )
    logging.info(
        "Submitting AlpaSim Wizard through srun: runtime_index=%d host=%s slurm_log=%s",
        runtime_index,
        host.hostname,
        config.artifact_paths.log_dir / f"wizard_{runtime_index}.log",
    )
    logging.info("Submitting AlpaSim Wizard command: %s", command)
    # `start_new_session=True` makes the srun client its own process-group leader, so the
    # shared `ensure_process_terminated` can `os.killpg` it on cleanup (srun then forwards
    # the signal to the remote Wizard step). Without it killpg targets a non-existent group
    # and silently no-ops, leaking the Wizard srun. This mirrors the local `start_wizard`.
    return subprocess.Popen(command, cwd=alpasim_checkout_root, start_new_session=True, text=True)


def _wizard_log_dir(config: RunConfig, runtime_index: int) -> Path:
    """Return the Wizard run directory for one runtime index."""
    return (config.artifact_paths.alpasim_log_dir / f"wizard_{runtime_index}").resolve()


def _build_cosmos_command(
    config: RunConfig,
    execution_backend: ExecutionBackend,
    topology: RunTopologyPlan,
    container_image: str | None,
) -> list[str]:
    """Build the Cosmos launcher command for the selected execution backend."""
    if not execution_backend.is_slurm_run:
        return _build_cosmos_launcher_command(
            config,
            project_root=alpagym_project_root(),
            no_sync=False,
            cosmos_gpus=topology.cosmos_host_plans[0].cosmos_gpus,
        )

    Path(config.execution.slurm.uv_cache_dir).mkdir(parents=True, exist_ok=True)
    cosmos_hosts = topology.cosmos_host_plans
    if len(cosmos_hosts) != 1:
        raise NotImplementedError(
            f"the posttrain entry runs on one host; got {len(cosmos_hosts)}. Multi-node needs a "
            "Ray cluster stood up across the allocation (RAY_ADDRESS), which replaced cosmos-rl's "
            "controller/worker addressing and is not built yet"
        )
    worker_commands: list[list[str]] = [
        _build_cosmos_launcher_command(
            config,
            project_root=Path(config.execution.slurm.container_workdir),
            no_sync=True,
            cosmos_gpus=cosmos_hosts[0].cosmos_gpus,
        )
    ]
    return build_cosmos_srun_command(
        cosmos_hosts=cosmos_hosts,
        slurm=config.execution.slurm,
        container_image=cast(str, container_image),
        workspace_sync_command=[
            "uv",
            "sync",
            "--frozen",
            "--inexact",
            "--all-packages",
            "--project",
            str(config.execution.slurm.container_workdir),
        ],
        worker_commands=tuple(worker_commands),
        log_dir=config.artifact_paths.log_dir,
    )


def _log_topology(topology: RunTopologyPlan) -> None:
    """Log the planned host roles and GPU placement."""
    logging.info(
        "Run topology: hosts=%d cosmos_hosts=%s alpasim_hosts=%s",
        len(topology.hosts),
        ",".join(topology.cosmos_hosts),
        ",".join(topology.alpasim_hosts),
    )
    for host in topology.hosts:
        logging.info(
            "Run host plan: host_index=%d hostname=%s cosmos_gpus=%d "
            "alpasim_gpus=%d cosmos_gpu_ids=%s alpasim_gpu_ids=%s",
            host.host_index,
            host.hostname,
            host.cosmos_gpus,
            host.alpasim_gpus,
            ",".join(str(gpu_id) for gpu_id in host.cosmos_gpu_ids) or "-",
            ",".join(str(gpu_id) for gpu_id in host.alpasim_gpu_ids) or "-",
        )


def _build_cosmos_launcher_command(
    config: RunConfig,
    project_root: Path,
    no_sync: bool,
    cosmos_gpus: int,
) -> list[str]:
    """Build the command that runs the closed-loop RL half of the run.

    This is posttrain's `entrypoints.alpagym_clrl`, which replaced
    `cosmos_rl.launcher.launch_all`. cosmos-rl's `--policy`/`--rollout`/`--num-workers`/
    `--worker-idx`/`--port`/`--url` have no counterpart: those addressed its controller and worker
    processes, and Ray does that now. The replica counts are likewise gone -- the entry derives
    both meshes from `cosmos_gpus` (policy `(1, gpus)` FSDP, rollout `(gpus, 1)`).

    `allowed_outdated_steps` is deliberately not passed. Trainer and rollout alternate, so every
    trajectory is consumed at the step that produced it: on-policy is structural here, and a knob
    whose only correct value is 0 would be a compatibility artifact for a code path that is gone.

    Args:
        config: the resolved run config; the entry re-reads it from disk inside each Ray actor.
        project_root: the uv project to run from -- the checkout on the host, or the container
            workdir under Slurm.
        no_sync: skip uv's dependency sync (the Slurm step syncs once, before this command).
        cosmos_gpus: GPUs available to the cosmos half of the node.
    """
    command = ["uv", "run"]
    if no_sync:
        command.append("--no-sync")
    else:
        command.append("--all-packages")
    launcher_args = [
        "--project",
        str(project_root),
    ]
    if no_sync:
        launcher_args.extend(["--package", "alpagym-runtime"])
    group_size = int(config.cosmos.rollout.n_generation)
    train_batch = int(config.cosmos.train.train_batch_per_replica)
    if train_batch % group_size:
        raise ValueError(
            f"train_batch_per_replica={train_batch} is not a multiple of n_generation={group_size}; "
            "a partial group would reach the group-relative advantage estimator and skew its "
            "per-prompt mean and std"
        )
    launcher_args.extend(
        [
            "python",
            "-m",
            "projects.cosmos3.posttrain.entrypoints.alpagym_clrl",
            "--resolved-config",
            str(config.artifact_paths.resolved_config_path),
            "--gpus",
            str(cosmos_gpus),
            "--steps",
            str(config.cosmos.train.max_num_steps),
            "--prompts-per-step",
            str(train_batch // group_size),
            "--group-size",
            str(group_size),
        ]
    )
    command.extend(launcher_args)
    return command


def _ensure_wizard_processes_running(processes: list[subprocess.Popen[str]]) -> None:
    """Raise if any Wizard process exited before runtime readiness."""
    for process in processes:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"AlpaSim Wizard exited before readiness with code {return_code}")
