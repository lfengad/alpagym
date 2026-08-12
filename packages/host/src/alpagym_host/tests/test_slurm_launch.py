# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from alpagym_host.config import SeparateNodesSlurmTopologyConfig, SlurmConfig
from alpagym_host.run_topology import RunHostPlan, RunTopologyPlan
from alpagym_host.slurm import (
    _cosmos_launcher_script,
    _gpu_mask,
    build_cosmos_srun_command,
    build_wizard_srun_command,
)


def test_build_wizard_srun_command_gives_the_wizard_every_gpu_on_the_host() -> None:
    """The Wizard step must see ALL of the host's GPUs, not just AlpaSim's share.

    A mask here constrains only the Wizard's own id validation, which it does against its
    RENUMBERED view (0..n-1). Service placement is not constrained at all: the Wizard spawns each
    service with its own `srun --overlap` carrying no GPU flags, so a service inherits the whole
    allocation and picks its device from `CUDA_VISIBLE_DEVICES=<physical topology id>`. Masking
    therefore rejects exactly the ids that would land correctly, and accepts ids that land on the
    trainer's GPUs.
    """
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=1,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=4,
        alpasim_gpus=4,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(),
        wizard_command=["python", "-m", "alpasim.wizard"],
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    assert "--nodelist=mixed-0" in command
    # 4 AlpaSim + 4 cosmos: the whole host, so a physical id means the same thing inside the
    # Wizard's view as it does in the topology config.
    assert "--gpus-per-task=8" in command
    assert not any(arg.startswith("--gpu-bind") for arg in command)
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(command)


def test_build_wizard_srun_command_scrubs_uv_project_env_before_exec() -> None:
    """The Wizard runs under `bash -lc`, which re-sources /etc/environment, so the script
    must unset the leaked venv vars before exec to keep them out of the service srun's.
    """
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=1,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=4,
        alpasim_gpus=4,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(),
        wizard_command=["python", "-m", "alpasim.wizard"],
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    script = command[-1]
    assert "unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV" in script
    assert script.index("unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV") < script.index("exec ")


def test_build_wizard_srun_command_disables_cpu_binding_for_nonexclusive_step() -> None:
    """Partial-node Wizard srun steps must not inherit packed CPU binding."""
    host = RunHostPlan(
        hostname="mixed-0",
        host_index=0,
        runs_cosmos=True,
        runs_alpasim=True,
        cosmos_gpus=2,
        alpasim_gpus=1,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(exclusive=False),
        wizard_command=["python", "-m", "alpasim.wizard"],
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    assert "--overlap" in command
    assert "--cpu-bind=none" in command


def test_build_wizard_srun_command_keeps_default_cpu_binding_for_exclusive_step() -> None:
    host = RunHostPlan(
        hostname="alpasim-0",
        host_index=0,
        runs_cosmos=False,
        runs_alpasim=True,
        cosmos_gpus=0,
        alpasim_gpus=8,
    )

    command = build_wizard_srun_command(
        host=host,
        slurm=_slurm_config(exclusive=True),
        wizard_command=["python", "-m", "alpasim.wizard"],
        log_path=Path("/tmp/alpagym/logs/wizard_0.log"),
    )

    assert "--cpu-bind=none" not in command


def test_build_cosmos_srun_command_dispatches_per_task_to_the_posttrain_entry() -> None:
    """Each SLURM_PROCID branch execs its own prebuilt command for the closed-loop entry."""
    topology = RunTopologyPlan(
        hosts=(
            RunHostPlan(
                hostname="policy-0",
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=False,
                cosmos_gpus=4,
                alpasim_gpus=0,
            ),
            RunHostPlan(
                hostname="mixed-0",
                host_index=1,
                runs_cosmos=True,
                runs_alpasim=True,
                cosmos_gpus=4,
                alpasim_gpus=4,
            ),
        )
    )

    command = build_cosmos_srun_command(
        cosmos_hosts=topology.cosmos_host_plans,
        slurm=_slurm_config(),
        container_image="/containers/alpagym.sqsh",
        workspace_sync_command=[
            "uv",
            "sync",
            "--frozen",
            "--inexact",
            "--all-packages",
            "--project",
            "/workspace/alpagym",
        ],
        worker_commands=(
            [
                "uv",
                "run",
                "python",
                "-m",
                "projects.cosmos3.posttrain.entrypoints.alpagym_clrl",
                "--resolved-config",
                "/tmp/resolved_config.yaml",
                "--gpus",
                "4",
            ],
            [
                "uv",
                "run",
                "python",
                "-m",
                "projects.cosmos3.posttrain.entrypoints.alpagym_clrl",
                "--resolved-config",
                "/tmp/resolved_config.yaml",
                "--gpus",
                "4",
            ],
        ),
        log_dir=Path("/tmp/alpagym/logs"),
    )

    script = command[-1]
    # PYTHONPATH is settled BEFORE the sync: this runs under `bash -lc`, so the login shell has
    # already run and anything it left on PYTHONPATH would otherwise be inherited.
    assert script.startswith(
        "unset PYTHONPATH\nuv sync --frozen --inexact --all-packages --project /workspace/alpagym\n"
    )
    branch = script.split("  1)", maxsplit=1)[1].split("    ;;", maxsplit=1)[0]
    for expected_arg in (
        "-m projects.cosmos3.posttrain.entrypoints.alpagym_clrl",
        "--resolved-config /tmp/resolved_config.yaml",
        "--gpus 4",
    ):
        assert expected_arg in branch
    assert "--gpu-bind=mask_gpu:0xf" in command
    assert "ALPAGYM_WORKER_INDEX" not in command[-1]


def test_gpu_mask_preserves_non_contiguous_gpu_ids() -> None:
    """GPU binding masks preserve sparse GPU id selections."""
    assert _gpu_mask((0, 2, 4)) == "0x15"


def _slurm_config(*, exclusive: bool = True) -> SlurmConfig:
    """Build Slurm settings used by launch command tests."""
    return SlurmConfig(
        job_name="alpagym",
        partition="batch",
        account="av",
        time="02:00:00",
        nodes=2,
        gpus_per_node=8,
        exclusive=exclusive,
        cpus_per_task=16,
        container_image="/containers/alpagym.sqsh",
        container_cache_root=None,
        container_workdir="/workspace/alpagym",
        uv_cache_dir="/tmp/uv",
        container_mounts=["/host/data:/container/data", "/tmp/uv:/tmp/uv"],
        export_env=["UV_CACHE_DIR=/tmp/uv"],
        topology=SeparateNodesSlurmTopologyConfig(cosmos_nodes=2, alpasim_nodes=1),
    )


def test_posttrain_repo_root_is_assigned_to_pythonpath() -> None:
    """PYTHONPATH is ASSIGNED, never appended to.

    The Cosmos step runs under `bash -lc`, so the login shell has already run; a checkout leaking
    in that way once shadowed the pinned cosmos-rl revision (see ALPAGYM_RUNBOOK.md). Appending
    would reopen exactly that door, and `srun --export` cannot win against the login shell either
    -- which is why this is set in the script rather than in `export_env`.
    """
    script = _cosmos_launcher_script(
        workspace_sync_command=["uv", "sync"],
        worker_commands=(["python", "-m", "projects.cosmos3.posttrain.entrypoints.alpagym_clrl"],),
        posttrain_repo_root="/repo/imaginaire4",
    )
    assert script.startswith("export PYTHONPATH=/repo/imaginaire4\n")
    assert "$PYTHONPATH" not in script


def test_pythonpath_is_cleared_when_no_repo_root_is_configured() -> None:
    """With no root configured the old behaviour stands: clear it rather than inherit it."""
    script = _cosmos_launcher_script(
        workspace_sync_command=["uv", "sync"],
        worker_commands=(["python", "-m", "whatever"],),
    )
    assert script.startswith("unset PYTHONPATH\n")
