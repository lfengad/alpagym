#!/bin/bash
# Launch alpamayo_1_5_clrl_test_run inside an existing Slurm allocation.
# Run from a shell INSIDE the allocation (srun --overlap --jobid ...), outside the
# container, so SLURM_NODELIST and srun/scontrol are available.
#
# Site notes for this cluster:
#  - No Docker, so `alpasim-base` cannot be built. A stock CUDA image stands in for
#    it (symlinked as alpasim_base_<version>.sqsh) and deploy=cw_dfw_slurm mounts the
#    AlpaSim checkout at /repo so those services find the venv it already carries.
#  - Slurm renumbers the GPUs it binds to a step, so AlpaSim addresses its four as 0-3.
#  - The container image has no redis-server, which Cosmos-RL starts itself; the
#    binary and its two private libs are bind-mounted in from lustre.
#  - The host CLI runs from .venv-host directly rather than through `uv run`: setting
#    UV_PROJECT_ENVIRONMENT here would leak through `srun --export=ALL` and override
#    the /opt/venv the Cosmos step builds for itself.
set -euo pipefail

BASE=/lustre/fsw/portfolios/sw/projects/sw_aidot/users/liangf
CACHE=$BASE/.cache/alpagym
UVC=$CACHE/uv
POLICY_CONFIGS=$BASE/alpagym/packages/policies/alpamayo_r1/src/alpagym_alpamayo_r1/configs

# Non-interactive shells skip ~/.bashrc; pull just the HF token from it.
eval "$(grep -m1 '^export HF_TOKEN=' "$HOME/.bashrc")"
# Compute nodes have no git-lfs; AlpaSim is an LFS repo. Use the staged copy.
export PATH="$BASE/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$UVC
# AlpaSim imports its service images with enroot and reads this from the
# environment; /etc/enroot/enroot.conf only sets enroot's own default.
export ENROOT_CONFIG_PATH="$HOME/.config/enroot"

mkdir -p "$UVC" "$CACHE/sqsh" "$CACHE/alpasim/checkouts"
cd "$BASE/alpagym"

MODE=${1:-cfg}    # cfg = print composed config only, run = actually launch
STEPS=${2:-1}     # closed-loop steps; reaches the entry as --steps (run_lifecycle.py)

HYDRA_ARGS=(
  "hydra.searchpath=[file://$POLICY_CONFIGS]"
  topology=slurm_full_node_1_3_4
  alpasim.wizard_args.topology=alpagym_4gpu
  alpasim.wizard_args.deploy=cw_dfw_slurm
  experiment=alpamayo_1_5_clrl_test_run
  policy.model.path="$BASE/alpagym/tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt"

  # Step count comes from $2; the default of 1 just proves the loop closes.
  cosmos.train.max_num_steps=$STEPS
  cosmos.train.num_epochs=1

  cache_root_dir="$CACHE"
  alpasim.checkout_cache_dir="$CACHE/alpasim/checkouts"

  execution.slurm.partition=interactive
  execution.slurm.account=sw_aidot
  execution.slurm.container_image="$BASE/images/torch.25.05.sqsh"
  execution.slurm.container_cache_root="$CACHE/sqsh"
  execution.slurm.uv_cache_dir="$UVC"
  execution.slurm.exclusive=false
  # Repo root for `projects.cosmos3.posttrain`, as seen inside the container. The closed-loop
  # entry lives there and imports absolutely. Assigned to PYTHONPATH inside the launcher
  # script (not via export_env): the step runs under `bash -lc`, so a login shell would win
  # over `srun --export`.
  execution.slurm.posttrain_repo_root="$BASE/imaginaire4"
  "execution.slurm.container_mounts=[\"/lustre:/lustre\",\"$UVC:$UVC\",\"$BASE/bin/redis-server:/usr/bin/redis-server\",\"$BASE/lib/liblzf.so.1:/usr/lib/x86_64-linux-gnu/liblzf.so.1\",\"$BASE/lib/libjemalloc.so.2:/usr/lib/x86_64-linux-gnu/libjemalloc.so.2\"]"
  "execution.slurm.export_env=[\"UV_CACHE_DIR=$UVC\",\"UV_PROJECT_ENVIRONMENT=/opt/venv\",\"VIRTUAL_ENV=/opt/venv\"]"
)

HOST_PYTHON=$BASE/alpagym/.venv-host/bin/python

if [ "$MODE" = "cfg" ]; then
  exec "$HOST_PYTHON" -m alpagym_host.cli --cfg job "${HYDRA_ARGS[@]}"
else
  exec "$HOST_PYTHON" -m alpagym_host.cli "${HYDRA_ARGS[@]}"
fi
