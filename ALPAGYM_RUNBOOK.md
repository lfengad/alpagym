# AlpaGym on cw-dfw-cs-001 — Runbook

Site-specific notes for running AlpaGym's closed-loop RL on the `cw-dfw-cs-001`
cluster (8×H100 per node, Slurm + pyxis/enroot, **no Docker**). Not part of
upstream AlpaGym; keep it out of PRs to NVlabs/alpagym.

Verified end to end on 2026-08-10: 22 valid closed-loop policy steps per episode,
6 rollouts, one GRPO step, checkpoint written.

## Table of Contents

- [Quick Start](#quick-start)
- [Machines and Paths](#machines-and-paths)
- [What Persists Between Allocations](#what-persists-between-allocations)
- [Cold Start](#cold-start)
- [Why This Cluster Needs Adaptation](#why-this-cluster-needs-adaptation)
- [Reading a Run](#reading-a-run)
- [Troubleshooting](#troubleshooting)
- [Upstream Bugs Worth Reporting](#upstream-bugs-worth-reporting)

## Quick Start

Everything heavy is already staged on `/lustre`. A run is three commands.

```bash
# 1. Get an allocation (login host shell). ~/run.sh is interactive; for an
#    unattended one, swap `--pty bash -i` for `sleep infinity` and background it.
ssh liangf@cw-dfw-cs-001-vscode-02.cw-dfw-cs-001.hpc.nvidia.com
bash ~/run.sh

# 2. Note the JOBID.
squeue -u liangf -o "%.10i %.9T %.11L %j" -h

# 3. Launch, from inside the allocation but OUTSIDE the container.
srun --overlap --jobid <JOBID> bash -c 'bash ~/work/run_alpagym_clrl.sh run'
```

### Driving this from an agent session

Two things bite immediately.

**This is the x86 cluster.** `cw-dfw-cs-001` has 8×H100 per node. The other
cluster in these notes, `gcp-iad-cs-001`, is GB200/aarch64 and **cannot run
AlpaGym at all** — the workspace lockfile is pinned to `x86_64` and flash-attn
has no aarch64 wheel, so `uv sync` fails before anything starts.

**The login shell is `csh`.** `$(...)`, `VAR=$(...)` and `2>&1` all fail there
with `Illegal variable name` / `Ambiguous output redirect`. Never pass bash
syntax as an ssh argument; pipe it to `bash -s`:

```bash
ssh -o BatchMode=yes liangf@cw-dfw-cs-001-vscode-02.cw-dfw-cs-001.hpc.nvidia.com 'bash -s' <<'EOF'
squeue -u liangf -o "%.10i %.9T %.11L %j" -h
EOF
```

An agent cannot hold `~/run.sh`'s interactive PTY. Launch a detached equivalent
instead — same settings, `sleep infinity` in place of `--pty bash -i`:

```bash
ssh -o BatchMode=yes liangf@cw-dfw-cs-001-vscode-02.cw-dfw-cs-001.hpc.nvidia.com 'bash -s' <<'EOF'
B=/lustre/fsw/portfolios/sw/projects/sw_aidot/users/liangf
nohup srun -N 1 -A sw_aidot --gpus-per-node 8 --time 4:0:0 \
  --partition=interactive \
  --container-image $B/images/torch.25.05.sqsh \
  --container-mounts /lustre:/lustre \
  --container-name liangf_dev --export=ALL --container-workdir=/ \
  sleep infinity > $B/slurm-node.log 2>&1 &
EOF
```

Then poll `squeue` for `RUNNING`, and run step 3 through `bash -s` as well. The
run takes ~8 minutes and prints nothing useful to stdout — tail the logs under
`tmp/alpagym-runs/<newest>/` instead (see [Reading a Run](#reading-a-run)).

`run_alpagym_clrl.sh cfg` prints the composed Hydra config and exits — use it to
check a config change without burning a run.

The script hardcodes no job id: the host CLI reads `SLURM_NODELIST`, so it binds
to whatever allocation it runs inside.

Expect ~9-10 minutes on a node that has not run this before: ~2 min for the
AlpaSim services, ~4 min to load three copies of the 10B policy, ~1 min of
rollout, ~2 min for the GRPO step and checkpoint. Measured 9.5 min end to end on
a cold node.

## Machines and Paths

| Where | Path |
|---|---|
| Local — read and edit only | `/workspace/alpagym` |
| Login host — run here | `~/work/alpagym` |
| Real path (and inside containers) | `/lustre/fsw/portfolios/sw/projects/sw_aidot/users/liangf/alpagym` |

`~/work` → `/lustre/fsw/portfolios/sw/users/liangf` → `../projects/sw_aidot/users/liangf`.
All three spellings are the same directory; containers only mount `/lustre`, so
use the long form there.

The host CLI must run **inside the allocation, outside the container**: it needs
`SLURM_NODELIST` plus `srun`/`scontrol`, which the container does not have. It
then fans work out with `srun --overlap`. Nothing is submitted with `sbatch`.

## What Persists Between Allocations

Losing an allocation costs nothing but the allocation. These all live on
`/lustre` and survive:

| Asset | Path (under `~/work/`) |
|---|---|
| Host CLI venv | `alpagym/.venv-host` |
| uv cache (~20 GB) | `.cache/alpagym/uv` |
| Alpamayo 1.5 10B + converted ckpt (~21 GB) | `alpagym/tmp/checkpoints/` |
| AlpaSim checkout + its venv | `.cache/alpagym/alpasim/checkouts/62400020a95f` |
| NRE renderer image (~26 GB) | `.../data/sqsh/nre_ga_26.04.sqsh` |
| `alpasim-base` stand-in (symlink) | `.../data/sqsh/alpasim_base_0.91.0.sqsh` |
| NuRec scene artifacts | `.../data/nre-artifacts/` |
| Staged `git-lfs`, `redis-server`, redis libs | `bin/`, `lib/` |
| Launch script | `run_alpagym_clrl.sh` |

What does **not** persist: anything `apt install`ed inside a container. That is
why `git-lfs` and `redis-server` are staged on `/lustre` instead.

## Cold Start

Only needed on a fresh cluster, or if the caches above are gone.

```bash
# Repo
cd ~/work && git clone https://github.com/NVlabs/alpagym.git

# Host CLI venv, on the LOGIN NODE (it has git-lfs; compute nodes do not).
# Keep it separate from the repo's .venv, and never export UV_PROJECT_ENVIRONMENT
# beyond this command — see Troubleshooting.
cd ~/work/alpagym
export UV_CACHE_DIR=~/work/.cache/alpagym/uv
UV_PROJECT_ENVIRONMENT=.venv-host uv sync --package alpagym-host

# Stage the binaries compute nodes lack.
mkdir -p ~/work/bin ~/work/lib
cp /usr/bin/git-lfs ~/work/bin/
# redis-server has to come out of a container (the login node has none):
#   srun --overlap --jobid <JOBID> --container-name liangf_dev bash -s <<'EOF'
#   apt-get update -qq && apt-get install -y -qq redis-server
#   cp -L /usr/bin/redis-server /lustre/.../liangf/bin/
#   cp -L /usr/lib/x86_64-linux-gnu/{liblzf.so.1,libjemalloc.so.2} /lustre/.../liangf/lib/
#   EOF

# Model (inside a container — needs GPU-capable torch for the conversion).
# HF_TOKEN must be set and the account approved for the gated NuRec dataset.
uv run --no-sync python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('nvidia/Alpamayo-1.5-10B', local_dir='./tmp/checkpoints/Alpamayo-1.5-10B'))"
uv run --no-sync --package alpagym-alpamayo-r1 python \
  packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py \
  --input ./tmp/checkpoints/Alpamayo-1.5-10B \
  --output ./tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt --overwrite
```

The first `run_alpagym_clrl.sh run` then clones AlpaSim, syncs its venv, and
imports the NRE image from nvcr.io (~26 GB, a few minutes). Two manual steps are
still required after that first attempt fails:

```bash
C=~/work/.cache/alpagym/alpasim/checkouts/<hash>

# 1. alpasim-base cannot be built without Docker; point its sqsh at a stock image.
ln -sfn ~/work/images/torch.25.05.sqsh "$C/data/sqsh/alpasim_base_0.91.0.sqsh"

# 2. Restore the subpackages the non-editable install drops (see Upstream Bugs).
SP=$C/.venv/lib/python3.12/site-packages
for pkg in controller runtime; do
  for d in "$C/src/$pkg"/*/; do
    n=$(basename "$d"); [ -d "$SP/$n" ] || continue
    rsync -a --ignore-existing --exclude __pycache__ "$d" "$SP/$n/"
  done
done
```

## Why This Cluster Needs Adaptation

Upstream's `deploy=slurm` presets are explicitly "not expected to run out of the
box". Five site facts drive every change below.

**No Docker anywhere.** Not on the login node, not in the container, and there is
no passwordless sudo. `execution.backend=local_process` hard-raises without
`docker compose`, so the run uses `backend=slurm`, where AlpaSim's Wizard
dispatches services through `srun` instead. That backend comes from the
`slurm_full_node_1_3_4` topology, and `command=run` (the default `deploy=local`)
keeps execution inline rather than submitting a new job.

**Slurm renumbers bound GPUs.** `--gpu-bind=mask_gpu:0xf0` exposes the node's
GPUs 4-7 as 0-3 inside the step. Upstream's `alpagym_4gpu` addresses them as
physical 4-7, which fails validation. `alpagym_4gpu_local_ids` is the same
topology with local ids. The Cosmos step is bound the same way and also sees its
four as 0-3, so the two never collide.

**`alpasim-base` cannot be built.** Four services (`driver`, `physics`,
`trafficsim`, `controller`) use an image built from AlpaSim's Dockerfile;
`defines.image_registry` is empty, so it is expected to exist locally. With
`driver: null` and `trafficsim: skip`, only `physics`, `controller` and `runtime`
actually need it. The `cw_dfw_slurm` deploy preset points those three at the
stock CUDA image with the AlpaSim checkout bind-mounted at `/repo`, so `uv run`
finds the venv the checkout already carries. `ensure_sqsh_path` reuses an
existing `.sqsh`, so a symlink is enough to skip the import.

**The container has no `redis-server`.** Cosmos-RL starts Redis itself. The
binary and its two private libs (`liblzf`, `libjemalloc`; everything else is in
the base image) are bind-mounted in file by file, which avoids touching the
container's `PATH`.

**The login shell leaks environment into containers.** `srun --export=ALL` plus
`bash -lc` means anything in `~/.bashrc` reaches the container and can win over
values set explicitly in `export_env`. Two variables mattered; both are covered
in Troubleshooting.

## Reading a Run

Under `~/work/alpagym/tmp/alpagym-runs/<timestamp>-<id>/`:

| Path | What it holds |
|---|---|
| `logs/wizard_0.log` | AlpaSim bring-up; service dispatch and readiness |
| `alpasim/wizard_0/txt-logs/out-*-<service>-*.log` | One per service container |
| `logs/cosmos_0.log` | Cosmos launcher: GPU plan, replica commands |
| `logs/logs_latest/{controller,policy_0,rollout_*}.log` | Training and rollout |
| `cosmos/<ts>/checkpoints/step_N` | Cosmos-format checkpoint |
| `cosmos/<ts>/safetensors/step_N` | Export for AlpaSim eval |
| `resolved_config.yaml` | The frozen config the run actually used |

A healthy run reaches these in order:

```
All addresses open.                              # 9 sim services up
AlpaSim RuntimeService is ready at <host>:<port> # 10th (runtime) up
Detected 4 GPUs: 0, 1, 2, 3                      # cosmos side
Packed AlpaGym replay artifact ... valid_steps=22 padded_steps=0 reward=...
AlpaGym trainer step start current_step=1 total_steps=1
AlpaGym trainer step end ... loss_avg=... clip_fraction=...
[Policy] Saving huggingface checkpoint at step 1
Cosmos launcher completed
Stopping 1 AlpaSim Wizard process(es)
```

**`Cosmos launcher completed` is the success marker.** What follows it looks
alarming but is not:

```
srun: forcing job termination
srun: error: pool0-XXXXX: task 0: Killed
srun: Terminating StepId=<jobid>.N
```

That is AlpaGym tearing down the Wizard's srun step. The launch script's trailing
`RUN_EXIT=` line usually never gets written because the teardown takes the shell
with it — judge the run by `Cosmos launcher completed` and the checkpoint on
disk, not by an exit code.

`valid_steps` should equal `expected_valid_steps` (22) with `padded_steps=0`.
Padding means episodes are ending early — usually a sim-side failure, not a
config problem.

On the smoke config an untuned policy scores around -9.4, which under
`progress_safety` is roughly one collision (-10) plus a little progress. That is
expected, not a bug.

## Troubleshooting

Every failure hit while bringing this up, in the order they appear.

| Symptom | Cause | Fix |
|---|---|---|
| `no viable alternative at input '[UV_CACHE_DIR='` | Hydra's override grammar cannot parse `=` inside a list | Quote each element: `export_env=["K=V","K2=V2"]` |
| `Found no NVIDIA driver` while syncing `grouped-gemm` | `--all-packages` on the login node builds CUDA extensions | Host venv gets `--package alpagym-host` only; reach policy configs via `hydra.searchpath` |
| `Could not override 'experiment'` | The policy package is not installed, so its config group is invisible | `hydra.searchpath=[file://.../alpamayo_r1/.../configs]` |
| `no token was found` (HuggingFace) | Non-interactive shells skip `~/.bashrc` | Script pulls just the `HF_TOKEN` line out of it |
| `git-lfs: command not found` | Compute nodes lack git-lfs; AlpaSim is an LFS repo | Staged binary on `/lustre`, prepended to `PATH` |
| `requested GPUs [4,5,6,7] but only 0..3 are available` | Slurm renumbers bound GPUs | `alpasim.wizard_args.topology=alpagym_4gpu_local_ids` |
| `ENROOT_CONFIG_PATH is not set` | `/etc/enroot/enroot.conf` sets enroot's own default, not the env var | `export ENROOT_CONFIG_PATH=$HOME/.config/enroot` |
| `enroot import ... docker://alpasim-base:0.91.0` fails | Image has no registry and cannot be built without Docker | Symlink a stock `.sqsh` into the squash cache |
| `Failed to spawn: physics_server` / `No module named 'alpasim_controller'` | Stock image has no `/repo`, so `uv run` falls back to a bare interpreter | `deploy=cw_dfw_slurm` mounts the checkout at `/repo`, sets `workdir` and `UV_NO_SYNC=1` |
| `No module named 'alpasim_controller.mpc_impl'` | Non-editable install drops subpackages | Restore them from the source tree (see Cold Start) |
| `redis-server: command not found` | Cosmos-RL starts Redis; the image has none | Bind-mount the binary and its two libs |
| `'PolicyStatusManager' has no attribute '_publish_payload_transport_cleanup'` | A `PYTHONPATH` in `~/.bashrc` shadows the pinned cosmos-rl with a local checkout | Remove it from `~/.bashrc` (backup: `~/.bashrc.bak.alpagym`) |
| Host venv suddenly broken, `bin/python` a dangling symlink | `UV_PROJECT_ENVIRONMENT` leaked through `--export=ALL`; the container's `uv sync` overwrote it | Never export it; invoke `.venv-host/bin/python` directly |
| Hang at first weight sync, both GPUs at 100% | NCCL advertises a P2P transport that stalls | Prefix with `NCCL_P2P_DISABLE=1` (not needed on this cluster) |

**Orphaned steps.** A failed run leaves `alpasim-*` srun steps behind. They hold **ports and GPU
memory** — the renderers (`pycena_nrm_full_cc serve-grpc`) and `physics_server` sit on 16-17 GiB
each — and they accumulate fast across retries. A later job on the same node then OOMs for no
visible reason, so check `nvidia-smi` before believing your own footprint is at fault.

List, then cancel by explicit id — never pattern-match into `scancel`:

```bash
squeue -u liangf -s -h -o "%i %j" | grep alpasim-
scancel -s KILL <jobid>.<step> <jobid>.<step> ...   # never the bare jobid, never the sleep step
```

**`-s KILL` is required.** A plain `scancel <jobid>.<step>` left all ten steps running and still
holding their GPU memory 15 s later; with `-s KILL` they died and every GPU returned to 0 MiB.

Never cancel the step holding the allocation (`sleep`), or `.extern`. That is usually `.0` — but
not always: any probe run before the allocation's own step registers takes `.0` and pushes `sleep`
to `.1`. Read the `%j` column, don't assume the number.

## Upstream Bugs Worth Reporting

**AlpaSim — `packages.find` drops subpackages.** `src/controller/pyproject.toml`
and `src/runtime/pyproject.toml` both use:

```toml
[tool.setuptools.packages.find]
include = [ "alpasim_controller", "benchmark", "tests" ]
```

Without a trailing `*`, setuptools matches only the top-level package. Eight
subpackages are silently dropped, including `alpasim_runtime/{worker,daemon,services,telemetry}`
and `alpasim_controller/mpc_impl` — which the package's own entry points import.
Invisible upstream because `uv sync` installs workspace members editable; AlpaGym
uses `--no-editable` for a relocatable venv and hits it. Fix: `include = ["alpasim_controller*", ...]`.

**AlpaGym — the Cosmos step does not isolate `PYTHONPATH`.** `build_wizard_srun_command`
already unsets `UV_PROJECT_ENVIRONMENT`/`VIRTUAL_ENV` because `bash -lc` re-sources
the login environment. `_cosmos_launcher_script` needs the same treatment for
`PYTHONPATH`: a Cosmos-RL checkout on a user's `PYTHONPATH` silently shadows the
revision the workspace pins, and the runtime is written against that revision.
An `unset PYTHONPATH` is in place locally, though on this cluster the leak was
ultimately removed at the source.
