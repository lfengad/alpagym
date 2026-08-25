# AlpaGym on cw-dfw-cs-001 — Runbook

Site-specific notes for running AlpaGym's closed-loop RL on the `cw-dfw-cs-001`
cluster (8×H100 per node, Slurm + pyxis/enroot, **no Docker**). Not part of
upstream AlpaGym; keep it out of PRs to NVlabs/alpagym.

Verified end to end on 2026-08-10: 22 valid closed-loop policy steps per episode,
6 rollouts, one GRPO step, checkpoint written.

## Two Modes

The training half has been replaced; the AlpaSim half has not. Which mode you get is
decided by the checkout, not by a config flag — the launcher runs one entry
unconditionally, and keeping both alive behind a switch would mean maintaining two
parallel implementations of the config schema, the log format and the checkpoint layout.

| Mode | Checkout | Training entry |
|---|---|---|
| **cosmos-rl** (pre-migration) | `main` | `cosmos_rl.launcher.launch_all` |
| **posttrain** (current) | `posttrain-migration` | `projects.cosmos3.posttrain.entrypoints.alpagym_clrl` |

Everything up to and including [Why This Cluster Needs Adaptation](#why-this-cluster-needs-adaptation)
applies to both. From [Reading a Run](#reading-a-run) on, the sections say which mode they
describe. posttrain additionally needs imaginaire4 checked out — see
[Running the posttrain path](#running-the-posttrain-path).

Note this runbook lives only on `posttrain-migration`; `main` predates it. Read it from
here even when running the cosmos-rl mode.

This file is the OPERATIONAL half: how to bring a cluster up and what to do when it breaks.
What the posttrain entry actually does -- why the two meshes are transposed, how the loop is
shaped, and which parts of the cosmos-rl recipe it reproduces line for line -- is
`projects/cosmos3/posttrain/docs/alpagym.md` in imaginaire4. Neither file restates the other:
this one cannot verify imaginaire4's internals, and that one cannot verify this cluster.

## Table of Contents

- [Quick Start](#quick-start)
- [Machines and Paths](#machines-and-paths)
- [What Persists Between Allocations](#what-persists-between-allocations)
- [Cold Start](#cold-start)
- [Why This Cluster Needs Adaptation](#why-this-cluster-needs-adaptation)
- [Two Modes](#two-modes)
- [Reading a Run](#reading-a-run)
- [Running cosmos-rl from `main`](#running-cosmos-rl-from-main)
- [Running the posttrain path](#running-the-posttrain-path)
- [Troubleshooting](#troubleshooting)
- [Upstream Bugs Worth Reporting](#upstream-bugs-worth-reporting)

## Quick Start

Everything heavy is already staged on `/lustre`. A run is three commands.

Check out the branch for the mode you want first — see [Two Modes](#two-modes). The
launch script is `scripts/cw-dfw/run_alpagym_clrl.sh` in this repo; the commands below
use the copy at `~/work/run_alpagym_clrl.sh`, which is that file with `BASE` pointed at
your `/lustre` tree. Copy it there once (`cp scripts/cw-dfw/run_alpagym_clrl.sh
~/work/`) rather than editing the tracked one, so the committed copy stays the
reference.

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

**Slurm renumbers bound GPUs — but only for the process it binds.**
`--gpu-bind=mask_gpu:0xf0` makes the Wizard step see GPUs 4-7 as 0-3, so upstream's
`alpagym_4gpu` (which names them 4-7) fails the Wizard's own id validation here. The
services it launches are NOT renumbered: the Wizard issues its own `srun --overlap` per
service with no GPU flags at all, so each inherits the whole allocation and picks a
device by absolute `CUDA_VISIBLE_DEVICES`.

The two frames therefore invert: ids that pass validation (0-3) land on the trainer's
GPUs, and the ids that land correctly (4-7) are rejected before the run starts.

- **posttrain mode** — fixed. `build_wizard_srun_command` gives the Wizard step every
  GPU on the host, so validation and placement share one frame and stock `alpagym_4gpu`
  is correct. Verified: AlpaSim on 4-7, training alone on 0-3.
- **cosmos-rl mode (`main`)** — not fixed there; see
  [Running cosmos-rl from `main`](#running-cosmos-rl-from-main) for what to carry over.
  Do NOT "fix" it by renumbering the topology to 0-3: that passes validation and silently
  runs the simulator on the trainer's GPUs, which fits a single step and exhausts GPU 0
  once the optimizer states exist. See [Upstream Bugs](#upstream-bugs-worth-reporting).

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
| `logs/cosmos_0.log` | The training launcher — cosmos-rl's GPU plan and replica commands, or the posttrain entry's Ray bring-up and per-step metrics |
| `logs/logs_latest/{controller,policy_0,rollout_*}.log` | cosmos-rl only: training and rollout |
| `logs/actor_<pid>.log` | posttrain only: one per Ray actor — trainer ranks, rollout replicas, reward/buffer |
| `cosmos/<ts>/checkpoints/step_N`, `cosmos/<ts>/safetensors/step_N` | cosmos-rl only: checkpoint and AlpaSim-eval export |
| `checkpoints/{model,optim,trainer}.pt` | posttrain only: written by `LLMTrainer.save` at the end of the run |
| `resolved_config.yaml` | The frozen config the run actually used |

### cosmos-rl mode (`main`)

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

### posttrain mode (`posttrain-migration`)

A healthy run reaches these in order:

```
All addresses open.                                  # 9 sim services up
AlpaSim RuntimeService is ready at <host>:<port>     # 10th (runtime) up
[alpagym] lr=1e-06, 20-step linear warmup          # LR schedule, logged once at build
[alpagym] scored episodes=6 ticks=132 aborted=0 ticks_per_episode=[22]
step    1/5  weight/version +1.0000  ratio/min +0.9999  clip/fraction +0.0000
[alpagym] saved checkpoint to .../checkpoints
```

The per-step line comes from posttrain's `ConsoleReporter`, so it is the same format
every posttrain entry prints: `step done/total` then the headline scalars, each rendered
`+%.4f`. `optim/lr` is deliberately not among them — at 1e-6 it would read `+0.0000`
every step, which is why the schedule is logged once at startup instead.

**`saved checkpoint` is the success marker.** What follows it looks alarming but is not:

```
srun: forcing job termination
srun: error: pool0-XXXXX: task 0: Killed
srun: Terminating StepId=<jobid>.N
```

That is AlpaGym tearing down the Wizard's srun step. Judge the run by `saved checkpoint`
and the files on disk, not by an exit code.

### The three numbers that say a run is healthy (posttrain)

| Line | Healthy | What a bad value means |
|---|---|---|
| `scored episodes=P ticks=T aborted=0` | `T == P × 22` | `aborted>0` or a short `ticks_per_episode` means episodes ended early — the buffer drops aborted trajectories, so the step trains on less than it reports |
| `ratio/min` per step | within ~1e-3 of 1.0 | The trainer rescores the rollout's own action with the same weights, so it must be ~1. **Drifting away over steps means the rollout is not getting the trainer's weights**; a sudden collapse means it is getting the wrong ones |
| `clip/fraction` | 0.0 early on | 1.0 means every sample hit the PPO clamp — usually the same weight-plane fault as above, not a hyperparameter to tune |

`ratio/min` is the one to watch. Two separate weight-sync bugs were invisible in every
other signal — the loss, the gradient norm, the version stamps and the checkpoint all
looked normal while the rollout sampled from a policy the trainer had left behind.

### Startup self-check (posttrain)

The loop syncs once before training and the rollout verifies that transfer is identity
(both sides still hold the same checkpoint). A failure raises immediately:

```
RuntimeError: weight sync is not identity at startup: <tensor> differs by <n>
  before any training, when both sides still hold the same checkpoint
```

That is never a flake. It means the weight plane is broken, and the named tensor is
where to start.

On the smoke config an untuned policy scores around -9.4, which under
`progress_safety` is roughly one collision (-10) plus a little progress. That is
expected, not a bug.

## Running cosmos-rl from `main`

`main` predates every site adaptation in this document, so a cosmos-rl run here needs
three things carried over from `posttrain-migration`. Two are files; the third is one
hunk, not a whole file.

| Carry over | Why | How |
|---|---|---|
| `packages/alpasim_configs/.../deploy/cw_dfw_slurm.yaml` | This cluster has no Docker, so `alpasim-base` cannot be built and the stock deploy presets do not apply. Without it AlpaSim does not come up at all. | `git checkout posttrain-migration -- <path>` |
| `scripts/cw-dfw/run_alpagym_clrl.sh` | Composes every site override (paths, account, container image, redis mounts, GPU split). Drop the `posttrain_repo_root` line — `main` has no such config key. | same |
| The Wizard GPU-visibility hunk in `packages/host/src/alpagym_host/slurm.py` | Without it the stock `alpagym_4gpu` topology fails the Wizard's id validation and the run never starts. | Copy the `--gpus-per-task` line and its comment ONLY. Do **not** take the whole file: it also carries the `export PYTHONPATH` that the posttrain entry needs and cosmos-rl does not. |

One more hunk in `slurm.py` is **optional**: `_cosmos_launcher_script` now assigns
`PYTHONPATH` (or clears it when no root is configured). The clearing half fixes the
`PolicyStatusManager has no attribute` failure in
[Troubleshooting](#troubleshooting) at its source, rather than by editing `~/.bashrc`.
Take it or use the `~/.bashrc` workaround; do not do both halves without the config key,
since `posttrain_repo_root` does not exist on `main`.

Everything else on this branch is posttrain-only and must NOT be carried over:
`run_lifecycle.py` (it replaces the cosmos-rl launcher outright and refuses multi-node),
`config.py`'s `posttrain_repo_root`, and `packages/runtime/.../posttrain/`.

`ALPAGYM_RUNBOOK.md` itself is also only on this branch. Read it from here.

## Running the posttrain path

The training half is `projects.cosmos3.posttrain.entrypoints.alpagym_clrl`, not
cosmos-rl. Two consequences for a fresh setup:

**imaginaire4 must be checked out beside alpagym**, at `~/work/imaginaire4`, on the
branch carrying the AlpaGym entry. `run_alpagym_clrl.sh` passes it as
`execution.slurm.posttrain_repo_root`, and `_cosmos_launcher_script` turns that into an
`export PYTHONPATH=<root>` **inside** the step's script — an assignment, never an
append, because the step runs under `bash -lc` and a login shell would otherwise win.

**posttrain is imported off Lustre, not staged by Ray.** See the trap in Troubleshooting:
editing it during a run breaks that run.

Step count is a Hydra override; the launch script pins it to 1 for a smoke run:

```bash
sed 's/max_num_steps=1/max_num_steps=5/' ~/work/run_alpagym_clrl.sh > ~/work/run5.sh
srun --overlap --jobid <JOBID> bash -c 'bash ~/work/run5.sh run'
```

Episodes per step come from `cosmos.train.train_batch_per_replica` divided by
`cosmos.rollout.n_generation` (prompts) times `n_generation` (group) — override both to
shrink a debug run: `train_batch_per_replica=2 n_generation=2` gives 2 episodes instead
of 6, which is 44 micro-batches instead of 132.

## Troubleshooting

Every failure hit while bringing this up, in the order they appear.

| Symptom | Cause | Fix |
|---|---|---|
| `no viable alternative at input '[UV_CACHE_DIR='` | Hydra's override grammar cannot parse `=` inside a list | Quote each element: `export_env=["K=V","K2=V2"]` |
| `Found no NVIDIA driver` while syncing `grouped-gemm` | `--all-packages` on the login node builds CUDA extensions | Host venv gets `--package alpagym-host` only; reach policy configs via `hydra.searchpath` |
| `Could not override 'experiment'` | The policy package is not installed, so its config group is invisible | `hydra.searchpath=[file://.../alpamayo_r1/.../configs]` |
| `no token was found` (HuggingFace) | Non-interactive shells skip `~/.bashrc` | Script pulls just the `HF_TOKEN` line out of it |
| `git-lfs: command not found` | Compute nodes lack git-lfs; AlpaSim is an LFS repo | Staged binary on `/lustre`, prepended to `PATH` |
| `requested GPUs [4,5,6,7] but only 0..3 are available` | The Wizard step is masked to AlpaSim's half, so its id validation runs in a renumbered frame while service placement does not | posttrain: already fixed (Wizard sees every GPU). cosmos-rl: cherry-pick that `slurm.py` change — renumbering the topology to 0-3 passes the check but puts the simulator on the trainer's GPUs |
| `ENROOT_CONFIG_PATH is not set` | `/etc/enroot/enroot.conf` sets enroot's own default, not the env var | `export ENROOT_CONFIG_PATH=$HOME/.config/enroot` |
| `enroot import ... docker://alpasim-base:0.91.0` fails | Image has no registry and cannot be built without Docker | Symlink a stock `.sqsh` into the squash cache |
| `Failed to spawn: physics_server` / `No module named 'alpasim_controller'` | Stock image has no `/repo`, so `uv run` falls back to a bare interpreter | `deploy=cw_dfw_slurm` mounts the checkout at `/repo`, sets `workdir` and `UV_NO_SYNC=1` |
| `No module named 'alpasim_controller.mpc_impl'` | Non-editable install drops subpackages | Restore them from the source tree (see Cold Start) |
| `redis-server: command not found` | Cosmos-RL starts Redis; the image has none | Bind-mount the binary and its two libs |
| `'PolicyStatusManager' has no attribute '_publish_payload_transport_cleanup'` | A `PYTHONPATH` in `~/.bashrc` shadows the pinned cosmos-rl with a local checkout | Remove it from `~/.bashrc` (backup: `~/.bashrc.bak.alpagym`) |
| Host venv suddenly broken, `bin/python` a dangling symlink | `UV_PROJECT_ENVIRONMENT` leaked through `--export=ALL`; the container's `uv sync` overwrote it | Never export it; invoke `.venv-host/bin/python` directly |
| Hang at first weight sync, both GPUs at 100% | NCCL advertises a P2P transport that stalls | Prefix with `NCCL_P2P_DISABLE=1` (not needed on this cluster) |
| A test fails on a name that no longer exists in your local tree | `rsync -az` decides by size+mtime, so a file whose size and timestamp both happen to match is skipped SILENTLY — you then "verify" against code you never sent. Here a renamed test kept the old spelling on the cluster while `slurm.py` beside it was current | Sync with `-c`: `rsync -az --checksum ...`. `rsync -avz --dry-run --checksum` lists exactly what drifted; ignore the trailing-slash directory lines, they are metadata only |
| Holder dies in seconds: `pyxis: failed to create container filesystem`, `dir_scan: failed to make directory /raid/enroot/data/user-<uid>/pyxis_<jobid>_liangf_dev/... File exists` | `--container-name liangf_dev` reuses a named enroot tree, and a node carrying a half-extracted one from an earlier job cannot re-extract into it. The node is the variable, so the same command works or fails depending on where it lands | Drop `--container-name` from the HOLDER — it only has to hold the allocation, and every step already carries its own `--container-image`/`--container-mounts`. Read `$B/slurm-node.log`: `srun` itself reports only `task 0: Exited with exit code 1` |

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

**Name the holder `sleep`, or the rule above stops working.** `~/run.sh` is interactive
(`--pty bash -i`) and dies without a tty, so a non-interactive allocation needs its own holder.
Start it as `srun ... sleep 14400`, NOT `srun ... bash -c 'sleep 14400'` — the latter registers
under `%j` = `bash`, so a cleanup filter that spares `sleep` kills the allocation instead. Spare
`.0` and `.extern` by step number as well as by name.

**Cleanup must include the cosmos step, not just `alpasim-*`.** A failed run's Ray actors keep
their GPU memory: after several aborted runs, GPU 0 held ~20 GiB across six leftover PIDs while
`squeue` showed nothing obviously wrong. That memory then shows up inside the *next* run's OOM
report as unattributed processes, which reads like the run's own footprint and sends you looking
for a leak that isn't there. Confirm `nvidia-smi` reads ~0 MiB before trusting any memory
measurement.

## Upstream Bugs Worth Reporting

**Editing imaginaire4 kills a running job.** Ray stages the *alpagym* working directory, so edits
there only affect the next run — but posttrain is imported straight off Lustre via `PYTHONPATH`,
and actors read those files as they go. A mid-flight edit surfaces as an `ImportError` or
`IndentationError` from a path under `imaginaire4/`, not as anything resembling a training fault.
Sync posttrain changes between runs, never during one.

**`srun` inside a `bash -s` heredoc eats the rest of the script.** It reads stdin, so every command
after it silently never runs — the symptom is a truncated report, not an error. Give it
`< /dev/null`.

**AlpaSim — the Slurm deployment validates GPU ids in one frame and places services in
another.** `services.py:283` checks every `gpus:` id against `context.get_num_gpus()`, which is
`nvidia-smi --query-gpu=count` *inside the Wizard process* — so a Wizard step bound to half the
node validates against `0..3`. Placement does not share that frame: the per-service
`srun --overlap` the Wizard issues carries no GPU flag at all, inherits the whole allocation, and
selects a device with `CUDA_VISIBLE_DEVICES=<id>`, an absolute index.

Whether this bites depends on the site. `detect_gpus` shells out to `nvidia-smi`, which ignores
`CUDA_VISIBLE_DEVICES` — so where Slurm does not constrain devices at the cgroup level it reports
the whole node and the stock `gpus: [4,5,6,7]` validates fine. cw-dfw *does* constrain them
(measured: `--gpu-bind=mask_gpu:0xf0` → `nvidia-smi --query-gpu=count` reports `4`, unbound
reports `8`), so the two frames invert here: the ids that pass validation (`0..3`) are exactly
the ones that land on the trainer's GPUs, and the ids that land correctly (`4..7`) are rejected
before the run starts. The stock config has therefore never been runnable on this cluster.

`build_wizard_srun_command` gives the Wizard step every GPU on the host so the frames agree,
which reproduces the behaviour sites without device constraints get for free. The upstream fix
is for the Wizard's per-service `srun` to carry the binding, or for `detect_gpus` to report the
allocation rather than the calling process's view.

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
