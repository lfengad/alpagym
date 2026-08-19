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
PLACEMENT=${3:-colocated}  # cosmos.mode: colocated (IPC weight edge) | disaggregated (NIXL)

# NIXL setup, disaggregated only -- the colocated path never imports it.
#
#  - `nixl` needs BOTH the dispatch shim and the matching `nixl-cuXX` backend, installed with
#    --no-deps into their own directory: a plain install pulls a newer torch that SHADOWS the
#    venv's, which flash-attn and vllm were compiled against. Reproduce $NIXL_EXTRA with:
#      uv pip install --target $NIXL_EXTRA --no-deps --python-version 3.12 \
#        nixl nixl-cu12 cupy-cuda12x 'cuda-bindings<13'
#    --python-version is NOT optional: run from the login host without it, uv resolves for the
#    login host's interpreter and lays down cp313 extension modules that the container's 3.12
#    cannot load -- `cuda.bindings` then imports but has no `driver`. `cuda-bindings` is pinned
#    below 13 to stay on the container's CUDA 12 line, and it is needed because `_fabric_buffer`
#    allocates the staging buffer through the driver API.
#    It rides ALPAGYM_EXTRA_PYTHONPATH, not PYTHONPATH: the launcher ASSIGNS PYTHONPATH (so a
#    login shell cannot shadow the pinned cosmos-rl) and would clobber anything sent directly.
#  - The UCX in play is the one VENDORED IN THE WHEEL, and nothing can point NIXL at another.
#    auditwheel rewrote the plugin's dependencies to hash-renamed private copies -- `ldd` on
#    .nixl_cu12.mesonpy.libs/plugins/libplugin_UCX.so resolves libucp-5c599099.so.0.0.0 out of
#    nixl_cu12.libs by RPATH -- so LD_LIBRARY_PATH cannot redirect it. That build is stripped:
#    it reports only mm, posix, self, shm, sm, sysv and tcp. Naming rc_mlx5, dc_mlx5, cuda_copy
#    or cuda_ipc in UCX_TLS just earns "transports ... are not available".
#    CONSEQUENCE, and the reason this is a stopgap: with no cuda_copy/cuda_ipc, UCX warns "GPU
#    memory is not supported", so weight sync crosses host memory instead of going GPU to GPU.
#    Correctness first; a wheel built against the container's UCX is what makes it fast.
#  - STATUS: disaggregated still does NOT complete a step, but the failure is now HALF
#    understood and there is a 90-second reproducer for the understood half.
#    FOUND: Ray only creates the `_ray_system` background thread -- the one RDT runs transfers
#    on -- when the actor asks for it with `enable_tensor_transport=True` (ray/actor.py, and see
#    the flag now set on posttrain's `_Host`). Ray also infers it from a method decorated
#    `@ray.method(tensor_transport=...)`, which posttrain has none of: it stages through
#    `ray.put(_tensor_transport="nixl")` inside the actor, which Ray cannot see. Without the flag
#    a publication is accepted and then never moves, and NOTHING logs an error, because the same
#    flag creates `_ray_system_error` and Ray's error path check-fails without it. Reproduce in
#    ~90s: two `@ray.remote(num_gpus=1)` actors, the producer returning a CUDA tensor from a
#    method decorated `tensor_transport="nixl"`. Without the flag the consumer times out with
#    "not found in RDT object store"; with it the tensor arrives.
#    STILL OPEN: the real run hangs at the first weight sync even with the flag set, so
#    something else is also wrong. Ruled out, each measured: missing CUDA transports in the
#    wheel (libuct_cuda.so and libucm_cuda.so ARE shipped and their deps resolve); UCX not
#    finding them (ai-dynamo/nixl#1628's UCX_MODULE_DIR workaround, set below, does not lift it);
#    the wheel version (1.3.2 same, 1.0.0 fails earlier at agent creation); the device list
#    (unset / lo / all / lo,cuda all identical); publishing overlapping views of one buffer (a
#    fresh tensor per call fails the same way).
#    Note for whoever picks this up: getting Ray's worker logs off this node defeated four
#    attempts -- ray.init uses log_to_driver=False, the container's /tmp dies with the container,
#    Ray re-points worker fd 2 after sitecustomize runs, and neither _temp_dir nor RAY_TMPDIR
#    moved the session dir. Solve that first; everything above was inferred without it.
#  - UCX_NET_DEVICES=lo, and this is the one that actually decides whether the run starts.
#    Everything here is single-node, so the agent's intra-agent wireup connects to ITSELF. The
#    shared-memory transports are all rejected for it ("no peer failure handler"), leaving tcp,
#    and tcp only works over an address the process can route back to. Every IB-over-Ethernet
#    device on this node (ibp26s0 et al) and the Ethernet port (enp90s0np0) advertise addresses
#    that are NOT locally routable -- "no route to 100.126.37.129:65535" -- and the backend then
#    fails to create with a bare NIXL_ERR_BACKEND. Loopback always routes.
#    This only bites once torch.distributed/NCCL is up, which is why it looks like a trainer-only
#    fault: a fresh process picks a workable device on its own.
NIXL_EXTRA=${NIXL_EXTRA:-$BASE/nixl_extra}
SCRIPT_ENV=""
if [ "$PLACEMENT" = "disaggregated" ]; then
  SCRIPT_ENV="\"ALPAGYM_EXTRA_PYTHONPATH=$NIXL_EXTRA\""
  SCRIPT_ENV="$SCRIPT_ENV,\"UCX_TLS=cuda_copy,cuda_ipc,sm,tcp,self\""
  SCRIPT_ENV="$SCRIPT_ENV,\"UCX_MODULE_DIR=$NIXL_EXTRA/nixl_cu12.libs/ucx\""
  SCRIPT_ENV="$SCRIPT_ENV,\"UCX_NET_DEVICES=lo\""
  SCRIPT_ENV="$SCRIPT_ENV,\"UCX_LOG_LEVEL=error\""
  [ -d "$NIXL_EXTRA/nixl" ] || { echo "NIXL_EXTRA=$NIXL_EXTRA has no nixl/ (see the pip line above)" >&2; exit 1; }
fi

HYDRA_ARGS=(
  "hydra.searchpath=[file://$POLICY_CONFIGS]"
  topology=slurm_full_node_1_3_4
  alpasim.wizard_args.topology=alpagym_4gpu
  alpasim.wizard_args.deploy=cw_dfw_slurm
  experiment=alpamayo_1_5_clrl_test_run
  policy.model.path="$BASE/alpagym/tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt"

  # Step count comes from $2; the default of 1 just proves the loop closes.
  cosmos.train.max_num_steps=$STEPS
  # colocated: every GPU carries an FSDP shard AND an engine. disaggregated: trainer and rollout
  # split the GPUs in half, the layout cosmos-rl ran -- reaches the entry as `--placement`.
  # NOT `cosmos.mode`: the host pins that to `disaggregated` for every Slurm run and the topology
  # preset pairs it with `transport: nccl`, both of which describe cosmos-rl's process split
  # rather than posttrain's GPU layout.
  cosmos.placement=$PLACEMENT
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
  "execution.slurm.script_env=[$SCRIPT_ENV]"
)

HOST_PYTHON=$BASE/alpagym/.venv-host/bin/python

if [ "$MODE" = "cfg" ]; then
  exec "$HOST_PYTHON" -m alpagym_host.cli --cfg job "${HYDRA_ARGS[@]}"
else
  exec "$HOST_PYTHON" -m alpagym_host.cli "${HYDRA_ARGS[@]}"
fi
