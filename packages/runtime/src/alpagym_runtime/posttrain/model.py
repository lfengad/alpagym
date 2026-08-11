# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Alpamayo training model, built the way posttrain builds an ``HFModel``.

Structure mirrors ``projects/cosmos3/posttrain/model/hf.py``: ``__init__`` allocates NO real
memory (the model is constructed on the ``meta`` device), and :meth:`setup` runs the load-bearing
lifecycle in the required order -- **parallelize -> materialize -> load**. Wrapping the model this
way is what lets posttrain's ``LLMTrainer`` drive it unchanged.

WHY THIS EXISTS. The rollout's model and the trainer's model are different classes:
``build_inference_engine`` yields ``ExpertModelRL`` (generation), while training needs
``ExpertModelCosmos``, whose ``forward`` the policy bundle patches into AlpaGym's replay contract
``(image_frames, ...) -> {"log_probs", "kl_div"}``. Under cosmos-rl that construction was implicit:
``ModelRegistry.build_model`` built it and ``GRPOTrainer.__init__`` drove the FSDP + weight-load
sequence. Neither exists here, so the sequence is written out.

NO cosmos-rl. ``ExpertModelCosmos`` exposes a cosmos-flavoured ``parallelize_fn`` that wants
cosmos-rl's ``ParallelDims`` and ``Config``, but every piece it ultimately needs is plain torch:

* ``_apply_fsdp2(dp_mesh, fsdp_config, reshard_fn)`` takes a ``DeviceMesh``, a dict and a callable.
* ``build_fsdp_config`` only assembles ``{"mesh", "mp_policy"}`` from two dtype fields, so the dict
  is built here directly instead.
* ``post_to_empty_hook`` accepts ``None`` -- it falls back to ``hf_config._name_or_path`` by its own
  code path, not by accident.
* ``load_hf_weights`` branches on ``detect_fsdp2_active(self.expert_model)``, reading the sharding
  off the model itself; its ``parallel_dims`` argument is optional and unused for that decision.

The LM is frozen by ``ExpertModelCosmos`` itself (the baseline logs ``Froze 398 LM params``); the
run config leaves ``freeze_pattern``/``trainable_pattern``/``trainable_map`` all ``None``, so no
trainable plan is applied here.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AlpamayoTrainModel(nn.Module):
    """Meta-init the Alpamayo expert model; shard, materialize and load in :meth:`setup`."""

    def __init__(self, model_name_or_path: str, dtype: torch.dtype = torch.bfloat16) -> None:
        """Construct on ``meta`` -- no parameter memory until :meth:`setup`.

        Args:
            model_name_or_path: the AlpaGym checkpoint directory (``policy.model.path``).
            dtype: compute dtype for the FSDP mixed-precision policy.
        """
        super().__init__()
        from accelerate import init_on_device
        from alpamayo1_x_rl.models.expert_model.cosmos_wrapper import ExpertModelCosmos
        from transformers import AutoConfig

        self.ckpt_path = model_name_or_path
        self.dtype = dtype
        self.hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        # Buffers are created real (include_buffers=False) so values like RoPE inv_freq keep their
        # init-time contents -- same reasoning as HFModel.
        with init_on_device("meta", include_buffers=False):
            self.model = ExpertModelCosmos(self.hf_config)
        logger.info("AlpamayoTrainModel: %s on meta, dtype=%s", self.hf_config.model_type, dtype)

    @property
    def net(self) -> nn.Module:
        """The trainable module -- the INNER `expert_model`, not the wrapper.

        `LLMTrainer` reaches parameters through `getattr(model, "net", model)`, and those names are
        what the weight-sync bucket plan is keyed on. The rollout's model is the inner
        `ExpertModelRL` (`InferenceEngine.get_model()`), so its keys read
        `action_in_proj.encoder...`; returning the wrapper here would key the plan
        `expert_model.action_in_proj.encoder...` and `_strict_plan_coverage` rejects the pair as
        divergent state dicts. Aligning the namespaces at this seam is cheaper -- and harder to get
        subtly wrong -- than carrying a rename map like cosmos-rl's
        `weight_mapper.rollout_map_local_key_to_hf_key`.

        The forward path is unaffected: `forward` still calls the wrapper, which is where the
        policy bundle's replay-contract patch lives.
        """
        return self.model.expert_model

    def setup(self, world_size: int) -> None:
        """parallelize -> materialize -> load, in that order.

        The order is load-bearing and mirrors ``HFModel.setup``: FSDP2 wrap first so parameters
        become DTensors, materialize the meta storage next, and only then stripe the checkpoint into
        each rank's shard. Loading before the wrap would fill unsharded storage and then throw it
        away.

        Args:
            world_size: the FSDP shard count -- this replica's rank-mesh width.
        """
        from alpamayo1_x_rl.utils.fsdp import build_reshard_fn
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy

        # "cuda", never f"cuda:{rank}": Ray scopes CUDA_VISIBLE_DEVICES per actor, so every actor
        # sees exactly one device numbered 0. Indexing by the distributed rank asks for an ordinal
        # that does not exist in this process. `HFModel._materialize_meta` takes the same plain
        # "cuda" for the same reason.
        device = torch.device("cuda")

        if world_size > 1:
            dp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard_cp",))
            # What `alpamayo1_x_rl.utils.fsdp.build_fsdp_config` assembles, minus its cosmos-rl
            # config plumbing. reduce_dtype stays fp32: the run config's `fsdp_reduce_dtype` is
            # float32, and reducing gradients in bf16 would change the update, not just its speed.
            fsdp_config = {
                "mesh": dp_mesh,
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=self.dtype,
                    reduce_dtype=torch.float32,
                    cast_forward_inputs=False,
                ),
            }
            self.model._apply_fsdp2(dp_mesh, fsdp_config, build_reshard_fn("default"))

        # Materialize like `HFModel._materialize_meta`, NOT with `to_empty`. `__init__` builds
        # buffers REAL (`include_buffers=False`), so tensors such as RoPE `inv_freq` already hold
        # their init-time values -- and `to_empty` would replace parameters AND buffers alike with
        # uninitialized storage. The checkpoint then refills the parameters but not the buffers,
        # which are computed rather than stored, leaving positional encodings as garbage: the model
        # still runs, still produces finite log-probs, and is simply wrong. Empty only what is on
        # meta; move everything else.
        self.model._apply(
            lambda t: torch.empty_like(t, device=device) if t.device.type == "meta" else t.to(device),
            recurse=True,
        )
        # None is a supported argument: the hook falls back to `hf_config._name_or_path`. It only
        # warms the HF auto-class registry.
        self.model.post_to_empty_hook(None)
        self.model.load_hf_weights(self.ckpt_path, device=device)

        # Gradient checkpointing is NOT optional here. The cosmos-rl baseline ran with
        # `model_gradient_checkpointing: True`, and without it a single trajectory's 22 ticks of
        # 4-camera activations come to ~28 GiB on top of the 13.65 GiB shard -- measured, as an OOM
        # inside the expert MLP. AlpaGym's own config does not surface the flag, so the baseline's
        # value is carried explicitly rather than inherited from whatever the model defaults to.
        # `ExpertModelCosmos.set_gradient_checkpointing_enabled`, NOT HF's
        # `gradient_checkpointing_enable`: the inner `ExpertModelRL` rejects the HF call outright
        # ("does not support gradient checkpointing"), and the wrapper is where the switch lives.
        enable = getattr(self.model, "set_gradient_checkpointing_enabled", None)
        if enable is None:
            raise RuntimeError(
                f"{type(self.model).__name__} exposes no set_gradient_checkpointing_enabled; "
                "without it a 22-tick trajectory does not fit beside the colocated rollout"
            )
        enable(True)
        logger.info(
            "AlpamayoTrainModel: weights loaded, grad-ckpt on (fsdp world_size=%d)", world_size
        )

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Pass through to the patched `ExpertModelCosmos.forward`.

        The policy bundle's `install_runtime_bridge` patch is what makes this return
        ``{"log_probs": Tensor[R], "kl_div": Tensor[R] | None}`` from raw `image_frames` -- the
        contract `alpagym_runtime.posttrain.objective` scores against.
        """
        return self.model(**kwargs)
