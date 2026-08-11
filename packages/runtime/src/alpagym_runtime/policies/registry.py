# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry-point registry for policy-owned runtime hooks."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Callable

import torch
from alpagym_host.config import RunConfig
from alpagym_plugins.plugins import PluginRegistry

from alpagym_runtime.inference.types import InferenceModel
from alpagym_runtime.replay import PolicyReplayData

if TYPE_CHECKING:
    # `build_data_packer` is the cosmos-rl path's hook and this name is only ever a type here.
    # Importing it eagerly runs `alpagym_runtime.cosmos.packer`, which imports `cosmos_rl` and
    # loads ~190 of its modules into every process that touches the policy registry -- including
    # the posttrain rollout actor, which never calls this hook.
    from alpagym_runtime.cosmos.packer import AlpagymDataPacker


@dataclass(frozen=True)
class PolicyBundle:
    """Runtime hooks owned by one installed policy package.

    ``build_data_packer`` builds the policy-specific trainer-side replay packer
    for this process's Cosmos role.

    ``build_model_inputs`` returns the trainer-side callable that converts one
    replay payload into model-forward kwargs plus the rollout-time old logprob.
    Each policy owns its model input dialect; the runtime packer stays
    policy-agnostic by receiving that callable from the bundle.
    """

    setup_tokenizer: Callable[[Any], Any | None]
    build_data_packer: Callable[[RunConfig, str | None], AlpagymDataPacker]
    install_runtime_bridge: Callable[[], None]
    load_inference_model: Callable[[RunConfig, torch.device, torch.dtype], InferenceModel]
    build_model_inputs: Callable[
        [RunConfig],
        Callable[[PolicyReplayData], tuple[dict[str, Any], torch.Tensor]],
    ]

    def __post_init__(self) -> None:
        """Validate that every bundle hook is callable."""
        for field in fields(self):
            hook = getattr(self, field.name)
            if not callable(hook):
                raise TypeError(f"PolicyBundle hook {field.name!r} must be callable")


policy_bundles = PluginRegistry("alpagym.policy_bundles")


def get_policy_bundle(kind: str) -> PolicyBundle:
    """Return the installed ``PolicyBundle`` for ``kind``."""
    bundle_factory = policy_bundles.get(kind)
    bundle = bundle_factory()
    if not isinstance(bundle, PolicyBundle):
        raise TypeError(
            f"PolicyBundle entry point for {kind!r} returned object of type {type(bundle).__name__}"
        )
    return bundle
