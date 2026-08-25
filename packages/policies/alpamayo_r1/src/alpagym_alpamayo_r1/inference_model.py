# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 1 / 1.5 inference model."""

import logging
from dataclasses import asdict
from typing import Any, Mapping

import torch
from alpagym_host.config import SamplingParamsConfig
from alpagym_runtime.inference.types import (
    BatchedModelInput,
    BatchedModelOutput,
    ModelInput,
    ModelOutput,
    drop_single_batch_axis,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData, require_payload_keys

from alpagym_alpamayo_r1.tokenize_online import tokenize_for_generation

logger = logging.getLogger(__name__)


class AlpamayoR1InferenceModel:
    """Adapter between alpagym typed I/O and an Alpamayo 1 / 1.5 model."""

    def __init__(
        self,
        model: Any,
        num_context_frames: int,
        last_component: str = "traj_future",
    ) -> None:
        """Bind the loaded ``ExpertModelRL`` and the ``last_component`` knob.

        ``last_component`` is Python-API-only today: no shipped alpagym
        config (yaml / ``SamplingParamsConfig`` / factory) exposes it,
        and ``load_inference_model`` in ``bundle.py`` always uses the default
        ``'traj_future'``. The ``'cot'`` branch + multi-sample guard
        stay so a future MR that wires CoT inference through the
        config does not have to re-derive the contract.
        """
        if last_component not in ("traj_future", "cot"):
            raise ValueError(
                "AlpamayoR1InferenceModel: last_component must be 'traj_future' or "
                f"'cot' (got {last_component!r})."
            )
        self._model = model
        self._num_context_frames = num_context_frames
        self._last_component = last_component

    def get_model(self) -> torch.nn.Module:
        """Return the Alpamayo R1 / 1.5 model instance that serves rollout inference."""
        return self._model

    def set_model(self, model: torch.nn.Module) -> None:
        """Replace the inference model, unwrapping FSDP/cosmos shells.

        Cosmos colocated weight sync hands the rollout the FSDP-wrapped
        ``ExpertModelCosmos``. The rollout adapter calls inner ``ExpertModel``
        helpers like ``_get_generation_mode_tokenized_data_online`` directly,
        bypassing the wrapper's ``forward`` (and thus FSDP's pre_forward
        hook). Two steps prepare the wrapper for that direct access:
        ``_lazy_init`` on the root so an inner pre_forward doesn't claim
        root first, and ``unshard()`` on every descendant so weights are
        plain Tensors rather than DTensors (a no-op for memory when
        ``dp_shard_size=1``).
        """
        if hasattr(model, "_get_fsdp_state"):
            from torch.distributed.fsdp import FSDPModule

            model._get_fsdp_state()._lazy_init()
            for submodule in model.modules():
                if isinstance(submodule, FSDPModule):
                    submodule.unshard()
        if hasattr(model, "expert_model"):
            self._model = model.expert_model
        else:
            self._model = model

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Pack, dispatch, and normalize one Alpamayo forward pass."""
        if (
            self._last_component == "cot"
            and sampling.num_traj_sets * sampling.num_traj_samples != 1
        ):
            # ``vlm_generated_ids_list`` has one entry per rollout but the
            # replay path consumes the singular ``vlm_generated_ids`` only;
            # without a select_ix plumb-through, multi-sample CoT would drift
            # to a fresh prefill silently.
            raise ValueError(
                "Alpamayo R1 last_component='cot' requires num_traj_sets=1 "
                "and num_traj_samples=1; got "
                f"num_traj_sets={sampling.num_traj_sets}, "
                f"num_traj_samples={sampling.num_traj_samples}."
            )

        batch_size = model_input.camera_frames.shape[0]
        data = build_alpamayo_r1_forward_inputs(
            model_input,
            self._num_context_frames,
        )
        self._transform_inputs(data)
        # ``_get_generation_mode_tokenized_data_online`` does ``(x + 1) / 2``
        # in place on ``data["image_frames"]`` when
        # ``legacy_inference_image_input_format`` is True. Clone first so the
        # sampler still sees the pre-rescale [-1, 1] view; double-rescaling
        # silently corrupts pixel values.
        helper_data = dict(data)
        helper_data["image_frames"] = data["image_frames"].clone()
        prompt_td = tokenize_for_generation(
            self._model, helper_data, last_component=self._last_component
        )
        data_for_sampling = dict(data)
        data_for_sampling["tokenized_data"] = dict(prompt_td)

        # ``traj_future`` prompts already end at ``<|traj_future_start|>`` and
        # skip the autoregressive VLM generate step (~36 padding tokens, ~2x
        # latency).
        with_vlm_rollout = self._last_component != "traj_future"

        diffusion_kwargs: dict[str, Any] = {
            key: value
            for key, value in {
                "int_method": sampling.diffusion_kwargs.int_method,
                "noise_level": sampling.diffusion_kwargs.noise_level,
                "inference_step": sampling.diffusion_kwargs.inference_step,
                "temperature": sampling.diffusion_kwargs.temperature,
            }.items()
            if value is not None
        }
        # The 4-way unpack below requires ExpertModelRL's SDE branch, which
        # only emits the info dict when ``return_info=True``. Not exposed in
        # the schema because the unpack depends on it.
        diffusion_kwargs["return_info"] = True
        if sampling.force_determinism:
            # One generator per row, repeated across the row's candidates, keeps
            # the single batched forward reproducible from each row's seed.
            generators: list[torch.Generator] = []
            for seed in model_input.seed.tolist():
                generator = torch.Generator(device=model_input.ego_history_xyz.device).manual_seed(
                    int(seed)
                )
                generators.extend(
                    generator for _ in range(sampling.num_traj_sets * sampling.num_traj_samples)
                )
            diffusion_kwargs["generator"] = generators

        kwargs: dict[str, Any] = {
            "with_vlm_rollout": with_vlm_rollout,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "temperature": sampling.temperature,
            "num_traj_samples": sampling.num_traj_samples,
            "num_traj_sets": sampling.num_traj_sets,
            "last_component": self._last_component,
            "diffusion_kwargs": diffusion_kwargs,
            "return_extra": return_trace_for_rl,
        }
        if sampling.max_generation_length is not None:
            kwargs["max_generation_length"] = sampling.max_generation_length

        with torch.inference_mode():
            raw = self._model.sample_trajectories_from_data(data_for_sampling, **kwargs)

        # The expert's VLM prefill can intermittently emit an all-NaN trajectory
        # (scene-state dependent, not a fixed input). Re-running the prefill lands
        # finite, so re-tokenize and re-sample rather than propagate the NaN.
        if sampling.retry_on_nonfinite:
            # Eight, not three. A non-finite prefill is common enough here to reach the LAST retry
            # on its own -- measured at roughly one non-finite sample per episode, with the third
            # retry reached twice in a 120-episode run -- and exhausting the budget aborts the
            # whole run, not the tick. Under three, two runs died mid-experiment; the runs that
            # survived did so on the extra sample the `for/else` structure happens to take after
            # the loop, which is luck rather than budget. At a ~5% per-sample rate eight
            # consecutive failures is ~1e-10, and the cost is a few extra forwards on the rare
            # tick that needs them.
            attempts = 8
            for attempt in range(1, attempts + 1):
                pred_xyz, pred_rot = _unpack_sde_tuple(raw)[:2]
                if torch.isfinite(pred_xyz).all() and torch.isfinite(pred_rot).all():
                    break
                logger.warning(
                    "Alpamayo R1 emitted a non-finite trajectory; re-running inference (%d/%d).",
                    attempt,
                    attempts,
                )
                helper_data = dict(data)
                helper_data["image_frames"] = data["image_frames"].clone()
                data_for_sampling = dict(data)
                data_for_sampling["tokenized_data"] = dict(
                    tokenize_for_generation(
                        self._model, helper_data, last_component=self._last_component
                    )
                )
                with torch.inference_mode():
                    raw = self._model.sample_trajectories_from_data(data_for_sampling, **kwargs)
            else:
                # Loop never broke: the final re-sample was not checked at the top
                # of the loop, so validate it and fail rather than emit NaN.
                pred_xyz, pred_rot = _unpack_sde_tuple(raw)[:2]
                if not (torch.isfinite(pred_xyz).all() and torch.isfinite(pred_rot).all()):
                    raise ValueError(
                        f"Alpamayo R1 emitted a non-finite trajectory after {attempts} retries."
                    )

        # Normalize ExpertModelRL's SDE tuple into the batched output. ``pred_xyz``
        # stays ``(B, num_traj_sets, num_traj_samples, T, 3)``, ``pred_rot``
        # ``(B, num_traj_sets, num_traj_samples, T, 3, 3)``, ``logprob``
        # ``(B, num_traj_sets, num_traj_samples)``. ``BatchedModelOutput.unbind()``
        # strips ``B``; the selector then indexes ``pred_xyz[set_ix, sample_ix]`` and
        # ``policy._postprocess`` reads ``shape[2]`` as the trajectory horizon.
        pred_xyz, pred_rot, logprob, sde_info = _unpack_sde_tuple(raw)
        if logprob.dim() != 3:
            raise ValueError(
                f"AlpamayoR1InferenceModel: SDE logprob expected rank 3 "
                f"(B, num_traj_sets, num_traj_samples); got {tuple(logprob.shape)}"
            )

        # Gate ``logprob`` / ``extra`` on the trace flag so a trace-disabled run does
        # not silently produce ``chosen_logprob`` / ``replay_data`` in
        # ``AlpamayoPolicy._postprocess``.
        logprob_out = logprob if return_trace_for_rl else None
        extra: dict[str, Any] = {}
        if return_trace_for_rl:
            extra = _pack_replay_extra(
                sde_info=sde_info,
                batch_size=batch_size,
                num_traj_sets=sampling.num_traj_sets,
                num_traj_samples=sampling.num_traj_samples,
            )
        return BatchedModelOutput(
            pred_xyz=pred_xyz,
            pred_rot=pred_rot,
            logprob=logprob_out,
            extra=extra,
        )

    def _transform_inputs(self, data: dict[str, Any]) -> None:
        """Apply Alpamayo R1 input transformations in place.

        Single seam for adapter-side input transforms so the contract is
        easy to move (e.g. into the policy) if other models want a
        different contract. Today it just normalizes uint8 frames to
        float32 in [-1, 1] for the model's ``(x + 1) / 2`` rescale.
        """
        data["image_frames"] = data["image_frames"].to(torch.float32) / 127.5 - 1.0

    def build_policy_replay_data(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        action_selection: ActionSelection,
    ) -> PolicyReplayData:
        """Pack selected-only R1 replay data for trainer-side ``cfm_logprob_sde``.

        Persists the raw ``ModelInput`` (via ``asdict``) plus the SDE
        trace fields ``cfm_logprob_sde`` consumes
        (``samples_list`` / ``timesteps``, optional ``noise_level`` /
        ``vlm_generated_ids`` / ``cot``). Trainer-side
        ``build_trainer_model_inputs`` re-runs ``build_alpamayo_r1_forward_inputs``
        on the rehydrated input — the same seam the rollout used — so
        the forward kwargs are bitwise-identical without persisting the
        post-`build_alpamayo_r1_forward_inputs` view.
        """
        if model_output.logprob is None:
            raise ValueError("Alpamayo R1 replay requires rollout logprob")

        old_logprob = model_output.logprob[action_selection.set_ix, action_selection.sample_ix]

        # ``samples_list`` and ``timesteps`` arrive as the full multi-candidate
        # SDE trace from FlowMatching._sde, flattened in (set, sample) order
        # per the einops rearrange at ``model.py:531``. Persist only the row
        # the rollout actually executed so trainer-side ``cfm_logprob_sde``
        # returns one log_prob that pairs 1:1 with ``old_logprob``.
        num_traj_samples = model_output.logprob.shape[1]
        flat_idx = action_selection.set_ix * num_traj_samples + action_selection.sample_ix
        samples_list = model_output.extra["samples_list"][flat_idx]
        timesteps = model_output.extra["timesteps"]

        payload: dict[str, Any] = {
            "model_input": asdict(model_input),
            "samples_list": samples_list,
            "timesteps": timesteps,
        }
        for optional_key in ("noise_level", "vlm_generated_ids", "cot"):
            if optional_key in model_output.extra:
                payload[optional_key] = model_output.extra[optional_key]

        return PolicyReplayData(
            replay_schema_version=1,
            payload_schema="alpamayo_r1.trajectory.v1",
            payload_schema_version=1,
            model_family="alpamayo_r1",
            action_selection=action_selection,
            old_logprob=torch.as_tensor(old_logprob, dtype=torch.float32).reshape(()),
            payload=payload,
        )

    @classmethod
    def build_trainer_model_inputs(
        cls,
        replay_data: PolicyReplayData,
        num_context_frames: int,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Pack one R1 replay payload into ``ExpertModelCosmos.forward`` kwargs.

        Rehydrates the persisted ``ModelInput`` and re-runs
        ``build_alpamayo_r1_forward_inputs`` to recover the R1-shaped
        forward inputs (single seam shared with the rollout-side
        adapter), then appends the SDE trace + optional flags. The
        bundle-side patch on ``ExpertModelCosmos.forward`` reruns
        ``_get_generation_mode_tokenized_data_online`` on these raw
        inputs to rebuild Qwen's ``pixel_values`` / ``image_grid_thw``
        / ``input_ids`` bitwise-identically, which keeps the artifact
        ~25x smaller than persisting the post-Qwen view.
        """
        if replay_data.payload_schema != "alpamayo_r1.trajectory.v1":
            raise ValueError(
                f"{replay_data.model_family} replay payload_schema "
                f"{replay_data.payload_schema!r} != 'alpamayo_r1.trajectory.v1'"
            )
        if replay_data.payload_schema_version != 1:
            raise ValueError(
                f"{replay_data.model_family} replay payload_schema_version "
                f"{replay_data.payload_schema_version} != 1"
            )
        payload = replay_data.payload
        require_payload_keys(
            replay_data.model_family,
            payload,
            ("model_input", "samples_list", "timesteps"),
        )
        model_input_payload = payload["model_input"]
        if not isinstance(model_input_payload, Mapping):
            raise TypeError(
                "alpamayo_r1 replay model_input must be a mapping, "
                f"got {type(model_input_payload).__name__}"
            )

        forward_inputs = drop_single_batch_axis(
            build_alpamayo_r1_forward_inputs(
                BatchedModelInput.stack([ModelInput.from_payload(model_input_payload)]),
                num_context_frames,
            )
        )

        model_inputs: dict[str, Any] = {
            "image_frames": forward_inputs["image_frames"],
            "camera_indices": forward_inputs["camera_indices"],
            "ego_history_xyz": forward_inputs["ego_history_xyz"],
            "ego_history_rot": forward_inputs["ego_history_rot"],
            "samples_list": torch.as_tensor(payload["samples_list"], dtype=torch.float32),
            "timesteps": torch.as_tensor(payload["timesteps"], dtype=torch.float32),
        }
        if "noise_level" in payload:
            model_inputs["noise_level"] = torch.as_tensor(
                payload["noise_level"], dtype=torch.float32
            )
        if "vlm_generated_ids" in payload:
            model_inputs["vlm_generated_ids"] = torch.as_tensor(
                payload["vlm_generated_ids"], dtype=torch.int64
            )

        if replay_data.old_logprob is None:
            raise ValueError("alpamayo_r1 replay requires old_logprob")
        old_logprob = torch.as_tensor(replay_data.old_logprob, dtype=torch.float32).reshape(())
        return model_inputs, old_logprob


def _unpack_sde_tuple(
    raw: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Validate ExpertModelRL's SDE 4-tuple and pull out the four fields."""
    if not isinstance(raw, tuple):
        raise TypeError(
            "Alpamayo R1 sample_trajectories_from_data must return a tuple; "
            f"got {type(raw).__name__}"
        )
    if len(raw) != 4:
        raise ValueError(
            f"Alpamayo R1 SDE path returned a {len(raw)}-tuple; expected 4 "
            "(pred_xyz, pred_rot, logprob, sde_info). Make sure "
            "diffusion_kwargs['return_info']=True."
        )
    pred_xyz, pred_rot, logprob, sde_info = raw
    if not isinstance(logprob, torch.Tensor):
        raise TypeError(f"Alpamayo R1 4-tuple logprob must be Tensor; got {type(logprob).__name__}")
    if not isinstance(sde_info, dict):
        raise TypeError(f"Alpamayo R1 4-tuple sde_info must be dict; got {type(sde_info).__name__}")
    return pred_xyz, pred_rot, logprob, sde_info


def _pack_replay_extra(
    sde_info: dict[str, Any],
    batch_size: int,
    num_traj_sets: int,
    num_traj_samples: int,
) -> dict[str, Any]:
    """Package SDE replay metadata for the trainer-side ``cfm_logprob_sde``.

    The raw ``ModelInput`` is persisted separately by ``build_policy_replay_data``;
    this only forwards the SDE trace plus the optional flags that aren't part of the
    rollout input contract.

    ``BatchedModelOutput.unbind`` slices every ``extra`` value along a leading
    ``batch_size`` axis, so the SDE candidate traces are repacked from the model's
    flat ``[B * num_traj_sets * num_traj_samples, ...]`` to
    ``[B, num_traj_sets * num_traj_samples, ...]`` (the selected candidate can then
    still be flattened in rollout order after unbind). ``cot`` is the one exception:
    it is forwarded as-is, valid only under the single-candidate
    ``last_component='cot'`` path where it already carries the leading batch axis.
    """
    extra: dict[str, Any] = {}
    num_candidates = num_traj_sets * num_traj_samples
    samples_list = sde_info.get("samples_list")
    if samples_list is not None:
        samples_tensor = torch.as_tensor(samples_list, dtype=torch.float32)
        extra["samples_list"] = samples_tensor.reshape(
            batch_size,
            num_candidates,
            *samples_tensor.shape[1:],
        )
    timesteps = sde_info.get("timesteps")
    if timesteps is not None:
        timesteps_tensor = torch.as_tensor(timesteps, dtype=torch.float32)
        extra["timesteps"] = timesteps_tensor.unsqueeze(0).expand(batch_size, -1)
    if "cot" in sde_info:
        extra["cot"] = sde_info["cot"]
    if "noise_level" in sde_info:
        extra["noise_level"] = [sde_info["noise_level"] for _ in range(batch_size)]
    vlm_ids_list = sde_info.get("vlm_generated_ids_list")
    if vlm_ids_list is not None:
        extra["vlm_generated_ids"] = list(vlm_ids_list)
    return extra


def build_alpamayo_r1_forward_inputs(
    batched_model_input: BatchedModelInput,
    num_context_frames: int,
) -> dict[str, Any]:
    """Build the ReasoningVLA forward ``data`` dict from typed batched input.

    Image layout matches the chat-template packer in
    ``alpamayo.data.chat_template.conversation.construct_image``:
    per-batch shape ``(N_total, 1, 3, H, W)`` where the leading singleton
    slot is the per-image T=1 dim the Qwen processor expects. Returned
    ``image_frames`` are still raw uint8; ``AlpamayoR1InferenceModel._transform_inputs``
    converts them to float32 [-1, 1] before model dispatch.
    """
    camera_frames = batched_model_input.camera_frames
    t_total = camera_frames.shape[1]
    if t_total % num_context_frames != 0:
        raise ValueError(
            "AlpamayoR1InferenceModel: camera_frames leading dim "
            f"{t_total} is not divisible by num_context_frames="
            f"{num_context_frames}; cam-major reshape requires "
            "T = N_cam * num_context_frames."
        )
    if camera_frames.dtype != torch.uint8:
        raise ValueError(
            "AlpamayoR1InferenceModel: camera_frames must be uint8 "
            f"(BatchedModelInput contract); got {camera_frames.dtype}."
        )
    # Model has its own per-camera sort inside
    # ``_get_generation_mode_tokenized_data_online``; no presort here.
    image_frames = camera_frames.unsqueeze(2)

    # ReasoningVLA expects an ``n_traj_group=1`` axis on history (the
    # sampler unpacks ``B, n_traj_group, T, _ = ego_history_xyz.shape``).
    # ``AlpamayoPolicy._extract_historical_motion`` already inserts the
    # axis at dim=1 before ``BatchedModelInput.stack``, so policy-fed
    # inputs are 4-D (5-D for rot) and need no further unsqueeze. The
    # conditional preserves the contract for callers that hand in the
    # bare 3-D / 4-D layout (e.g. synthetic smoke inputs).
    ego_history_xyz = batched_model_input.ego_history_xyz
    if ego_history_xyz.ndim == 3:
        ego_history_xyz = ego_history_xyz.unsqueeze(1)
    ego_history_rot = batched_model_input.ego_history_rot
    if ego_history_rot.ndim == 4:
        ego_history_rot = ego_history_rot.unsqueeze(1)

    data: dict[str, Any] = {
        "image_frames": image_frames,
        "camera_indices": batched_model_input.camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    # Same n_traj_group=1 axis contract as ego history. ``route_xy``
    # is the legacy field name; XYZ flows through the same key.
    route_xy = batched_model_input.route_xy
    if route_xy.ndim == 3:
        route_xy = route_xy.unsqueeze(1)
    data["route_xy"] = route_xy
    return data
