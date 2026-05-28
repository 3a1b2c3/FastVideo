# SPDX-License-Identifier: Apache-2.0
"""SANA-WM (bidirectional) pipeline presets."""

from __future__ import annotations

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Flow-matching denoising pass with camera UCPE conditioning",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

SANA_WM_720P = InferencePreset(
    name="sana_wm_720p",
    version=1,
    model_family="sana_wm",
    description="SANA-WM bidirectional, 720p camera-controlled video generation",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 20,
        "negative_prompt": "",
    },
)

ALL_PRESETS = (SANA_WM_720P, )
