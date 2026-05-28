# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-WM (bidirectional) inference."""

from __future__ import annotations

from dataclasses import dataclass

from fastvideo.api.sampling_param import SamplingParam


@dataclass
class SanaWMSamplingParam(SamplingParam):
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 24
    guidance_scale: float = 1.0
    num_inference_steps: int = 20
    negative_prompt: str | None = None
    # Camera-control DSL string, parsed by Phase 3's CameraConditioningStage.
    # Example: "w-31" (forward for 31 frames), "w-10,iw-5,none-3".
    cam_ctrl_string: str = "w-31"
