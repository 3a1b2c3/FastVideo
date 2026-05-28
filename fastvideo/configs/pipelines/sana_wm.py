# SPDX-License-Identifier: Apache-2.0
"""SANA-WM (bidirectional) pipeline config.

Extends ``LTX2T2VConfig`` because SANA-WM reuses LTX-2's VAE
(``AutoencoderKLLTXVideo``) and Gemma 2B-IT text encoder. The only swap is
the DiT, which is replaced with ``SanaWMConfig`` (a
``SanaMSVideoCamCtrl_1600M_P1_D20`` arch).

Phase 1 wiring only — actual pipeline class + camera-conditioning stage land
in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models import DiTConfig
from fastvideo.configs.models.dits.sana_wm import SanaWMConfig
from fastvideo.pipelines.basic.ltx2.pipeline_configs import LTX2T2VConfig


@dataclass
class SanaWM720PConfig(LTX2T2VConfig):
    """SANA-WM 720p config (camera-controlled, 81 frames @ 24 fps, 20 steps)."""

    dit_config: DiTConfig = field(default_factory=SanaWMConfig)

    flow_shift: float | None = 9.8
    num_inference_steps: int = 20

    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 24
