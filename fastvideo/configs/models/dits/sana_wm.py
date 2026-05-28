# SPDX-License-Identifier: Apache-2.0
"""SANA-WM (bidirectional world model) transformer architecture config.

Mirrors the upstream config at
``Sana/output/pretrained_models/SANA-WM_bidirectional/config.yaml`` for the
``SanaMSVideoCamCtrl_1600M_P1_D20`` factory.

Phase 1 wiring only — module/forward implementations land in Phase 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_sana_wm_blocks(name: str, _module) -> bool:
    return re.search(r"(?:^|\.)blocks\.\d+$", name) is not None


@dataclass
class SanaWMArchConfig(DiTArchConfig):
    """Architecture configuration for SANA-WM (`SanaMSVideoCamCtrl_1600M_P1_D20`)."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_sana_wm_blocks])
    _compile_conditions: list = field(default_factory=lambda: [is_sana_wm_blocks])

    param_names_mapping: dict = field(default_factory=lambda: {})
    reverse_param_names_mapping: dict = field(default_factory=lambda: {})
    lora_param_names_mapping: dict = field(default_factory=lambda: {})

    # Core transformer dims for the 1600M variant (depth=20, hidden=2240,
    # heads=20, head_dim=112). Source: upstream factory
    # SanaMSVideoCamCtrl_1600M_P1_D20.
    num_layers: int = 20
    num_attention_heads: int = 20
    attention_head_dim: int = 112
    hidden_size: int = 2240

    # Latent input/output match the LTX-2 VAE (latent_dim=128, stride
    # [8, 32, 32]). Patch size (T, H, W) = (1, 1, 1) for the P1 variant.
    num_channels_latents: int = 128
    patch_size: tuple[int, int, int] = (1, 1, 1)
    in_channels: int | None = None
    out_channels: int | None = None

    # Latent-grid resolution used for the learned pos_embed. SANA-WM_bidirectional
    # was trained with a 22x22 latent grid (722px / VAE stride 32 ≈ 22), and
    # other resolutions are handled at inference via ``pe_interpolation``.
    # Override when porting checkpoints trained at a different latent size.
    input_size: int = 22
    pe_interpolation: float = 1.0

    # Camera-control attention. The Triton kernels live at
    # diffusion/model/ops/frame_gdn/* in the upstream Sana repo and will be
    # ported in Phase 2 — these names are recorded so the model loader knows
    # which kernel path to dispatch to.
    attn_type: str = "BidirectionalGDNTriton"
    camctrl_type: str = "BidirectionalGDNUCPESinglePathLiteLABothTriton"
    softmax_every_n: int = 4

    # FFN
    ffn_type: str = "GLUMBConvTemp"
    t_kernel_size: int = 3
    conv_kernel_size: int = 4
    k_conv_only: bool = True
    mlp_ratio: int = 3
    mlp_acts: tuple[str | None, ...] = field(default_factory=lambda: ("silu", "silu", None))

    # Position embedding
    use_pe: bool = True
    pos_embed_type: str = "wan_rope"

    # Norms
    qk_norm: bool = True
    cross_norm: bool = True
    fp32_attention: bool = True

    # Chunking / camera attention
    chunk_split_strategy: str = "first_chunk_plus_one"
    cam_attn_compress: int = 1
    init_cam_from_base: bool = True
    use_chunk_plucker_post_attn: bool = True
    chunk_plucker_channels: int = 48
    chunk_plucker_post_attn_blocks: int = 20

    # Text conditioning (Gemma 2B-IT hidden_size=2304, model_max_length=300)
    text_max_length: int = 300
    text_hidden_size: int = 2304
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01

    # Camera input shape: (B, F, 20) = 16 C2W flat + 4 intrinsics
    cam_dim_in: int = 20

    def __post_init__(self):
        super().__post_init__()
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        if self.in_channels is None:
            self.in_channels = self.num_channels_latents * patch_volume
        if self.out_channels is None:
            self.out_channels = self.in_channels


@dataclass
class SanaWMConfig(DiTConfig):
    """SANA-WM transformer config (top-level wrapper)."""

    arch_config: DiTArchConfig = field(default_factory=SanaWMArchConfig)
    prefix: str = "sana_wm"
