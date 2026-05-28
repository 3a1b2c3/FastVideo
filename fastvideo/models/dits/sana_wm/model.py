# SPDX-License-Identifier: Apache-2.0
"""FastVideo wrapper for SANA-WM (`SanaMSVideoCamCtrl_1600M_P1_D20`).

Strategy: rather than vendor the 18 kLOC ``diffusion/`` package from upstream
Sana, we import the model class directly from a local Sana checkout via
``sys.path`` injection. This keeps the FastVideo side thin (~1 file) and
guarantees bit-exact forward equivalence with upstream — at the cost of a
Sana-repo dependency in development environments.

Resolution order for the Sana repo root:

1. ``$SANA_REPO_PATH`` env var (developer override)
2. ``./Sana`` next to this FastVideo checkout
3. ``C:/workspace/world/Sana`` (matches the working tree on the lead dev's box;
   reachable from WSL as ``/mnt/c/workspace/world/Sana``)

Phase 2E (current): expose a FastVideo-style class that instantiates the
upstream factory. Forward signature adaptation is minimal — we pass tensors
through and let upstream handle them. Phase 3 will wire the
``CameraConditioningStage`` so callers don't need to construct the camera
tensor themselves.
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch
import torch.nn as nn

_REPO_PATH_CANDIDATES: tuple[str, ...] = (
    os.environ.get("SANA_REPO_PATH") or "",
    str(pathlib.Path(__file__).resolve().parents[6] / "Sana"),
    "/mnt/c/workspace/world/Sana",
    "C:/workspace/world/Sana",
)


def _locate_sana_repo() -> str:
    for cand in _REPO_PATH_CANDIDATES:
        if not cand:
            continue
        p = pathlib.Path(cand)
        if (p / "diffusion" / "model" / "nets" / "sana_multi_scale_video_camctrl.py").is_file():
            return str(p)
    raise FileNotFoundError(
        "Sana repo not found. Set $SANA_REPO_PATH to the directory containing "
        "diffusion/model/nets/sana_multi_scale_video_camctrl.py, or clone the "
        "NVlabs/Sana repo into ./Sana relative to your FastVideo checkout."
    )


def _ensure_sana_on_sys_path() -> str:
    repo = _locate_sana_repo()
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


class SanaWMTransformer3DModel(nn.Module):
    """FastVideo-side wrapper for `SanaMSVideoCamCtrl_1600M_P1_D20`.

    The wrapper holds a reference to the upstream module under ``.model`` so
    state-dict loading + introspection work without translation. The forward
    signature is intentionally upstream-compatible — adaptation to FastVideo's
    pipeline stage protocol happens in the camera-conditioning + denoising
    stages (Phase 3).
    """

    def __init__(self, config):
        super().__init__()
        from fastvideo.configs.models.dits.sana_wm import SanaWMConfig

        if not isinstance(config, SanaWMConfig):
            raise TypeError(
                f"SanaWMTransformer3DModel expects SanaWMConfig, got {type(config).__name__}"
            )
        self.config = config

        repo_root = _ensure_sana_on_sys_path()
        try:
            from diffusion.model.nets.sana_multi_scale_video_camctrl import (
                SanaMSVideoCamCtrl_1600M_P1_D20,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Failed to import upstream Sana DiT from {repo_root!r}. "
                f"Install Sana's runtime deps "
                f"(`pip install -r {repo_root}/requirements/sana_wm.txt --no-build-isolation`) "
                f"in the FastVideo venv. Original error: {e}"
            ) from e

        arch = config.arch_config
        # ``input_size`` is the *latent* resolution (post-VAE), not pixel size.
        # Configurable via SanaWMArchConfig.input_size; defaults to 22 to
        # match SANA-WM_bidirectional. Other resolutions handled via
        # pe_interpolation.
        self.model = SanaMSVideoCamCtrl_1600M_P1_D20(
            input_size=arch.input_size,
            in_channels=arch.num_channels_latents,
            pe_interpolation=arch.pe_interpolation,
            pred_sigma=False,
            learn_sigma=False,
            # Forward SANA-WM_bidirectional's config.yaml verbatim — these
            # defaults differ from the upstream factory's:
            attn_type=arch.attn_type,
            camctrl_type=arch.camctrl_type,
            softmax_every_n=arch.softmax_every_n,
            ffn_type=arch.ffn_type,
            t_kernel_size=arch.t_kernel_size,
            conv_kernel_size=arch.conv_kernel_size,
            k_conv_only=arch.k_conv_only,
            mlp_ratio=arch.mlp_ratio,
            mlp_acts=list(arch.mlp_acts),
            use_pe=arch.use_pe,
            pos_embed_type=arch.pos_embed_type,
            qk_norm=arch.qk_norm,
            cross_norm=arch.cross_norm,
            fp32_attention=arch.fp32_attention,
            linear_head_dim=arch.attention_head_dim,
            chunk_split_strategy=arch.chunk_split_strategy,
            cam_attn_compress=arch.cam_attn_compress,
            init_cam_from_base=arch.init_cam_from_base,
            use_chunk_plucker_post_attn=arch.use_chunk_plucker_post_attn,
            chunk_plucker_channels=arch.chunk_plucker_channels,
            chunk_plucker_post_attn_blocks=arch.chunk_plucker_post_attn_blocks,
            mixed_precision=str(self.dtype_str()),
        )

    @staticmethod
    def dtype_str() -> str:
        # SANA-WM upstream config uses bf16; keep that default unless the
        # caller manually casts the model.
        return "bf16"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_config(cls, config) -> "SanaWMTransformer3DModel":
        return cls(config)
