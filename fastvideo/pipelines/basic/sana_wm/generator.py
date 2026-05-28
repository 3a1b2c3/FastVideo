# SPDX-License-Identifier: Apache-2.0
"""High-level SANA-WM generation helper.

Wraps the upstream ``inference_sana_wm.SanaWMPipeline`` for callers who just
want ``(prompt, image, cam_dsl) -> mp4``. Bypasses FastVideo's component
loader chain (which expects a diffusers ``model_index.json``); the SANA-WM HF
repo ships its own ``config.yaml`` instead.

A deeper, fully native FastVideo integration with ``TransformerLoader`` etc.
can come later — see ``apps/sana_wm/PLAN.md`` Phase 3B/3C notes.

Usage::

    gen = SanaWMGenerator()              # locates HF cache + Sana repo
    out = gen.generate(
        image=Image.open("first_frame.png"),
        prompt="a sunny meadow",
        cam_dsl="w-31",
        num_frames=32,
    )
    out["video"]   # (T, H, W, 3) uint8 numpy array
"""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from fastvideo.logger import init_logger
from fastvideo.models.dits.sana_wm.model import _locate_sana_repo

logger = init_logger(__name__)

DEFAULT_HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"


@dataclass
class SanaWMGenerationParams:
    """Per-call generation knobs (mirrors upstream ``GenerationParams``)."""

    num_frames: int = 32
    fps: int = 16
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    flow_shift: float | None = None
    seed: int = 42
    negative_prompt: str = ""


def _hf_snapshot_root(repo_id: str = DEFAULT_HF_REPO) -> pathlib.Path:
    """Resolve a local snapshot of the SANA-WM HF repo (no network call)."""
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(repo_id, local_files_only=True))


class SanaWMGenerator:
    """High-level (prompt, image, camera DSL) → mp4 generator."""

    def __init__(
        self,
        repo_id: str = DEFAULT_HF_REPO,
        device: str = "cuda",
        offload_vae: bool = False,
        offload_refiner: bool = True,
    ):
        sana_root = _locate_sana_repo()
        if sana_root not in sys.path:
            sys.path.insert(0, sana_root)
        # The upstream package needs its inference scripts dir importable too.
        scripts_dir = os.path.join(sana_root, "inference_video_scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        import pyrallis

        from inference_sana_wm import InferenceConfig, SanaWMPipeline

        snap = _hf_snapshot_root(repo_id)
        config_yaml = snap / "config.yaml"
        if not config_yaml.exists():
            raise FileNotFoundError(
                f"SANA-WM config.yaml missing from snapshot at {snap}. "
                f"Re-run `huggingface-cli download {repo_id}`."
            )

        config = pyrallis.parse(
            config_class=InferenceConfig, config_path=str(config_yaml), args=[]
        )
        # The config.yaml leaves ``vae_pretrained`` as the hf://... URI;
        # rewrite to the local snapshot dir so the upstream builder can find
        # the ``vae/`` subfolder without network access.
        config.vae.vae_pretrained = f"hf://{repo_id}"

        model_safetensors = snap / "dit" / "sana_wm_1600m_720p.safetensors"
        self._pipeline = SanaWMPipeline(
            config=config,
            model_path=str(model_safetensors),
            device=device,
            refiner=None,
            offload_vae=offload_vae,
            offload_refiner=offload_refiner,
        )
        self.config = config
        self.repo_id = repo_id
        self.device = torch.device(device)
        logger.info("SanaWMGenerator ready (repo=%s, device=%s)", repo_id, device)

    def generate(
        self,
        image: Image.Image | str | pathlib.Path,
        prompt: str,
        cam_dsl: str = "w-31",
        params: SanaWMGenerationParams | None = None,
    ) -> dict:
        """Generate a video. Returns the upstream dict
        (``video``, ``c2w``, ``latent``).

        For details on inputs and output shape, see
        ``inference_sana_wm.SanaWMPipeline.generate`` in the Sana repo.
        """
        from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w

        params = params or SanaWMGenerationParams()
        if isinstance(image, (str, pathlib.Path)):
            image = Image.open(image)
        image = image.convert("RGB")

        # Upstream's generate() expects an image already cropped to (704, 1280)
        # — without this preprocess the LTX-2 VAE tiled_encode mis-shapes and
        # raises. Use upstream's own resize_and_center_crop to stay in lockstep
        # with their intrinsics convention.
        from inference_sana_wm import resize_and_center_crop

        image, _src_size, _resized_size, _crop_offset = resize_and_center_crop(image)

        # Action DSL → camera trajectory; upstream wants (F, 4, 4) C2W +
        # (F, 4) intrinsics.
        c2w = action_string_to_c2w(cam_dsl)
        if c2w.shape[0] < params.num_frames:
            raise ValueError(
                f"cam_dsl {cam_dsl!r} produced {c2w.shape[0]} poses; need at least "
                f"{params.num_frames}. Extend the DSL or reduce num_frames."
            )
        c2w = c2w[: params.num_frames]
        intrinsics_vec4 = np.broadcast_to(
            np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32),
            (params.num_frames, 4),
        ).copy()

        from inference_sana_wm import GenerationParams as _UpstreamParams

        upstream_params = _UpstreamParams(
            num_frames=params.num_frames,
            fps=params.fps,
            step=params.num_inference_steps,
            cfg_scale=params.guidance_scale,
            flow_shift=params.flow_shift,
            seed=params.seed,
            negative_prompt=params.negative_prompt,
        )
        return self._pipeline.generate(
            image=image,
            prompt=prompt,
            c2w=c2w,
            intrinsics_vec4=intrinsics_vec4,
            params=upstream_params,
        )


__all__ = ["SanaWMGenerator", "SanaWMGenerationParams", "DEFAULT_HF_REPO"]
