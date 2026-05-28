"""Minimal SANA-WM inference example.

Run from the FastVideo repo root in a venv that has Sana's runtime deps
installed (see ``apps/sana_wm/PLAN.md`` Phase 2B'):

    python examples/inference/basic/basic_sana_wm.py \
        --image /path/to/first_frame.png \
        --prompt "a sunny meadow with grazing cows" \
        --cam_dsl "w-31" \
        --output outputs/sana_wm.mp4

The first invocation pays a ~5-min model build cost. Subsequent calls reuse
the resident pipeline.
"""

from __future__ import annotations

import argparse
import pathlib

import imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=pathlib.Path,
        required=True,
        help="First-frame RGB image (any size; upstream center-crops to 704x1280).",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--cam_dsl",
        type=str,
        default="w-31",
        help="Camera DSL string, e.g. 'w-31' or 'w-10,iw-5,none-3'.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("outputs/sana_wm.mp4"),
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offload_vae",
        action="store_true",
        help="Move VAE to CPU between calls (saves ~1 GB VRAM; slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from fastvideo.pipelines.basic.sana_wm.generator import (
        SanaWMGenerationParams,
        SanaWMGenerator,
    )

    print(f"[basic_sana_wm] building pipeline (one-time, ~5 min)...")
    gen = SanaWMGenerator(offload_vae=args.offload_vae)
    print(f"[basic_sana_wm] generating: '{args.prompt}' / cam='{args.cam_dsl}'")
    out = gen.generate(
        image=args.image,
        prompt=args.prompt,
        cam_dsl=args.cam_dsl,
        params=SanaWMGenerationParams(
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed,
        ),
    )

    video = out["video"]  # (T, H, W, 3) uint8
    if not isinstance(video, np.ndarray):
        video = video.cpu().numpy() if hasattr(video, "cpu") else np.asarray(video)
    imageio.mimwrite(str(args.output), video, fps=args.fps, codec="libx264", quality=8)
    print(f"[basic_sana_wm] wrote {args.output} ({video.shape[0]} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
