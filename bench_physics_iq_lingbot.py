"""Stage 1 of the Physics-IQ on lingbot bench: GENERATE videos only.

Reads the Physics-IQ manifest CSV directly (no fastvideo import), pulls
take-1 / switch-frame paths from the local DeepMind-bucket mirror, runs
flashdreams-lingbot per scenario with a STATIC camera, saves mp4 outputs
at the dataset's ``expected_gen_filename``.

Scoring is decoupled: run ``score_physics_iq.py`` afterwards inside
FastVideo's .venv (which has fastvideo.eval) on the same videos-dir.

This split avoids dragging fastvideo's 40+ runtime deps into the
flashdreams .venv -- they live in different ecosystems.

Run inside flashdreams's .venv via run_physics_iq_lingbot.bat::

    "C:\\workspace\\world\\FastVideo\\run_physics_iq_lingbot.bat" --% --limit 4

Score afterwards::

    "C:\\workspace\\world\\FastVideo\\.venv\\Scripts\\python.exe" \\
        "C:\\workspace\\world\\FastVideo\\score_physics_iq.py" \\
        --videos-dir outputs_video/bench_physics_iq_lingbot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_FAST_FLASH,
)
from lingbot.encoder.camctrl import CamCtrlInput

# --- Constants vendored from fastvideo.eval.datasets.physics_iq ---------------

MANIFEST_CSV = Path(
    r"C:\workspace\world\FastVideo\fastvideo\eval\metrics\physics_iq\_vendored\descriptions.csv"
)
DEFAULT_DATASET_ROOT = (
    Path.home() / ".cache" / "fastvideo" / "eval" / "datasets" / "physics_iq"
)
BUCKET_URL = os.environ.get(
    "FASTVIDEO_PHYSICS_IQ_BUCKET_URL",
    "https://storage.googleapis.com/physics-iq-benchmark",
)

# Lingbot output config
DEFAULT_W, DEFAULT_H = 832, 480
PHYSICS_IQ_FPS = 30

SLUG_TO_CFG = {
    "lingbot-world-fast": PIPELINE_LINGBOT_WORLD_FAST,
    "lingbot-world-fast-flash": PIPELINE_LINGBOT_WORLD_FAST_FLASH,
}


# --- Manifest + asset paths (no fastvideo dep) --------------------------------


def load_manifest_rows(limit: int | None) -> list[dict]:
    """Read descriptions.csv and return one row per take-1 scenario."""
    with MANIFEST_CSV.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    take1 = [r for r in rows if "take-1" in r["scenario"]]
    if limit is not None:
        take1 = take1[:limit]
    return take1


def _bucket_fetch(rel_path: str, dest: Path) -> Path:
    """Download <bucket>/<rel_path> -> dest if not already present."""
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET_URL}/{rel_path}"
    print(f"  [fetch] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def resolve_switch_frame(scenario_id: str, view: str, scenario_name: str,
                         dataset_root: Path) -> Path:
    """Per Physics-IQ layout: switch_frames/<scenario>_<view>_<name>.png"""
    fname = f"{scenario_id}_{view}_{scenario_name}.png"
    local = dataset_root / "switch_frames" / fname
    return _bucket_fetch(f"switch_frames/{fname}", local)


def resolve_take1_video(scenario_id: str, view: str, scenario_name: str,
                        dataset_root: Path) -> Path:
    """Per Physics-IQ layout: video-data/take-1/<file>.mp4 (30fps)"""
    fname = f"{scenario_id}_{view}_take-1_trimmed-{scenario_name}.mp4"
    local = dataset_root / "videos" / fname
    return _bucket_fetch(f"video-data/30FPS/{fname}", local)


def parse_scenario_filename(s: str) -> tuple[str, str, str]:
    """`0001_perspective-left_take-1_trimmed-ball-and-block-fall.mp4`
    -> (scenario_id='0001', view='perspective-left', scenario_name='ball-and-block-fall')
    """
    stem = s.rsplit(".", 1)[0]
    parts = stem.split("_", 3)
    # parts[0]='0001', parts[1]='perspective-left', parts[2]='take-1',
    # parts[3]='trimmed-ball-and-block-fall'
    scenario_id, view = parts[0], parts[1]
    name = parts[3]
    if name.startswith("trimmed-"):
        name = name[len("trimmed-"):]
    return scenario_id, view, name


# --- Lingbot generation -------------------------------------------------------


def load_first_frame(path: str, w: int, h: int, device: torch.device) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=torch.bfloat16)


def make_static_camctrl(num_frames: int, device: torch.device) -> CamCtrlInput:
    """Identity camera (poses + intrinsics constant across all frames)."""
    intr = (
        torch.tensor(
            [DEFAULT_W / 2.0, DEFAULT_W / 2.0, DEFAULT_W / 2.0, DEFAULT_H / 2.0],
            dtype=torch.float32,
        )
        .expand(num_frames, 4)
        .contiguous()
        .to(device=device)
    )
    poses = (
        torch.eye(4, dtype=torch.float32)
        .expand(num_frames, 4, 4)
        .contiguous()
        .to(device=device)
    )
    return CamCtrlInput(intrinsics=intr, poses=poses, world_scale=1.0)


def chunks_to_uint8(chunks: list[torch.Tensor]) -> np.ndarray:
    video = torch.cat([c.detach().cpu().float() for c in chunks], dim=0)
    video = ((video.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
    return video.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


def write_mp4(frames_thwc: np.ndarray, out_path: Path, fps: int = PHYSICS_IQ_FPS) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames_thwc.shape
    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "18"}
        for frame_np in frames_thwc:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def generate_one(pipeline, row: dict, dataset_root: Path, out_path: Path,
                 total_blocks: int, frames_per_block: int,
                 device: torch.device) -> None:
    scenario_id, view, name = parse_scenario_filename(row["scenario"])
    first_frame_path = resolve_switch_frame(scenario_id, view, name, dataset_root)
    first_t = load_first_frame(str(first_frame_path), DEFAULT_W, DEFAULT_H, device)

    sp = pipeline.decoder.spatial_compression_ratio
    cache = pipeline.initialize_cache(
        text=[row["description"]],
        image=first_t,
        height=DEFAULT_H // sp,
        width=DEFAULT_W // sp,
    )

    chunks: list[torch.Tensor] = []
    for i in range(total_blocks):
        camctrl = make_static_camctrl(frames_per_block, device)
        chunk = pipeline.generate(autoregressive_index=i, cache=cache, input=camctrl)
        pipeline.finalize(autoregressive_index=i, cache=cache)
        chunks.append(chunk)

    frames = chunks_to_uint8(chunks)
    target_frames = 5 * PHYSICS_IQ_FPS
    if frames.shape[0] > target_frames:
        frames = frames[:target_frames]
    write_mp4(frames, out_path, fps=PHYSICS_IQ_FPS)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local Physics-IQ mirror (auto-creates + fetches missing assets).",
    )
    p.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(r"C:\workspace\world\FastVideo\outputs_video\bench_physics_iq_lingbot"),
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--slug", default="lingbot-world-fast", choices=list(SLUG_TO_CFG))
    p.add_argument(
        "--total-blocks",
        type=int,
        default=6,
        help="AR blocks (6 * 25 frames = 150 = 5s @ 30fps).",
    )
    p.add_argument("--frames-per-block", type=int, default=25)
    args = p.parse_args()

    if not MANIFEST_CSV.is_file():
        print(f"FATAL: manifest CSV missing at {MANIFEST_CSV}", file=sys.stderr)
        return 2

    rows = load_manifest_rows(args.limit)
    print(f"[load] {len(rows)} take-1 scenarios from {MANIFEST_CSV.name}")

    print(f"[gen]  loading {args.slug} pipeline ...")
    pipeline = SLUG_TO_CFG[args.slug].setup().to("cuda").eval()
    device = torch.device("cuda")
    print("[gen]  pipeline ready")

    args.videos_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows):
        out_name = row["generated_video_name"]
        out = args.videos_dir / out_name
        if out.is_file():
            print(f"  [{i+1}/{len(rows)}] skip {out_name} (exists)")
            continue
        print(f"  [{i+1}/{len(rows)}] gen  {out_name}")
        try:
            t0 = time.perf_counter()
            generate_one(
                pipeline, row, args.dataset_root, out,
                args.total_blocks, args.frames_per_block, device,
            )
            print(f"    {time.perf_counter() - t0:.1f}s -> {out}")
        except Exception as e:
            import traceback
            print(f"    FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    print()
    print("[done] Generated videos in:", args.videos_dir)
    print("Next: score them inside FastVideo's .venv:")
    print(
        f'  "C:\\workspace\\world\\FastVideo\\.venv\\Scripts\\python.exe" '
        f'"C:\\workspace\\world\\FastVideo\\score_physics_iq.py" '
        f'--videos-dir "{args.videos_dir}"'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
