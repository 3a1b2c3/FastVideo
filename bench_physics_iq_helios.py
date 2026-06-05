"""Stage 1 of Physics-IQ on Helios: GENERATE videos only.

Uses Helios's OWN diffusers-version pipeline (NOT the stock
WanImageToVideoPipeline -- that's a different model class and loading the
Helios HF repo into it segfaults with ACCESS_VIOLATION on component 2).

Loader mirrors C:\\workspace\\world\\MIND\\src\\_helios_i2v_worker.py which
already does this in-process for MIND. Pulls per-component subfolders
(transformer, vae, scheduler) from the Helios HF repo.

Run inside Helios's .venv via run_physics_iq_helios.bat::

    & "C:\\workspace\\world\\FastVideo\\run_physics_iq_helios.bat" --% --limit 4

Score afterwards inside FastVideo's .venv::

    & "C:\\workspace\\world\\FastVideo\\.venv\\Scripts\\python.exe" \\
        "C:\\workspace\\world\\FastVideo\\score_physics_iq.py" \\
        --videos-dir outputs_video\\bench_physics_iq_helios
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import urllib.request
from pathlib import Path


# Pre-import env tweaks before importing torch.
# DO NOT enable HF_PARALLEL_LOADING here: disable_mmap=True below reads
# each 9 GB Helios shard fully into RAM, and 8 parallel workers needed
# 72 GB peak (OOM'd on a 64 GB-free box). Sequential load is fine.
os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "no")
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "1"
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Make Helios's package importable when launched from elsewhere.
_HELIOS_REPO = Path(r"C:\workspace\world\Helios")
if _HELIOS_REPO.exists() and str(_HELIOS_REPO) not in sys.path:
    sys.path.insert(0, str(_HELIOS_REPO))

import av
import numpy as np
import torch
from PIL import Image

from diffusers.models import AutoencoderKLWan  # noqa: E402
from diffusers.utils import load_image  # noqa: E402
from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline  # noqa: E402
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler  # noqa: E402
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel  # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402


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

# Helios native defaults (from infer_helios.py).
HELIOS_W, HELIOS_H = 640, 384            # Helios's native low-res; multiples of 16
HELIOS_NUM_FRAMES = 99                   # native chunk
HELIOS_NUM_INFERENCE_STEPS = 30          # default 50 in infer_helios; 30 is decent quality
HELIOS_GUIDANCE_SCALE = 5.0
PHYSICS_IQ_FPS = 30                      # PyAV encode rate (Physics-IQ scorer realigns time)

VARIANT_TO_REPO = {
    "base":      "BestWishYsh/Helios-Base",
    "mid":       "BestWishYsh/Helios-Mid",
    "distilled": "BestWishYsh/Helios-Distilled",
}


# --- Manifest + asset paths ---------------------------------------------------


def load_manifest_rows(limit: int | None) -> list[dict]:
    with MANIFEST_CSV.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    take1 = [r for r in rows if "take-1" in r["scenario"]]
    if limit is not None:
        take1 = take1[:limit]
    return take1


def _bucket_fetch(rel_path: str, dest: Path, retries: int = 4) -> Path:
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET_URL}/{rel_path}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"  [fetch] {url}" + (f"  (attempt {attempt})" if attempt > 1 else ""))
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001 - WinError 10053 aborts, timeouts, transient 5xx
            last_exc = e
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts: {last_exc}")


def resolve_switch_frame(scenario_id: str, view: str, scenario_name: str,
                         dataset_root: Path) -> Path:
    # Official Physics-IQ layout (matches FastVideo's loader + the local cache):
    #   switch-frames/<id>_switch-frames_anyFPS_<view>_<name>.jpg
    fname = f"{scenario_id}_switch-frames_anyFPS_{view}_{scenario_name}.jpg"
    local = dataset_root / "switch-frames" / fname
    return _bucket_fetch(f"switch-frames/{fname}", local)


def parse_scenario_filename(s: str) -> tuple[str, str, str]:
    stem = s.rsplit(".", 1)[0]
    parts = stem.split("_", 3)
    scenario_id, view = parts[0], parts[1]
    # parts[2] is "take-N"; parts[3] is the scenario name. KEEP the "trimmed-"
    # prefix -- the official switch-frame filename includes it.
    name = parts[3]
    return scenario_id, view, name


# --- Helios pipeline build (mirrors _helios_i2v_worker.py) --------------------


def build_pipeline(repo_id: str, dtype: torch.dtype, low_vram: bool):
    """Load Helios's three components and stitch into HeliosPipeline."""
    print(f"[gen]  loading transformer from {repo_id}/transformer ...")
    transformer = HeliosTransformer3DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        # disable_mmap avoids the 80 GB Python RAM blowup + pagefile crash
        # documented in session memory; the loader streams shards instead.
        disable_mmap=True,
    )
    print("[gen]  loading vae ...")
    vae = AutoencoderKLWan.from_pretrained(
        repo_id, subfolder="vae", torch_dtype=dtype,
    )
    print("[gen]  loading scheduler ...")
    scheduler = HeliosScheduler.from_pretrained(repo_id, subfolder="scheduler")
    # Load the UMT5-XXL text encoder explicitly with low_cpu_mem_usage so the
    # pipeline assembler doesn't mmap-load it itself -- that path has no
    # low_cpu_mem_usage/disable_mmap and blows up host RAM/pagefile, crashing
    # with 0xC0000005 mid "Loading pipeline components". Passing it in (like
    # transformer/vae/scheduler) leaves nothing heavy for from_pretrained.
    print("[gen]  loading text encoder (UMT5-XXL) ...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        repo_id, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
    print("[gen]  assembling HeliosPipeline ...")
    pipe = HeliosPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=dtype,
    )
    if low_vram:
        pipe.enable_model_cpu_offload()
        print("[gen]  model CPU offload enabled")
    else:
        pipe.to("cuda")
    # VAE tiling/slicing slashes the video-decode VRAM peak -- usually the
    # single biggest spike for a 99-frame clip, and a common OOM source.
    for fn in ("enable_tiling", "enable_slicing"):
        if hasattr(pipe.vae, fn):
            getattr(pipe.vae, fn)()
    return pipe


# --- Output mp4 ---------------------------------------------------------------


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


def pil_frames_to_uint8(frames) -> np.ndarray:
    arr = np.stack([np.asarray(f) for f in frames], axis=0)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def generate_one(pipe, row: dict, dataset_root: Path, out_path: Path,
                 num_frames: int, num_steps: int, guidance: float,
                 seed: int, w: int, h: int) -> None:
    scenario_id, view, name = parse_scenario_filename(row["scenario"])
    first_frame_path = resolve_switch_frame(scenario_id, view, name, dataset_root)
    image = load_image(str(first_frame_path)).resize((w, h))

    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        image=image,
        prompt=row["description"],
        negative_prompt="",
        height=h,
        width=w,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        generator=gen,
    )

    frames = pil_frames_to_uint8(out.frames[0])
    write_mp4(frames, out_path, fps=PHYSICS_IQ_FPS)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(r"C:\workspace\world\FastVideo\outputs_video\bench_physics_iq_helios"),
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--variant", default="base", choices=list(VARIANT_TO_REPO))
    p.add_argument("--repo-id", default=None)
    p.add_argument("--height", type=int, default=HELIOS_H)
    p.add_argument("--width", type=int, default=HELIOS_W)
    p.add_argument("--num-frames", type=int, default=HELIOS_NUM_FRAMES)
    p.add_argument("--num-inference-steps", type=int, default=HELIOS_NUM_INFERENCE_STEPS)
    p.add_argument("--guidance-scale", type=float, default=HELIOS_GUIDANCE_SCALE)
    p.add_argument("--seed", type=int, default=42)
    # CPU-offload is the default: the 14B transformer + UMT5-XXL + VAE don't
    # fit resident on a 32 GB GPU. Use --high-vram to keep everything on-GPU
    # (needs a >40 GB card).
    p.add_argument("--low-vram", dest="low_vram", action="store_true", default=True,
                   help="enable_model_cpu_offload (default; slower, lower VRAM peak).")
    p.add_argument("--high-vram", dest="low_vram", action="store_false",
                   help="keep all components resident on GPU (needs >40 GB VRAM).")
    args = p.parse_args()

    if not MANIFEST_CSV.is_file():
        print(f"FATAL: manifest CSV missing at {MANIFEST_CSV}", file=sys.stderr)
        return 2

    rows = load_manifest_rows(args.limit)
    print(f"[load] {len(rows)} take-1 scenarios from {MANIFEST_CSV.name}")

    repo_id = args.repo_id or VARIANT_TO_REPO[args.variant]
    pipe = build_pipeline(repo_id, torch.bfloat16, args.low_vram)
    print(f"[gen]  pipeline ready ({args.width}x{args.height}, "
          f"{args.num_frames} frames, {args.num_inference_steps} steps, "
          f"cfg={args.guidance_scale})")

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
                pipe, row, args.dataset_root, out,
                args.num_frames, args.num_inference_steps,
                args.guidance_scale, args.seed,
                args.width, args.height,
            )
            print(f"    {time.perf_counter() - t0:.1f}s -> {out}")
        except Exception as e:
            import traceback
            print(f"    FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            # Free per-scenario VRAM so leaks can't accumulate into the hard
            # OOM that killed the sweep at scenario ~198.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("[done] Generated videos in:", args.videos_dir)
    print("Next: score them inside FastVideo's .venv:")
    print(
        f'  & "C:\\workspace\\world\\FastVideo\\.venv\\Scripts\\python.exe" '
        f'"C:\\workspace\\world\\FastVideo\\score_physics_iq.py" '
        f'--videos-dir "{args.videos_dir}"'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
