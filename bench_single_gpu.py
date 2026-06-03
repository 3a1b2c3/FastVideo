#!/usr/bin/env python
"""Single-GPU FastVideo inference benchmark.

Mirrors the metrics of fastvideo/tests/performance (latency, throughput, peak
memory, and per-stage text-encoder / DiT / VAE-decode times) but runs on one
GPU, so it works on this box (1x RTX 5090, 32 GB) where the shipped 2-GPU
wan-t2v-1.3b benchmark can't.

Examples:
  # Default: Wan2.1-T2V-1.3B, 480x832, 45 frames, 4 steps (matches the shipped
  # 2-GPU config's workload so numbers are roughly comparable).
  python bench_single_gpu.py

  # More steps / different size
  python bench_single_gpu.py --steps 30 --height 480 --width 832 --frames 81

  # Try NVFP4 FA4 attention (Blackwell); falls back loudly if unsupported.
  python bench_single_gpu.py --nvfp4-fa4

Writes raw results to fastvideo/tests/performance/results/perf_<id>_<n>gpu.json
(same dir the pytest suite uses) and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Windows safety: transformers' async/threaded loaders can deadlock here.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
# Required so generate_video(...).logging_info carries per-stage timings.
os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

# Map pipeline stage class names -> the suite's component metric keys.
STAGE_METRIC_MAP = {
    "TextEncodingStage": "text_encoder_time_s",
    "DenoisingStage": "dit_time_s",
    "DmdDenoisingStage": "dit_time_s",
    "DecodingStage": "vae_decode_time_s",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--short-name", default=None, help="label for the result file (default: derived from --model)")
    p.add_argument("--prompt", default="Will Smith casually eats noodles, mid-shot framing, vibrant lighting.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=45)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--flow-shift", type=float, default=7.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5, help="measured runs (after warmup)")
    p.add_argument("--nvfp4-fa4", action="store_true", help="enable NVFP4 FA4 attention (Blackwell)")
    p.add_argument("--save-video", action="store_true", help="also write the mp4 (slower; excluded from gen time)")
    return p.parse_args()


def extract_stage_times(result: dict) -> dict[str, float]:
    """Best-effort pull of per-stage execution times from logging_info."""
    out: dict[str, float] = {}
    info = result.get("logging_info")
    stages = getattr(info, "stages", None)
    if not isinstance(stages, dict):
        return out
    for stage_name, stage_info in stages.items():
        metric = STAGE_METRIC_MAP.get(stage_name)
        if metric is None:
            continue
        if isinstance(stage_info, dict):
            t = stage_info.get("execution_time")
        else:
            t = getattr(stage_info, "execution_time", None)
        if t is not None:
            out[metric] = out.get(metric, 0.0) + float(t)
    return out


def main() -> int:
    args = parse_args()

    # Import after env vars are set so they take effect.
    import torch

    from fastvideo import VideoGenerator

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible.")
    device_name = torch.cuda.get_device_name(0)

    init_kwargs: dict = {
        "num_gpus": 1,
        "flow_shift": args.flow_shift,
        "vae_tiling": True,
        "text_encoder_precisions": ["fp32"],
    }
    if args.nvfp4_fa4:
        init_kwargs["nvfp4_fa4"] = True

    print(f"Loading {args.model} on {device_name} ...")
    generator = VideoGenerator.from_pretrained(args.model, **init_kwargs)

    short_name = args.short_name or args.model.rsplit("/", 1)[-1]
    out_dir = Path("fastvideo/tests/performance/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_bench_scratch.mp4"

    gen_kwargs: dict = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        fps=args.fps,
        output_path=str(scratch),
        save_video=args.save_video,
    )

    gen_times: list[float] = []
    e2e_times: list[float] = []
    peak_mems: list[float] = []
    stage_acc: dict[str, list[float]] = {}

    total = args.warmup + args.runs
    try:
        for i in range(total):
            tag = "warmup " if i < args.warmup else "measure"
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            result = generator.generate_video(**gen_kwargs)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            if not isinstance(result, dict):  # legacy API returns dict; be defensive
                result = {}
            gen_t = float(result.get("generation_time") or wall)
            e2e_t = float(result.get("e2e_latency") or wall)
            peak = result.get("peak_memory_mb")
            peak = float(peak) if peak is not None else torch.cuda.max_memory_allocated() / 1e6

            print(f"[{i + 1:>2}/{total}] {tag}  gen={gen_t:6.2f}s  e2e={e2e_t:6.2f}s  peak={peak:8.0f} MB")

            if i >= args.warmup:  # record measured runs only
                gen_times.append(gen_t)
                e2e_times.append(e2e_t)
                peak_mems.append(peak)
                for k, v in extract_stage_times(result).items():
                    stage_acc.setdefault(k, []).append(v)
    finally:
        try:
            generator.shutdown()
        except Exception:
            pass

    avg_gen = statistics.mean(gen_times)
    record = {
        "benchmark_id": f"{short_name}-1gpu",
        "model_short_name": short_name,
        "model_path": args.model,
        "device": device_name,
        "num_gpus": 1,
        "nvfp4_fa4": args.nvfp4_fa4,
        "generation_kwargs": {
            "height": args.height, "width": args.width, "num_frames": args.frames,
            "num_inference_steps": args.steps, "guidance_scale": args.guidance_scale,
            "seed": args.seed, "fps": args.fps,
        },
        "num_warmup_runs": args.warmup,
        "num_measurement_runs": args.runs,
        "avg_generation_time_s": round(avg_gen, 3),
        "individual_times_s": [round(t, 3) for t in gen_times],
        "avg_e2e_latency_s": round(statistics.mean(e2e_times), 3),
        "throughput_fps": round(args.frames / avg_gen, 3) if avg_gen else None,
        "max_peak_memory_mb": round(max(peak_mems), 1),
        "text_encoder_time_s": round(statistics.mean(stage_acc["text_encoder_time_s"]), 3) if "text_encoder_time_s" in stage_acc else None,
        "dit_time_s": round(statistics.mean(stage_acc["dit_time_s"]), 3) if "dit_time_s" in stage_acc else None,
        "vae_decode_time_s": round(statistics.mean(stage_acc["vae_decode_time_s"]), 3) if "vae_decode_time_s" in stage_acc else None,
    }

    out_file = out_dir / f"perf_{short_name}_1gpu.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\n=== Benchmark summary (avg over measured runs) ===")
    for k in ("avg_generation_time_s", "avg_e2e_latency_s", "throughput_fps",
              "max_peak_memory_mb", "text_encoder_time_s", "dit_time_s", "vae_decode_time_s"):
        print(f"  {k:24} {record[k]}")
    print(f"\nWrote {out_file.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
