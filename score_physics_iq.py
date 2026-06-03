"""Stage 2 of the Physics-IQ on lingbot bench: SCORE pre-generated videos.

Runs inside FastVideo's .venv (has fastvideo.eval). Reads any directory of
generated mp4s named per the Physics-IQ manifest's ``generated_video_name``
column, scores them with the official ``physics_iq`` metric (mse,
spatial_iou, spatiotemporal_iou, weighted_spatial_iou), aggregates, and
writes ``scores.json`` to the videos-dir.

Usage::

    "C:\\workspace\\world\\FastVideo\\.venv\\Scripts\\python.exe" \\
        "C:\\workspace\\world\\FastVideo\\score_physics_iq.py" \\
        --videos-dir outputs_video/bench_physics_iq_lingbot \\
        [--limit 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastvideo.eval import create_evaluator, get_metric
from fastvideo.eval.datasets import get_dataset


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory of generated mp4s named per Physics-IQ manifest.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--scores-out", type=Path, default=None)
    args = p.parse_args()

    if not args.videos_dir.is_dir():
        print(f"FATAL: videos-dir not found: {args.videos_dir}", file=sys.stderr)
        return 2

    ds = get_dataset("physics_iq", limit=args.limit)
    rows = list(ds)
    print(f"[load] {len(rows)} scenarios from {ds.dataset_dir}")

    samples = []
    matched = []
    for row in rows:
        gen_name = row["auxiliary_info"]["expected_gen_filename"]
        video_path = args.videos_dir / gen_name
        if not video_path.is_file():
            print(f"  [skip] missing: {gen_name}")
            continue
        samples.append({"video": str(video_path), **row})
        matched.append(row)

    if not samples:
        print("No generated videos matched the dataset's expected filenames.")
        return 1
    print(f"[eval] scoring {len(samples)} samples ...")

    ev = create_evaluator(metrics=["physics_iq"], num_gpus=args.num_gpus)
    results = ev.evaluate(samples=samples)
    ev.shutdown()

    metric = get_metric("physics_iq")
    components = metric.aggregate_components(
        [r["physics_iq"] for r in results]
    )

    print()
    print("=== Physics-IQ aggregate ===")
    for name, value in components.items():
        print(f"  {name:24s}  {value:.4f}")

    detailed = [
        {
            "scenario": row["auxiliary_info"]["scenario_id"],
            "view": row["view"],
            "scenario_name": row["auxiliary_info"]["scenario_name"],
            "score": r["physics_iq"].score,
        }
        for row, r in zip(matched, results)
    ]
    out = args.scores_out or (args.videos_dir / "scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"aggregate": components, "per_scenario": detailed},
        indent=2,
    ))
    print(f"\n[done] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
