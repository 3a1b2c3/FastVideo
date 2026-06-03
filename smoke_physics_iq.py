"""Smoke test the Physics-IQ dataset machinery in FastVideo.

Validates that:
  1. fastvideo + fastvideo.eval are importable.
  2. get_dataset('physics_iq') resolves and constructs.
  3. Per-scenario auto-fetch reaches DeepMind's GCS bucket.
  4. At least one row decodes cleanly.

Pulls a small subset (default --limit 2, ~25 MB) so it's safe to run
ad-hoc to catch network / install / cache issues before kicking off a
real benchmark.

Usage::

    python smoke_physics_iq.py
    python smoke_physics_iq.py --limit 4
    python smoke_physics_iq.py --cache-dir D:\\fv-eval

After this passes, run a real benchmark with::

    python examples/inference/eval/bench_physics_iq.py --limit 4 --num-gpus 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--limit",
        type=int,
        default=2,
        help="How many scenarios to fetch + validate. Default 2 (~25 MB).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Override FASTVIDEO_EVAL_CACHE for this run. "
            "Defaults to %USERPROFILE%\\.cache\\fastvideo\\eval\\."
        ),
    )
    p.add_argument(
        "--no-fetch",
        action="store_true",
        help=(
            "Pass auto_download=False. Only succeeds if the dataset is "
            "already on disk at --cache-dir / FASTVIDEO_EVAL_CACHE."
        ),
    )
    args = p.parse_args()

    if args.cache_dir is not None:
        os.environ["FASTVIDEO_EVAL_CACHE"] = str(args.cache_dir)
        print(f"[env]    FASTVIDEO_EVAL_CACHE = {args.cache_dir}")

    print(f"[python] {sys.executable}")
    print(f"[args]   limit={args.limit}  no-fetch={args.no_fetch}")

    try:
        from fastvideo.eval.datasets import get_dataset
    except ImportError as e:
        print(f"\nERROR: cannot import fastvideo.eval.datasets — {e}")
        print(
            "Fix: in the FastVideo repo, "
            "uv pip install -e .[eval]"
        )
        return 2

    try:
        ds = get_dataset(
            "physics_iq",
            limit=args.limit,
            auto_download=not args.no_fetch,
        )
    except Exception as e:
        print(f"\nERROR: get_dataset('physics_iq') failed: {type(e).__name__}: {e}")
        if args.no_fetch:
            print("Hint: --no-fetch but assets are missing. Drop --no-fetch to download.")
        else:
            print("Hint: check network access to https://storage.googleapis.com/physics-iq-benchmark")
        return 3

    try:
        rows = list(ds)
    except Exception as e:
        print(f"\nERROR: iterating dataset failed: {type(e).__name__}: {e}")
        return 4

    print(f"\n[ok]     {len(rows)} rows loaded")
    print(f"[ok]     cache dir : {ds.dataset_dir}")

    if not rows:
        print("\nWARN: dataset returned 0 rows. limit too small or manifest broken.")
        return 5

    first = rows[0]
    aux = first.get("auxiliary_info", {})
    print()
    print("=== first row ===")
    print(f"  scenario_id         : {aux.get('scenario_id')}")
    print(f"  scenario_name       : {aux.get('scenario_name')}")
    print(f"  view                : {first.get('view')}")
    print(f"  category            : {aux.get('category', '(not surfaced)')}")
    print(f"  prompt              : {first.get('prompt', '')[:80]}...")
    print(f"  reference (take-1)  : {first.get('reference')}")
    print(f"  reference_take2     : {first.get('reference_take2')}")
    print(f"  reference_mask      : {first.get('reference_mask')}")
    print(f"  reference_take2_mask: {first.get('reference_take2_mask')}")
    print(f"  expected_gen_name   : {aux.get('expected_gen_filename')}")

    # Verify the referenced files actually exist on disk.
    missing = []
    for key in ("reference", "reference_take2", "reference_mask", "reference_take2_mask"):
        path = first.get(key)
        if path is None or not Path(path).is_file():
            missing.append((key, path))
    if missing:
        print("\nWARN: row contains paths that don't exist on disk:")
        for key, path in missing:
            print(f"  {key:<22} -> {path}")
        return 6

    print("\n[ok]     all referenced files exist on disk")
    print("\nNext: run a real benchmark with")
    print("  python examples/inference/eval/bench_physics_iq.py --limit 4 --num-gpus 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
