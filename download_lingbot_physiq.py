"""Pre-fetch every artefact needed to run bench_physics_iq_lingbot.py:

  - lingbot-world-fast pipeline (DiT + Wan VAE + UMT5 text encoder + CLIP image encoder)
  - Physics-IQ dataset (take-1, take-2, masks, switch-frames)

The lingbot pipeline pulls its components transitively through flashdreams's
config-driven loaders; calling ``.setup()`` once forces all HF / S3 / URL
fetches without running any inference. Physics-IQ is fetched via the
``get_dataset("physics_iq")`` auto-download path.

Run this BEFORE the bench so the long run isn't gated on network stalls.

Usage::

    python download_lingbot_physiq.py
    python download_lingbot_physiq.py --slug lingbot-world-fast-flash
    python download_lingbot_physiq.py --physics-iq-limit 4   (~50 MB physics-iq pull)
    python download_lingbot_physiq.py --skip-lingbot         (only physics-iq)
    python download_lingbot_physiq.py --skip-physics-iq      (only lingbot)
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
    p.add_argument("--slug", default="lingbot-world-fast",
                   choices=["lingbot-world-fast", "lingbot-world-fast-flash"])
    p.add_argument("--physics-iq-limit", type=int, default=None,
                   help="Truncate Physics-IQ to first N scenarios. None = full pull (~4-6 GB).")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Override FASTVIDEO_EVAL_CACHE. Default: ~/.cache/fastvideo/eval/")
    p.add_argument("--skip-lingbot", action="store_true",
                   help="Don't pull the lingbot pipeline (only Physics-IQ).")
    p.add_argument("--skip-physics-iq", action="store_true",
                   help="Don't pull Physics-IQ (only lingbot).")
    args = p.parse_args()

    if args.cache_dir is not None:
        os.environ["FASTVIDEO_EVAL_CACHE"] = str(args.cache_dir)
        print(f"[env]  FASTVIDEO_EVAL_CACHE = {args.cache_dir}")

    print(f"[python] {sys.executable}")
    rc = 0

    # 1. Lingbot pipeline (~25 GB: DiT + Wan VAE + UMT5 + CLIP).
    if not args.skip_lingbot:
        try:
            print(f"\n=== fetch lingbot pipeline: {args.slug} ===")
            print("(transitively pulls DiT + Wan VAE + UMT5 text encoder + CLIP image encoder)")
            from lingbot.config import (
                PIPELINE_LINGBOT_WORLD_FAST,
                PIPELINE_LINGBOT_WORLD_FAST_FLASH,
            )
            cfg = {
                "lingbot-world-fast":       PIPELINE_LINGBOT_WORLD_FAST,
                "lingbot-world-fast-flash": PIPELINE_LINGBOT_WORLD_FAST_FLASH,
            }[args.slug]
            # .setup() instantiates encoder + decoder + transformer + scheduler,
            # forcing every HF/URL download. .to("cuda") not called — we don't
            # need to materialize on GPU just to download.
            pipeline = cfg.setup()
            print(f"[ok] lingbot {args.slug} pipeline ready (all weights on disk)")
            del pipeline
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError as e:
            print(f"ERROR: cannot import lingbot.config -- {e}")
            print("Run this script inside the flashdreams uv env:")
            print("  cd C:\\workspace\\world\\flashdreams")
            print("  uv run --package flashdreams-lingbot python "
                  "C:\\workspace\\world\\FastVideo\\download_lingbot_physiq.py")
            rc = 2
        except Exception as e:
            print(f"ERROR fetching lingbot: {type(e).__name__}: {e}")
            rc = 3

    # 2. Physics-IQ dataset.
    if not args.skip_physics_iq:
        try:
            print(f"\n=== fetch Physics-IQ (limit={args.physics_iq_limit}) ===")
            from fastvideo.eval.datasets import get_dataset
            ds = get_dataset("physics_iq", limit=args.physics_iq_limit)
            rows = list(ds)
            print(f"[ok] Physics-IQ: {len(rows)} scenarios at {ds.dataset_dir}")
        except ImportError as e:
            print(f"ERROR: cannot import fastvideo.eval.datasets -- {e}")
            print("Fix: cd C:\\workspace\\world\\FastVideo && uv pip install -e .[eval]")
            rc = 4
        except Exception as e:
            print(f"ERROR fetching Physics-IQ: {type(e).__name__}: {e}")
            rc = 5

    if rc == 0:
        print("\n[done] all downloads complete.")
        print("Next:")
        print("  python C:\\workspace\\world\\FastVideo\\bench_physics_iq_lingbot.py --limit 4")
    return rc


if __name__ == "__main__":
    sys.exit(main())
