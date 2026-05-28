"""Phase 0 feasibility smoke tests for SANA-WM port (WSL/Linux).

We do NOT want to install the full Sana runtime stack (mmcv, bitsandbytes,
flash-linear-attention, openai/clip, image-reward, gradio, ...) just to
validate compatibility — we'll be reimplementing the architecture inside
FastVideo anyway. Instead, this script:

  1. Confirms torch + CUDA + Triton are usable.
  2. Loads the SANA-WM Triton kernel source files directly via importlib
     (bypassing the Sana repo's __init__.py chain) and validates that the
     module parses and the @triton.jit decorators evaluate cleanly under
     Triton 3.6.0.
  3. Confirms diffusers ships AutoencoderKLLTXVideo (the VAE SANA-WM uses).
  4. Confirms transformers can construct the Gemma-2B-IT config.
  5. Compiles one Triton kernel end-to-end on the GPU with dummy inputs to
     prove sm_120 codegen works.
"""

import importlib.util
import pathlib
import sys
import time
import traceback


def step(name):
    print(f"\n=== {name} ===", flush=True)


def ok(msg):
    print(f"  [OK] {msg}", flush=True)


def fail(msg, exc):
    print(f"  [FAIL] {msg}: {exc.__class__.__name__}: {exc}", flush=True)
    traceback.print_exc(limit=4)


def load_module_from_path(name, path):
    """Import a .py file directly without going through its package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SANA_ROOT = pathlib.Path("/mnt/c/workspace/world/Sana")


step("1. Torch + CUDA + Triton sanity")
try:
    import torch
    import triton

    assert torch.cuda.is_available(), "CUDA unavailable"
    cap = torch.cuda.get_device_capability(0)
    ok(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"triton={triton.__version__} dev={torch.cuda.get_device_name(0)} "
        f"compute={cap[0]}.{cap[1]}"
    )
except Exception as e:
    fail("torch/triton import", e)
    sys.exit(1)


step("2. SANA-WM Triton kernel modules (direct file load)")
kernel_files = [
    ("sana_fused_recurrent", SANA_ROOT / "diffusion/model/ops/frame_gdn/fused_recurrent_triton.py"),
    ("sana_scan_triton", SANA_ROOT / "diffusion/model/ops/frame_gdn/scan_triton.py"),
]
loaded_kernels = {}
for modname, modpath in kernel_files:
    if not modpath.exists():
        fail(f"kernel file missing: {modpath}", FileNotFoundError(str(modpath)))
        continue
    try:
        t0 = time.time()
        mod = load_module_from_path(modname, modpath)
        loaded_kernels[modname] = mod
        kernels = [n for n in dir(mod) if not n.startswith("_") and ("kernel" in n.lower() or "fwd" in n or "bwd" in n)]
        ok(f"{modname}: loaded in {time.time() - t0:.2f}s, kernels: {kernels[:6]}")
    except Exception as e:
        fail(f"{modname} load", e)


step("3. LTX-2 VAE class import (diffusers)")
try:
    from diffusers import AutoencoderKLLTXVideo

    ok(
        f"diffusers.AutoencoderKLLTXVideo present at "
        f"{AutoencoderKLLTXVideo.__module__}"
    )
except Exception as e:
    fail("AutoencoderKLLTXVideo import", e)


step("4. Gemma 2B-IT config (no weight download)")
try:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("google/gemma-2-2b-it")
    ok(
        f"gemma-2-2b-it: hidden={cfg.hidden_size} "
        f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
        f"vocab={cfg.vocab_size}"
    )
except Exception as e:
    fail("Gemma config load", e)


step("5. Compile one Triton kernel end-to-end (sm_120 codegen check)")
try:
    import triton.language as tl

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)
    _add_kernel[(4,)](x, y, z, n, BLOCK=256)
    err = (z - (x + y)).abs().max().item()
    ok(f"compiled + ran trivial kernel; max abs err = {err:.2e} (n={n})")
except Exception as e:
    fail("Triton compile+launch", e)


print("\n=== Phase 0 complete ===", flush=True)
