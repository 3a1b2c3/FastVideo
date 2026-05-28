"""Phase 1 smoke: registry resolves SANA-WM model path to our skeleton config.

Run under the WSL venv:
    source ~/sana-wm-venv/bin/activate
    pip install -e /mnt/c/workspace/world/FastVideo
    python /mnt/c/workspace/world/FastVideo/apps/sana_wm/phase1_smoke.py

(If FastVideo isn't installed editable yet, this script falls back to
inserting the repo on sys.path.)
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import is_dataclass


def ok(msg):
    print(f"  [OK] {msg}", flush=True)


def fail(msg, exc):
    print(f"  [FAIL] {msg}: {exc.__class__.__name__}: {exc}", flush=True)
    traceback.print_exc(limit=4)


# Fallback path injection so the smoke runs without `pip install -e`.
REPO_ROOT = "/mnt/c/workspace/world/FastVideo"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


print("\n=== 1. Import SanaWMArchConfig + SanaWMConfig ===", flush=True)
try:
    from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig, SanaWMConfig

    arch = SanaWMArchConfig()
    cfg = SanaWMConfig()
    ok(
        f"arch: hidden={arch.hidden_size} layers={arch.num_layers} "
        f"heads={arch.num_attention_heads} head_dim={arch.attention_head_dim} "
        f"latent_ch={arch.num_channels_latents} patch={arch.patch_size} "
        f"in/out={arch.in_channels}/{arch.out_channels}"
    )
    ok(f"config prefix: {cfg.prefix}, is_dataclass: {is_dataclass(cfg)}")
except Exception as e:
    fail("SanaWMConfig import", e)
    sys.exit(1)


print("\n=== 2. Import SanaWM720PConfig (extends LTX2T2VConfig) ===", flush=True)
try:
    from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig

    pc = SanaWM720PConfig()
    ok(
        f"SanaWM720PConfig: dit={type(pc.dit_config).__name__} "
        f"vae={type(pc.vae_config).__name__} "
        f"text={type(pc.text_encoder_configs[0]).__name__} "
        f"flow_shift={pc.flow_shift} steps={pc.num_inference_steps} "
        f"{pc.height}x{pc.width} frames={pc.num_frames} fps={pc.fps}"
    )
except Exception as e:
    fail("SanaWM720PConfig import", e)


print("\n=== 3. Import SanaWMSamplingParam ===", flush=True)
try:
    from fastvideo.api.sana_wm import SanaWMSamplingParam

    sp = SanaWMSamplingParam()
    ok(
        f"sampling param: {sp.height}x{sp.width} frames={sp.num_frames} "
        f"fps={sp.fps} steps={sp.num_inference_steps} cam='{sp.cam_ctrl_string}'"
    )
except Exception as e:
    fail("SanaWMSamplingParam import", e)


print("\n=== 4. Import preset ===", flush=True)
try:
    from fastvideo.pipelines.basic.sana_wm.presets import ALL_PRESETS, SANA_WM_720P

    ok(
        f"preset {SANA_WM_720P.name} v{SANA_WM_720P.version} "
        f"family={SANA_WM_720P.model_family} workload={SANA_WM_720P.workload_type} "
        f"stages={[s.name for s in SANA_WM_720P.stage_schemas]} "
        f"defaults_keys={list(SANA_WM_720P.defaults.keys())}"
    )
    assert SANA_WM_720P in ALL_PRESETS
    ok(f"ALL_PRESETS contains preset; n={len(ALL_PRESETS)}")
except Exception as e:
    fail("preset import", e)


print("\n=== 5. Registry resolves Efficient-Large-Model/SANA-WM_bidirectional ===",
      flush=True)
try:
    from fastvideo.registry import (
        get_default_preset,
        get_model_family,
        get_pipeline_config_cls_from_name,
        get_sampling_param_cls_for_name,
    )

    path = "Efficient-Large-Model/SANA-WM_bidirectional"
    fam = get_model_family(path)
    ok(f"model_family: {fam}")
    assert fam == "sana_wm", f"expected 'sana_wm', got {fam!r}"
    preset_name = get_default_preset(path)
    ok(f"default preset for path: {preset_name}")
    assert preset_name == "sana_wm_720p", (
        f"expected 'sana_wm_720p', got {preset_name!r}"
    )
    pcfg_cls = get_pipeline_config_cls_from_name(path)
    ok(f"pipeline_config_cls: {pcfg_cls.__name__}")
    spm_cls = get_sampling_param_cls_for_name(path)
    ok(f"sampling_param_cls: {spm_cls.__name__ if spm_cls else None}")
except Exception as e:
    fail("registry resolution", e)


print("\n=== Phase 1 smoke complete ===", flush=True)
