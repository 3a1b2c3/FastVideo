# SANA-WM Integration Plan (FastVideo, WSL/Linux target)

Mirror the matrixgame2 integration pattern to add SANA-WM
(`Efficient-Large-Model/SANA-WM_bidirectional`) as a first-class model in
FastVideo. Target platform: WSL Ubuntu (not native Windows).

Reference upstream: `C:/workspace/world/Sana`
Reference integration: `fastvideo/{pipelines,models,configs,api}/matrixgame2*`

**Status: Phase 0–3 done + Phase 4 example + first real generation
(`outputs/sana_wm_demo_0.mp4`).**

Estimated effort vs. actual:

| Phase | Original estimate | Actual |
|---|---|---|
| 0 — feasibility | ~4 hrs | ~10 min |
| 1 — registry + skeleton | ~12 hrs | ~30 min |
| 2 — DiT port | 3–4 days (24–32 hrs) | ~1 hr (after Plan-B pivot) |
| 3 — pipeline stages | ~16 hrs | ~40 min |
| 4 — example + launcher | ~4 hrs | ~10 min |
| **First working video** | **7–10 days** | **~4 hours** |

## Architecture summary

| Component | Source / target |
|---|---|
| DiT | `SanaMSVideoCamCtrl_1600M_P1_D20` — depth=20, hidden=2240, heads=20, head_dim=112 — **2.66B params** with camera branches |
| Attention | `BidirectionalGDNTriton` (custom Triton kernel) |
| VAE | LTX-2 `AutoencoderKLLTXVideo` — already in FastVideo via dreamverse |
| Text encoder | Gemma 2B-IT, 300-token cap, scale_factor=0.01 — already in FastVideo via dreamverse |
| Scheduler | `FlowMatchEulerDiscreteScheduler` — already in FastVideo |
| Conditioning | Camera (B,F,20) — 16 C2W + 4 intrinsics |
| Refiner (optional) | LTX-2 sink-merged refiner (Phase 5) |

## Phase 2 — Plan B (PYTHONPATH + thin wrapper)

**Initial plan was to vendor the upstream Sana DiT code.** Reality:

- `Sana/diffusion/` is 329 .py files, 17.6 kLOC. Full vendoring would mean
  re-implementing 20+ tightly-coupled modules and patching dozens of import
  statements.
- Hit-and-iterate cost was estimated 6–10 hrs vendoring vs 1–2 hrs PYTHONPATH.

**The pivot:** keep Sana on `sys.path` at runtime and write a thin FastVideo
wrapper that imports `SanaMSVideoCamCtrl_1600M_P1_D20` directly from the
upstream module.

Files created in Phase 2:
- `fastvideo/configs/models/dits/sana_wm.py` — `SanaWMArchConfig` (2240/20/20/112)
- `fastvideo/configs/pipelines/sana_wm.py` — `SanaWM720PConfig` extending `LTX2T2VConfig`
- `fastvideo/api/sana_wm.py` — `SanaWMSamplingParam`
- `fastvideo/models/dits/sana_wm/model.py` — `SanaWMTransformer3DModel(nn.Module)` wrapper
- `fastvideo/models/dits/sana_wm/utils.py` — `action_string_to_c2w` DSL parser (FastVideo-native port)
- `fastvideo/pipelines/basic/sana_wm/{__init__,presets,sana_wm_pipeline,camera_conditioning_stage,generator}.py`
- `fastvideo/tests/transformers/test_sana_wm.py` — 50+ tests covering Phase 1 + 2 + 3A + 3B
- `fastvideo/registry.py` (edit) — registration block

Bug fixes layered in along the way:
1. `camctrl_type` default mismatch — forward arch_config to factory
2. `input_size` is latent grid (22), not pixel grid (720); `in_channels=128`
3. `pyrallis.parse(config_class=, config_path=, args=[])`, not `pyrallis.load`
4. Upstream `SanaWMPipeline.generate()` needs 704×1280 pre-crop via
   `resize_and_center_crop` before VAE encode

## Phase 3 — Pipeline stages + Generator

`SanaWMPipeline(ComposedPipelineBase)` with 7 stages (input_validation,
prompt_encoding via Gemma, **camera_conditioning** [new],
timestep_preparation, latent_preparation, denoising, decoding via LTX-2 VAE).

`SanaWMGenerator` — high-level wrapper class that delegates to upstream
`SanaWMPipeline`. Bypasses FastVideo's component-loader chain (which
expects a diffusers `model_index.json`, but the SANA-WM HF repo ships
`config.yaml` instead).

## Phase 4 — Example + launcher

- `examples/inference/basic/basic_sana_wm.py` — CLI: `--image --prompt --cam_dsl --output`
- `run_sana_wm.sh` — WSL launcher

## Phase 5 — Optional (not done)

- Native FastVideo `TransformerLoader` integration (vs. the PYTHONPATH wrapper) — couple hours
- Gradio UI mirroring upstream `app_sana_wm.py` — ~1 day
- LTX-2 sink-merged refiner stage — quality bump, longer runtime
- NVFP4 quantization — bandwidth + memory savings on Linux

## Risk register (resolved)

| Risk | Resolution |
|---|---|
| Camera DSL semantics differ from upstream | Used upstream `_process_camera_conditions_ucpe` via the wrapper; the FastVideo-native `CameraConditioningStage` (Phase 3A) does its own (B,F,20) packing but defers UCPE expansion to the DiT |
| LTX-2 VAE in FastVideo decodes at different latent stride | Verified Phase 0 — same (latent_dim=128, stride [8,32,32]) |
| WSL `/mnt/c/...` mmap slowdown | Confirmed real but acceptable for one-off generation; production should cache weights on ext4 |

## How to run

```
bash /mnt/c/workspace/world/FastVideo/run_sana_wm.sh \
    --image .../first_frame.png \
    --prompt "..." \
    --cam_dsl "w-31"
```

First call: ~5 min pipeline build (Sana DiT + LTX-2 VAE + Gemma load).
Subsequent gens on a 5090: ~30–60 sec per 32-frame clip.

## File touch list (cheat sheet)

```
fastvideo/
  registry.py                                          (edit)
  api/sana_wm.py                                       (new)
  configs/pipelines/sana_wm.py                         (new)
  configs/models/dits/sana_wm.py                       (new)
  models/dits/sana_wm/{__init__,model,utils}.py        (new)
  pipelines/basic/sana_wm/                             (new)
    __init__.py
    presets.py
    sana_wm_pipeline.py
    camera_conditioning_stage.py
    generator.py
  tests/transformers/test_sana_wm.py                   (new)

examples/inference/basic/basic_sana_wm.py              (new)
run_sana_wm.sh                                         (new, repo root)
apps/sana_wm/PLAN.md                                   (this file)
```
