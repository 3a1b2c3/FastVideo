# SANA-WM Integration Plan (FastVideo, WSL/Linux target)

Mirror the matrixgame2 integration pattern to add SANA-WM
(`Efficient-Large-Model/SANA-WM_bidirectional`) as a first-class model in
FastVideo. Target platform: WSL Ubuntu (not native Windows) — avoids the
Triton-windows / flash-attn / libuv / Py3.12 atexit cascade.

Reference upstream: `C:/workspace/world/Sana`
Reference integration: `fastvideo/{pipelines,models,configs,api}/matrixgame2*`

Estimated effort: **~1.5–2 weeks** (vs. 2–3 weeks on native Windows).

## Architecture summary

| Component | Source / target |
|---|---|
| DiT | `SanaMSVideoCamCtrl_1600M_P1_D20` — depth=20, hidden=2240, heads=20, head_dim=112, patch=(1,2,2) |
| Attention | `BidirectionalGDNTriton` (custom Triton kernel, ports clean on Linux) |
| FFN | `GLUMBConvTemp` (3D temporal-conv FFN) |
| VAE | LTX-2 (`AutoencoderKLLTX2Video`) — already in FastVideo via dreamverse |
| Text encoder | `Efficient-Large-Model/gemma-2-2b-it`, 300-token cap, scale_factor=0.01 |
| Scheduler | `FlowMatchEulerDiscreteScheduler` — already in FastVideo |
| Conditioning | Camera (B,F,20) — 16 C2W + 4 intrinsics — fused via UCPE ray-matrix + 3-channel absmap |
| Action DSL | `"w-10,iw-5,none-3"` → C2W trajectory |
| Refiner (optional) | LTX-2 sink-merged refiner |
| Weights | `Efficient-Large-Model/SANA-WM_bidirectional` (10.16 GB DiT) |

## Phase 0 — Feasibility (½ day)

1. Confirm WSL venv works (or create new Linux-native venv under `~/`,
   not `/mnt/c`).
2. Confirm LTX-2 VAE in FastVideo decodes at SANA-WM dims
   (latent_dim=128, stride [8,32,32]).
3. Build `BidirectionalGDNTriton` smoke (`python -c "import diffusion.model.nets.sana_multi_scale_video_camctrl"`).
4. Verify Gemma 2B-IT load via transformers.

Exit: Triton kernel imports clean; LTX-2 decode runs on a dummy latent;
Gemma loads with the expected hidden state shape.

## Phase 1 — Registry + skeleton (1–2 days)

| New / edited file | Mirrors matrixgame2 |
|---|---|
| `fastvideo/registry.py` (edit) | matrixgame2 entry at lines ~472–490 |
| `fastvideo/configs/pipelines/sana_wm.py` | `configs/pipelines/matrixgame2.py` |
| `fastvideo/configs/models/dits/sana_wm.py` | `configs/models/dits/matrixgame2.py` |
| `fastvideo/api/sana_wm.py` | `api/matrixgame2.py` (SamplingParam) |
| `fastvideo/pipelines/basic/sana_wm/{__init__.py,presets.py}` | `pipelines/basic/matrixgame2/presets.py` |

Defaults in preset: `height=720, width=1280, num_frames=81, fps=24, num_inference_steps=20, guidance_scale=1.0, sana_wm_720p`.

Exit: `VideoGenerator.from_pretrained("Efficient-Large-Model/SANA-WM_bidirectional")`
resolves to the SanaWM config without crashing.

## Phase 2 — DiT port (3–4 days, biggest task)

Files under `fastvideo/models/dits/sana_wm/`:

- `model.py` — `SanaMSVideoCamCtrl` + `SanaMSVideoCamCtrl_1600M_P1_D20`
- `blocks.py` — `BidirectionalGDNTriton` (port verbatim from Sana)
- `camera_module.py` — `_process_camera_conditions_ucpe` (rays + absmap)
- `ffn.py` — `GLUMBConvTemp`
- `utils.py` — `action_string_to_c2w` (camera DSL parser)

`param_names_mapping` written once during port; exercised by the test in
Phase 4.

## Phase 3 — Pipeline stages (2 days)

`fastvideo/pipelines/basic/sana_wm/sana_wm_pipeline.py`:

1. `input_validation_stage` — prompt + camera DSL validation
2. **NEW** `Gemma2TextEncodingStage` — 300-token cap, layernorm + 0.01 scale
3. **NEW** `CameraConditioningStage` — DSL → C2W → UCPE → (rays, absmap)
4. `timestep_preparation_stage` — reuse FlowMatchEuler path
5. `latent_preparation_stage` — image-conditioned VAE encode of first frame
6. `denoising_stage` — DiT forward with camera conds in kwargs
7. `decoding_stage` — LTX-2 VAE

## Phase 4 — Examples + tests (½ day)

| File | Notes |
|---|---|
| `examples/inference/basic/basic_sana_wm.py` | Mirror `basic_matrixgame2.py`. Default camera DSL `"w-31"` (matches existing SANA-WM logs) |
| `fastvideo/tests/transformers/test_sana_wm.py` | Mirror `test_matrixgame2.py` — fixed-seed forward, latent.sum() asserted against a reference recorded during Phase 2 |
| `run_sana_wm.sh` (repo root) | Shell, not `.bat`. Source `.venv`, run the example. No Windows env tweaks needed. |

## Phase 5 — Optional (skip for v1)

- Gradio demo (camera-control mapped to WASD-style keys)
- LTX-2 sink-merged refiner stage
- NVFP4 quantization (works on Linux — flashinfer has Linux wheels)

## Risk register

| Risk | Mitigation |
|---|---|
| Camera DSL semantics differ subtly from `_process_camera_conditions_ucpe` reference | Phase 2 numerical-equivalence test vs. upstream Sana on a fixed prompt + trajectory |
| LTX-2 VAE in FastVideo decodes at different latent stride than Sana expects | Verified in Phase 0 |
| WSL `/mnt/c/...` filesystem is slow for safetensors mmap (10–30× vs. ext4) | Pre-cache weights into `~/.cache/huggingface/hub` on the Linux side, not `/mnt/c` |
| Conda / system Python vs. venv conflicts | Dedicated Linux venv; no PYTHONPATH leakage |

## File touch list (cheat sheet)

```
fastvideo/
  registry.py                                          (edit)
  api/sana_wm.py                                       (new)
  configs/pipelines/sana_wm.py                         (new)
  configs/models/dits/sana_wm.py                       (new)
  models/dits/sana_wm/{__init__,model,blocks,camera_module,ffn,utils}.py  (new)
  pipelines/basic/sana_wm/{__init__,presets,sana_wm_pipeline}.py          (new)
  pipelines/stages/text_encoding.py                    (edit — add Gemma2TextEncodingStage)
  pipelines/stages/conditioning.py                     (edit — add CameraConditioningStage) or new file
  tests/transformers/test_sana_wm.py                   (new)

examples/inference/basic/basic_sana_wm.py              (new)
run_sana_wm.sh                                         (new, repo root)
apps/sana_wm/PLAN.md                                   (this file)
```
