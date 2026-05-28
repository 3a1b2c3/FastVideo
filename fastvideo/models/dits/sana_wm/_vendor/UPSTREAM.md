# Vendored upstream code

Source repo: <https://github.com/NVlabs/Sana>

Local reference checkout: `C:/workspace/world/Sana` (also reachable from WSL
at `/mnt/c/workspace/world/Sana`).

## Why vendor

SANA-WM ships a custom Triton kernel (`BidirectionalGDNTriton`), a non-trivial
camera-conditioning module (UCPE: rays + 3-channel absmap), and a 1600M-param
DiT (`SanaMSVideoCamCtrl`). Reimplementing this in FastVideo-native style is
a 3–4 day project; vendoring + thin-wrap takes ~6–10 hours and preserves
bit-exact weight compatibility with upstream checkpoints.

See `apps/sana_wm/PLAN.md` for the full Phase 2 plan.

## Files vendored so far

| Vendored path | Upstream path | LOC | Notes |
|---|---|---:|---|
| `ops/frame_gdn/fused_recurrent_triton.py` | `diffusion/model/ops/frame_gdn/fused_recurrent_triton.py` | 268 | Frame-wise GDN fused recurrent kernel (fwd only). No patches. |
| `ops/frame_gdn/scan_triton.py` | `diffusion/model/ops/frame_gdn/scan_triton.py` | 461 | D×D state scan (fwd + bwd). No patches. |

## Pending vendoring (Phase 2B–2D)

- `nets/sana_gdn_blocks_triton.py`
- `nets/sana_gdn_camctrl_blocks.py`
- `nets/sana_camctrl_blocks.py`
- `nets/basic_modules.py`
- `nets/sana_blocks.py`
- `nets/sana_multi_scale.py`
- `nets/sana_multi_scale_video_camctrl.py`

## Patch policy

Vendored files must remain functionally equivalent to upstream. Only changes
allowed:

1. Import rewrites: `from diffusion.model.x import ...` → relative imports.
2. Stub-replacement for non-essential helpers (`get_rank`, `is_xformers_available`)
   → inline noop versions.

Anything beyond that is a real divergence and must be flagged in this file
with a `## Divergences` section + rationale.
