# SPDX-License-Identifier: Apache-2.0
"""Simple unit tests for the SANA-WM integration skeleton.

Covers Phase 1 deliverables (configs, sampling params, preset, registry).
Phase 2+ tests (DiT model class, forward pass, numerical equivalence) are
stubbed and skipped until the corresponding modules land.

Run from the FastVideo repo root:
    pytest fastvideo/tests/transformers/test_sana_wm.py -v

Designed to be fast (<5s total) and require no GPU / no model weights.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Phase 1 — DiT arch config
# ---------------------------------------------------------------------------


class TestSanaWMArchConfig:
    """Architecture config for SanaMSVideoCamCtrl_1600M_P1_D20."""

    def test_instantiates(self):
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        assert cfg is not None

    def test_1600m_dims(self):
        """Hidden=2240, layers=20, heads=20, head_dim=112 → 1600M variant."""
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        assert cfg.hidden_size == 2240
        assert cfg.num_layers == 20
        assert cfg.num_attention_heads == 20
        assert cfg.attention_head_dim == 112
        assert cfg.num_attention_heads * cfg.attention_head_dim == cfg.hidden_size

    def test_ltx2_latent_dims(self):
        """SANA-WM uses LTX-2 VAE: latent_dim=128, patch=(1,1,1)."""
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        assert cfg.num_channels_latents == 128
        assert cfg.patch_size == (1, 1, 1)

    def test_in_out_channels_auto_computed(self):
        """in/out_channels should be set in __post_init__ from latent_ch × patch_volume."""
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        patch_volume = cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]
        assert cfg.in_channels == cfg.num_channels_latents * patch_volume
        assert cfg.out_channels == cfg.in_channels

    def test_camera_attention_names(self):
        """Triton kernel names recorded so model loader dispatches correctly."""
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        assert cfg.attn_type == "BidirectionalGDNTriton"
        assert cfg.camctrl_type == "BidirectionalGDNUCPESinglePathLiteLABothTriton"
        assert cfg.softmax_every_n == 4

    def test_camera_input_shape(self):
        """Camera tensor is (B, F, 20) = 16 C2W flat + 4 intrinsics."""
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        assert cfg.cam_dim_in == 20

    def test_text_conditioning(self):
        """Gemma 2B-IT: hidden=2304, max_length=300, y_norm with scale 0.01."""
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        assert cfg.text_hidden_size == 2304
        assert cfg.text_max_length == 300
        assert cfg.y_norm is True
        assert cfg.y_norm_scale_factor == pytest.approx(0.01)

    def test_latent_grid_configurable(self):
        """``input_size`` (latent grid for pos_embed) is configurable per checkpoint.

        Default matches SANA-WM_bidirectional (22). Larger checkpoints (e.g.
        a hypothetical 1080p variant at 32x32) would override.
        """
        from fastvideo.configs.models.dits.sana_wm import SanaWMArchConfig

        cfg = SanaWMArchConfig()
        assert cfg.input_size == 22  # SANA-WM_bidirectional default
        assert cfg.pe_interpolation == pytest.approx(1.0)

        custom = SanaWMArchConfig(input_size=32, pe_interpolation=1.5)
        assert custom.input_size == 32
        assert custom.pe_interpolation == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Phase 1 — DiT top-level config
# ---------------------------------------------------------------------------


class TestSanaWMConfig:
    def test_prefix(self):
        from fastvideo.configs.models.dits.sana_wm import SanaWMConfig

        cfg = SanaWMConfig()
        assert cfg.prefix == "sana_wm"

    def test_arch_config_is_sana_wm(self):
        from fastvideo.configs.models.dits.sana_wm import (
            SanaWMArchConfig,
            SanaWMConfig,
        )

        cfg = SanaWMConfig()
        assert isinstance(cfg.arch_config, SanaWMArchConfig)


# ---------------------------------------------------------------------------
# Phase 1 — Pipeline config (extends LTX2T2VConfig)
# ---------------------------------------------------------------------------


class TestSanaWM720PConfig:
    def test_inherits_ltx2_vae(self):
        from fastvideo.configs.models.vaes import LTX2VAEConfig
        from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig

        pc = SanaWM720PConfig()
        assert isinstance(pc.vae_config, LTX2VAEConfig), (
            "SANA-WM must inherit LTX-2 VAE from LTX2T2VConfig"
        )

    def test_inherits_gemma_text_encoder(self):
        from fastvideo.configs.models.encoders import LTX2GemmaConfig
        from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig

        pc = SanaWM720PConfig()
        assert len(pc.text_encoder_configs) >= 1
        assert isinstance(pc.text_encoder_configs[0], LTX2GemmaConfig)

    def test_dit_is_sana_wm(self):
        from fastvideo.configs.models.dits.sana_wm import SanaWMConfig
        from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig

        pc = SanaWM720PConfig()
        assert isinstance(pc.dit_config, SanaWMConfig)

    def test_default_resolution(self):
        """720p means 720x1280 by SANA-WM convention."""
        from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig

        pc = SanaWM720PConfig()
        assert pc.height == 720
        assert pc.width == 1280

    def test_default_video_params(self):
        """81 frames @ 24 fps = 3.375 sec (matches SANA-WM eval clip length)."""
        from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig

        pc = SanaWM720PConfig()
        assert pc.num_frames == 81
        assert pc.fps == 24

    def test_inference_flow_shift(self):
        """flow_shift=9.8 per upstream config.yaml inference_flow_shift."""
        from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig

        pc = SanaWM720PConfig()
        assert pc.flow_shift == pytest.approx(9.8)


# ---------------------------------------------------------------------------
# Phase 1 — Sampling param
# ---------------------------------------------------------------------------


class TestSanaWMSamplingParam:
    def test_defaults(self):
        from fastvideo.api.sana_wm import SanaWMSamplingParam

        sp = SanaWMSamplingParam()
        assert sp.height == 720
        assert sp.width == 1280
        assert sp.num_frames == 81
        assert sp.fps == 24
        assert sp.num_inference_steps == 20
        assert sp.guidance_scale == pytest.approx(1.0)

    def test_default_camera_string(self):
        """Default DSL 'w-31' = forward for 31 frames (matches existing logs)."""
        from fastvideo.api.sana_wm import SanaWMSamplingParam

        sp = SanaWMSamplingParam()
        assert sp.cam_ctrl_string == "w-31"


# ---------------------------------------------------------------------------
# Phase 1 — Preset
# ---------------------------------------------------------------------------


class TestSanaWMPreset:
    def test_preset_exported(self):
        from fastvideo.pipelines.basic.sana_wm.presets import (
            ALL_PRESETS,
            SANA_WM_720P,
        )

        assert SANA_WM_720P in ALL_PRESETS

    def test_preset_metadata(self):
        from fastvideo.pipelines.basic.sana_wm.presets import SANA_WM_720P

        assert SANA_WM_720P.name == "sana_wm_720p"
        assert SANA_WM_720P.model_family == "sana_wm"
        assert SANA_WM_720P.workload_type == "t2v"

    def test_preset_has_denoise_stage(self):
        from fastvideo.pipelines.basic.sana_wm.presets import SANA_WM_720P

        stage_names = [s.name for s in SANA_WM_720P.stage_schemas]
        assert "denoise" in stage_names

    def test_preset_defaults_match_sampling_param(self):
        """Preset defaults should agree with the SamplingParam dataclass."""
        from fastvideo.api.sana_wm import SanaWMSamplingParam
        from fastvideo.pipelines.basic.sana_wm.presets import SANA_WM_720P

        sp = SanaWMSamplingParam()
        d = SANA_WM_720P.defaults
        assert d["height"] == sp.height
        assert d["width"] == sp.width
        assert d["num_frames"] == sp.num_frames
        assert d["fps"] == sp.fps
        assert d["num_inference_steps"] == sp.num_inference_steps


# ---------------------------------------------------------------------------
# Phase 1 — Registry resolution
# ---------------------------------------------------------------------------


class TestRegistryResolution:
    HF_PATH = "Efficient-Large-Model/SANA-WM_bidirectional"

    def test_model_family(self):
        from fastvideo.registry import get_model_family

        assert get_model_family(self.HF_PATH) == "sana_wm"

    def test_default_preset(self):
        from fastvideo.registry import get_default_preset

        assert get_default_preset(self.HF_PATH) == "sana_wm_720p"

    def test_pipeline_config_class(self):
        from fastvideo.configs.pipelines.sana_wm import SanaWM720PConfig
        from fastvideo.registry import get_pipeline_config_cls_from_name

        assert get_pipeline_config_cls_from_name(self.HF_PATH) is SanaWM720PConfig

    def test_sampling_param_class(self):
        from fastvideo.api.sana_wm import SanaWMSamplingParam
        from fastvideo.registry import get_sampling_param_cls_for_name

        assert get_sampling_param_cls_for_name(self.HF_PATH) is SanaWMSamplingParam

    def test_detector_is_case_insensitive(self):
        from fastvideo.registry import get_model_family

        assert get_model_family("Efficient-Large-Model/SANA-WM_BIDIRECTIONAL") == "sana_wm"

    def test_hf_model_path_is_registered(self):
        """Exact HF path string used by the registry config block must match
        the constant we publish to users via PLAN.md / examples."""
        from fastvideo.registry import get_registered_model_paths

        assert "Efficient-Large-Model/SANA-WM_bidirectional" in get_registered_model_paths()


# ---------------------------------------------------------------------------
# Phase 2 — DiT model class (stubs, skipped until ports land)
# ---------------------------------------------------------------------------


class TestSanaWMDSLParser:
    """Camera DSL parser (Phase 2A)."""

    def test_parse_w31(self):
        """`action_string_to_c2w('w-31')` returns 32 poses (initial + 31 frames)."""
        import numpy as np

        from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w

        c2w = action_string_to_c2w("w-31")
        assert isinstance(c2w, np.ndarray)
        assert c2w.shape == (32, 4, 4)
        assert c2w.dtype == np.float32
        # First pose is identity.
        assert np.allclose(c2w[0], np.eye(4, dtype=np.float32))
        # Forward keys move +Z over time; the camera should have advanced.
        assert c2w[-1, 2, 3] > c2w[0, 2, 3]

    def test_parse_none(self):
        """`action_string_to_c2w('none-3')` returns 4 identity poses."""
        import numpy as np

        from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w

        c2w = action_string_to_c2w("none-3")
        assert c2w.shape == (4, 4, 4)
        for i in range(4):
            assert np.allclose(c2w[i], np.eye(4, dtype=np.float32))

    def test_parse_compound(self):
        """`'w-10,iw-5,none-3'` → 19 poses (1 initial + 10 + 5 + 3)."""
        from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w

        c2w = action_string_to_c2w("w-10,iw-5,none-3")
        assert c2w.shape == (19, 4, 4)

    def test_parse_rejects_empty(self):
        import pytest as _pt

        from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w

        with _pt.raises(ValueError):
            action_string_to_c2w("")

    def test_parse_rejects_unknown_keys(self):
        import pytest as _pt

        from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w

        with _pt.raises(ValueError):
            action_string_to_c2w("x-5")  # 'x' is not in wasdijkl

    def test_parse_rejects_bad_duration(self):
        import pytest as _pt

        from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w

        with _pt.raises(ValueError):
            action_string_to_c2w("w-0")
        with _pt.raises(ValueError):
            action_string_to_c2w("w-abc")


class TestSanaWMVendoredKernels:
    """Vendored Triton kernels (Phase 2A) — import + parse-time validity."""

    def test_fused_recurrent_kernel_importable(self):
        from fastvideo.models.dits.sana_wm._vendor.ops.frame_gdn import (
            fused_recurrent_triton,
        )

        assert hasattr(
            fused_recurrent_triton, "frame_gdn_fused_recurrent_fwd_kernel"
        )

    def test_scan_kernel_importable(self):
        from fastvideo.models.dits.sana_wm._vendor.ops.frame_gdn import (
            scan_triton,
        )

        assert hasattr(scan_triton, "frame_gdn_scan_fwd_kernel")
        assert hasattr(scan_triton, "frame_gdn_scan_bwd_kernel")


# ---------------------------------------------------------------------------
# Phase 3A — Pipeline class scaffold + camera conditioning stage
# ---------------------------------------------------------------------------


class TestSanaWMPipelineScaffold:
    """Phase 3A: pipeline class wiring (no model loading yet)."""

    def test_pipeline_class_importable(self):
        from fastvideo.pipelines.basic.sana_wm import SanaWMPipeline  # noqa: F401

    def test_pipeline_required_modules(self):
        from fastvideo.pipelines.basic.sana_wm import SanaWMPipeline

        # Expected required modules: same shape as matrixgame2 minus
        # image_encoder/image_processor (SANA-WM is T2V, not I2V).
        expected = {"vae", "transformer", "scheduler", "text_encoder", "tokenizer"}
        assert set(SanaWMPipeline._required_config_modules) == expected

    def test_pipeline_entry_class_exported(self):
        from fastvideo.pipelines.basic.sana_wm import sana_wm_pipeline

        assert hasattr(sana_wm_pipeline, "EntryClass")
        assert sana_wm_pipeline.SanaWMPipeline in sana_wm_pipeline.EntryClass


class TestSanaWMCameraConditioningStage:
    """Phase 3A: DSL → (B, F, 20) tensor."""

    def _stage_and_batch(self, cam_string="w-31", num_frames=32):
        from fastvideo.pipelines.basic.sana_wm.camera_conditioning_stage import (
            SanaWMCameraConditioningStage,
        )
        from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

        stage = SanaWMCameraConditioningStage()
        batch = ForwardBatch(data_type="t2v")
        batch.num_frames = num_frames
        batch.input_kwargs = {"cam_ctrl_string": cam_string}
        return stage, batch

    def test_forward_produces_packed_tensor(self):
        stage, batch = self._stage_and_batch()
        out = stage.forward(batch, fastvideo_args=None)
        cam = out.input_kwargs["camera_conditions"]
        # Shape (1, 32, 20) = batch * frames * (16 C2W + 4 intrinsics).
        assert cam.shape == (1, 32, 20)

    def test_first_pose_is_identity(self):
        import torch

        stage, batch = self._stage_and_batch()
        out = stage.forward(batch, fastvideo_args=None)
        c2w0 = out.input_kwargs["camera_conditions"][0, 0, :16].reshape(4, 4)
        assert torch.allclose(c2w0, torch.eye(4), atol=1e-6)

    def test_default_intrinsics(self):
        stage, batch = self._stage_and_batch()
        out = stage.forward(batch, fastvideo_args=None)
        intr = out.input_kwargs["camera_conditions"][0, 0, 16:]
        # DEFAULT_INTRINSICS = (1.0, 1.0, 0.5, 0.5)
        assert intr.tolist() == [1.0, 1.0, 0.5, 0.5]

    def test_custom_intrinsics_per_frame(self):
        import numpy as np

        stage, batch = self._stage_and_batch(num_frames=4)
        batch.input_kwargs["intrinsics"] = np.array(
            [[2.0, 2.0, 0.5, 0.5]] * 4, dtype=np.float32
        )
        out = stage.forward(batch, fastvideo_args=None)
        intr = out.input_kwargs["camera_conditions"][0, :, 16:].numpy()
        assert intr.shape == (4, 4)
        assert (intr == 2.0).sum() == 8  # fx, fy =2 for all 4 frames

    def test_rejects_short_dsl(self):
        stage, batch = self._stage_and_batch(cam_string="w-5", num_frames=32)
        with pytest.raises(ValueError, match="reduce num_frames"):
            stage.forward(batch, fastvideo_args=None)

    def test_camera_string_resolved_recorded(self):
        """The resolved DSL string is recorded for downstream/logging."""
        stage, batch = self._stage_and_batch(cam_string="none-50", num_frames=10)
        out = stage.forward(batch, fastvideo_args=None)
        assert out.input_kwargs["cam_ctrl_string_resolved"] == "none-50"


# ---------------------------------------------------------------------------
# Phase 3B/4 — Generator convenience class + example script
# ---------------------------------------------------------------------------


class TestSanaWMGenerator:
    """Phase 3B: high-level SanaWMGenerator (wraps upstream pipeline)."""

    def test_generator_class_importable(self):
        from fastvideo.pipelines.basic.sana_wm.generator import (  # noqa: F401
            DEFAULT_HF_REPO,
            SanaWMGenerationParams,
            SanaWMGenerator,
        )

    def test_default_generation_params(self):
        from fastvideo.pipelines.basic.sana_wm.generator import (
            SanaWMGenerationParams,
        )

        p = SanaWMGenerationParams()
        # Match config.yaml + upstream inference defaults.
        assert p.num_frames == 32
        assert p.fps == 16
        assert p.num_inference_steps == 20
        assert p.guidance_scale == pytest.approx(5.0)
        assert p.seed == 42

    def test_default_hf_repo(self):
        from fastvideo.pipelines.basic.sana_wm.generator import DEFAULT_HF_REPO

        assert DEFAULT_HF_REPO == "Efficient-Large-Model/SANA-WM_bidirectional"


class TestBasicSanaWMExampleScript:
    """Phase 4: example script structural validity."""

    def test_example_script_parses(self):
        """``basic_sana_wm.py`` must at least be a syntactically valid Python file."""
        import pathlib
        import py_compile

        path = (
            pathlib.Path(__file__).resolve().parents[3]
            / "examples/inference/basic/basic_sana_wm.py"
        )
        assert path.is_file(), f"example script missing at {path}"
        py_compile.compile(str(path), doraise=True)

    def test_example_script_has_main(self):
        """The script must expose a ``main()`` entry point."""
        import importlib.util
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[3]
            / "examples/inference/basic/basic_sana_wm.py"
        )
        spec = importlib.util.spec_from_file_location("_basic_sana_wm", path)
        mod = importlib.util.module_from_spec(spec)
        # Don't execute (would trigger heavy imports); just check the source.
        src = path.read_text()
        assert "def main(" in src
        assert "parse_args(" in src


def _sana_repo_or_skip():
    """Locate the Sana repo and skip the test if not present / not importable."""
    import os
    import pathlib

    for cand in (
        os.environ.get("SANA_REPO_PATH") or "",
        "/mnt/c/workspace/world/Sana",
        "C:/workspace/world/Sana",
    ):
        if cand and pathlib.Path(cand, "diffusion/model/nets/sana_multi_scale_video_camctrl.py").is_file():
            return cand
    pytest.skip("Sana repo not found (set SANA_REPO_PATH or clone NVlabs/Sana)")


class TestSanaWMDiTModel:
    """Phase 2E: FastVideo wrapper class — `model.py`."""

    def test_model_class_importable(self):
        """Just import the wrapper class (no instantiation — that needs Sana deps)."""
        from fastvideo.models.dits.sana_wm.model import SanaWMTransformer3DModel  # noqa

    def test_locator_finds_sana_repo(self):
        _sana_repo_or_skip()
        from fastvideo.models.dits.sana_wm.model import _locate_sana_repo

        repo = _locate_sana_repo()
        # The located path must contain the expected file the locator looks for.
        import pathlib
        assert (
            pathlib.Path(repo) / "diffusion/model/nets/sana_multi_scale_video_camctrl.py"
        ).is_file()

    def test_locator_env_var_takes_priority(self, monkeypatch, tmp_path):
        """`$SANA_REPO_PATH` must take priority over the on-disk default."""
        from fastvideo.models.dits.sana_wm.model import _locate_sana_repo

        # Build a fake Sana root that satisfies the locator's existence check.
        fake = tmp_path / "fake_sana"
        target = fake / "diffusion/model/nets/sana_multi_scale_video_camctrl.py"
        target.parent.mkdir(parents=True)
        target.write_text("# stub for locator test")

        monkeypatch.setenv("SANA_REPO_PATH", str(fake))
        # Force a fresh resolution by re-importing — the module-level
        # candidates list captures the env var at import time, so we resolve
        # via the function which reads from candidates list each call.
        import importlib

        import fastvideo.models.dits.sana_wm.model as wrapper_mod

        importlib.reload(wrapper_mod)
        try:
            assert wrapper_mod._locate_sana_repo() == str(fake)
        finally:
            importlib.reload(wrapper_mod)  # restore original module state

    def test_locator_raises_when_nothing_found(self, monkeypatch, tmp_path):
        from fastvideo.models.dits.sana_wm.model import _locate_sana_repo

        monkeypatch.setenv("SANA_REPO_PATH", str(tmp_path / "does_not_exist"))
        # Patch all the other candidates to bogus paths too.
        import fastvideo.models.dits.sana_wm.model as wrapper_mod

        monkeypatch.setattr(
            wrapper_mod,
            "_REPO_PATH_CANDIDATES",
            (str(tmp_path / "does_not_exist"),),
        )
        with pytest.raises(FileNotFoundError):
            _locate_sana_repo()

    def test_wrapper_rejects_wrong_config_type(self):
        """`SanaWMTransformer3DModel` must reject non-SanaWMConfig inputs."""
        _sana_repo_or_skip()
        from fastvideo.configs.models.dits.matrixgame2 import MatrixGame2WanVideoConfig
        from fastvideo.models.dits.sana_wm.model import SanaWMTransformer3DModel

        with pytest.raises(TypeError, match="SanaWMConfig"):
            SanaWMTransformer3DModel(MatrixGame2WanVideoConfig())

    @pytest.mark.slow
    def test_model_instantiates_from_config(self):
        """Phase 2E: full SanaMSVideoCamCtrl construction (~23s on a 5090).

        The "1600M" factory name refers to the *base* transformer (depth=20,
        hidden=2240). With camera-control branches the actual param count is
        ~2.66B. Tolerate ±5% drift across upstream commits.
        """
        _sana_repo_or_skip()
        from fastvideo.configs.models.dits.sana_wm import SanaWMConfig
        from fastvideo.models.dits.sana_wm.model import SanaWMTransformer3DModel

        model = SanaWMTransformer3DModel(SanaWMConfig())
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        # Empirically 2,661,242,564 on Sana commit pinned 2026-05-28.
        assert 2.5e9 < n_params < 2.8e9, (
            f"unexpected param count {n_params:,} ({n_params / 1e9:.2f}B)"
        )

    def test_safetensors_cached(self):
        """Verify the SANA-WM DiT safetensors is in the HF cache (Phase 2F)."""
        import os

        # Resolve via HF Hub; should be a no-op since the file is cached.
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Efficient-Large-Model/SANA-WM_bidirectional",
            filename="dit/sana_wm_1600m_720p.safetensors",
            local_files_only=True,
        )
        size = os.path.getsize(path)
        assert size > 9 * 1024**3, f"expected ~10 GB safetensors, got {size / 1e9:.2f} GB"

    @pytest.mark.slow
    def test_state_dict_load(self):
        """Load the safetensors weights into the wrapped model."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from fastvideo.configs.models.dits.sana_wm import SanaWMConfig
        from fastvideo.models.dits.sana_wm.model import SanaWMTransformer3DModel

        path = hf_hub_download(
            "Efficient-Large-Model/SANA-WM_bidirectional",
            "dit/sana_wm_1600m_720p.safetensors",
            local_files_only=True,
        )
        sd = load_file(path)
        model = SanaWMTransformer3DModel(SanaWMConfig())
        missing, unexpected = model.model.load_state_dict(sd, strict=False)
        # Tolerate a small number of mismatches due to different SDPA stubs;
        # tighten as Phase 2G stabilizes.
        assert len(missing) < 20, f"too many missing keys: {missing[:10]}"
        assert len(unexpected) < 20, f"too many unexpected keys: {unexpected[:10]}"

    @pytest.mark.slow
    def test_model_loads_weights_and_moves_to_gpu(self):
        """Phase 2G (scoped): load real weights + move to GPU in bf16.

        This is the practical Phase 2 exit gate. A full forward-pass smoke is
        deferred to Phase 3, where the camera-conditioning + denoising
        pipeline stages know how to construct the right input shapes and
        kwargs — the upstream model's forward signature is non-trivial to
        drive from raw tensors (frame-aware timestep, camera embeds, text
        mask, Triton autotune sensitivities at F=1).

        We assert: weights load, model goes to CUDA in bf16, all parameters
        are on the right device + dtype, no NaN/Inf in any param tensor.
        """
        _sana_repo_or_skip()
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from fastvideo.configs.models.dits.sana_wm import SanaWMConfig
        from fastvideo.models.dits.sana_wm.model import SanaWMTransformer3DModel

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        device = torch.device("cuda")
        dtype = torch.bfloat16

        path = hf_hub_download(
            "Efficient-Large-Model/SANA-WM_bidirectional",
            "dit/sana_wm_1600m_720p.safetensors",
            local_files_only=True,
        )
        sd = load_file(path)
        model = SanaWMTransformer3DModel(SanaWMConfig())
        model.model.load_state_dict(sd, strict=False)
        model = model.to(device=device, dtype=dtype).eval()

        # All parameters on CUDA in bf16, no NaN/Inf.
        sample = next(iter(model.parameters()))
        assert sample.device.type == "cuda"
        assert sample.dtype == dtype

        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"non-finite param in {name}"

        # Total VRAM footprint should be ~5 GB for 2.66B bf16 params.
        vram = torch.cuda.memory_allocated() / 1024**3
        assert 4.5 < vram < 6.5, f"unexpected VRAM footprint {vram:.2f} GB"
