# SPDX-License-Identifier: Apache-2.0
"""SANA-WM (bidirectional) inference pipeline.

Phase 3A: structural wiring + camera-conditioning stage. Component loading
(transformer, VAE, text encoder) lands in Phase 3B; the denoising stage
needs camera_conditions plumbed through to the DiT forward, which lands in
Phase 3C alongside the first real generation smoke.

Stage order:
    input_validation_stage
    prompt_encoding_stage     (Gemma 2B-IT — inherited from LTX2 path)
    camera_conditioning_stage (NEW — SANA-WM specific)
    timestep_preparation_stage
    latent_preparation_stage
    denoising_stage           (will need camera_conditions kwarg in 3C)
    decoding_stage            (LTX-2 VAE)
"""

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.sana_wm.camera_conditioning_stage import (
    SanaWMCameraConditioningStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)

logger = init_logger(__name__)


class SanaWMPipeline(ComposedPipelineBase):
    """SANA-WM bidirectional T2V pipeline with camera control."""

    _required_config_modules = [
        "vae",
        "transformer",
        "scheduler",
        "text_encoder",
        "tokenizer",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        # The scheduler choice (FlowMatchEuler with shift=9.8) is inherited
        # from LTX2T2VConfig — no override needed here. Override below if a
        # SANA-WM-specific scheduler emerges from upstream.
        pass

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        if (
            self.get_module("text_encoder", None) is not None
            and self.get_module("tokenizer", None) is not None
        ):
            self.add_stage(
                stage_name="prompt_encoding_stage",
                stage=TextEncodingStage(
                    text_encoders=[self.get_module("text_encoder")],
                    tokenizers=[self.get_module("tokenizer")],
                ),
            )

        # SANA-WM specific: DSL → (B, F, 20) camera tensor that downstream
        # denoising forwards as ``camera_conditions=`` to the DiT.
        self.add_stage(
            stage_name="camera_conditioning_stage",
            stage=SanaWMCameraConditioningStage(),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


EntryClass = [SanaWMPipeline]
