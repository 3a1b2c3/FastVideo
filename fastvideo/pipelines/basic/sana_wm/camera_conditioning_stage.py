# SPDX-License-Identifier: Apache-2.0
"""SANA-WM camera conditioning pipeline stage.

Parses the SANA-WM camera-control DSL (e.g. ``"w-31"``) into a per-frame
camera-to-world (C2W) trajectory, then packs each frame's pose + intrinsics
into a ``(B, F, 20)`` tensor that the upstream ``SanaMSVideoCamCtrl`` forward
expects under ``camera_conditions``.

Layout of the trailing 20-dim:
    [0..15]: 4x4 C2W matrix flattened (row-major)
    [16..19]: intrinsics [fx, fy, cx, cy]

This stage is FastVideo-side; it has no model deps. The downstream UCPE
expansion to ray-matrix + absmap is done inside the upstream DiT.
"""

from __future__ import annotations

import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.sana_wm.utils import action_string_to_c2w
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import V, VerificationResult

logger = init_logger(__name__)

# Identity intrinsics in normalized image coordinates — fx=fy=1, cx=cy=0.5.
# Callers may override per-frame by writing to ``batch.input_kwargs["intrinsics"]``
# as an ``(F, 4)`` array of ``[fx, fy, cx, cy]``.
DEFAULT_INTRINSICS: tuple[float, float, float, float] = (1.0, 1.0, 0.5, 0.5)


class SanaWMCameraConditioningStage(PipelineStage):
    """Build the ``(B, F, 20)`` camera tensor SANA-WM expects.

    Reads ``batch.input_kwargs["cam_ctrl_string"]`` (or falls back to
    ``"w-31"``), rolls the DSL out via ``action_string_to_c2w``, slices to
    ``batch.num_frames`` poses, optionally combines with per-frame intrinsics,
    and writes the packed tensor to ``batch.input_kwargs["camera_conditions"]``.

    No model deps; runs on CPU and lets the downstream denoising stage move
    the tensor to the GPU at forward time.
    """

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        input_kwargs = batch.input_kwargs or {}
        cam_string: str = input_kwargs.get("cam_ctrl_string", "w-31")
        num_frames: int = batch.num_frames or 32

        # `action_string_to_c2w` returns (N+1, 4, 4) where N = sum of DSL durations.
        # The +1 is the initial identity pose, which we keep as the first frame.
        c2w = action_string_to_c2w(cam_string)
        if c2w.shape[0] < num_frames:
            raise ValueError(
                f"Camera DSL {cam_string!r} produced {c2w.shape[0]} poses but "
                f"pipeline requested {num_frames} frames. Extend the DSL or "
                f"reduce num_frames."
            )
        c2w = c2w[:num_frames]  # (F, 4, 4)

        intrinsics = input_kwargs.get("intrinsics")
        if intrinsics is None:
            intrinsics_arr = np.broadcast_to(
                np.array(DEFAULT_INTRINSICS, dtype=np.float32), (num_frames, 4)
            )
        else:
            intrinsics_arr = np.asarray(intrinsics, dtype=np.float32)
            if intrinsics_arr.shape == (4,):
                intrinsics_arr = np.broadcast_to(intrinsics_arr, (num_frames, 4))
            if intrinsics_arr.shape != (num_frames, 4):
                raise ValueError(
                    f"intrinsics must be shape (4,) or ({num_frames}, 4); got "
                    f"{intrinsics_arr.shape}"
                )

        # Pack to (F, 20) = 16 C2W flat + 4 intrinsics.
        c2w_flat = c2w.reshape(num_frames, 16)
        packed = np.concatenate([c2w_flat, intrinsics_arr], axis=1)  # (F, 20)
        cam_tensor = torch.from_numpy(packed).unsqueeze(0).contiguous()  # (1, F, 20)

        if batch.input_kwargs is None:
            batch.input_kwargs = {}
        batch.input_kwargs["camera_conditions"] = cam_tensor
        batch.input_kwargs["cam_ctrl_string_resolved"] = cam_string

        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        kwargs = batch.input_kwargs or {}
        cam = kwargs.get("camera_conditions")
        result.add_check("camera_conditions", cam, V.is_tensor)
        result.add_check(
            "camera_conditions_shape",
            cam,
            lambda t: t is not None and t.ndim == 3 and t.shape[-1] == 20,
        )
        return result
