# SPDX-License-Identifier: Apache-2.0
"""SANA-WM camera DSL parser → camera-to-world (C2W) trajectory.

Ported (not vendored) from
``Sana/inference_video_scripts/inference_sana_wm.py``. The DSL groups segments
as ``<keys>-<frames>`` joined by commas. ``"none"`` means no keys held.
Movement keys (``wasd``) translate on the world XZ plane; rotation keys
(``ijkl``) apply pitch / yaw. Coordinate convention: OpenCV
(``+X right, +Y down, +Z forward``).

Examples:
    "w-31"             → 32 poses (initial + 31 forward steps)
    "w-10,iw-5,none-3" → 19 poses (10 fwd, then 5 fwd+pitch_up, then 3 hold)
    "none-1"           → 2 poses (initial + 1 frame with no movement)
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_TRANSLATION_SPEED: float = 0.05
DEFAULT_ROTATION_SPEED_DEG: float = 1.2
DEFAULT_PITCH_LIMIT_DEG: float = 85.0
ALLOWED_ACTION_KEYS: frozenset[str] = frozenset("wasdijkl")


def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def parse_action_string(action: str) -> list[list[str]]:
    """``"w-10,iw-5,none-3"`` → list of per-frame held-key lists."""
    cleaned = "".join(action.replace("，", ",").split())
    if not cleaned:
        raise ValueError("action string is empty")
    per_frame: list[list[str]] = []
    for segment in cleaned.split(","):
        if not segment or "-" not in segment:
            raise ValueError(
                f"Invalid action segment {segment!r}: expected '<keys>-<duration>'."
            )
        keys_part, dur_str = segment.rsplit("-", 1)
        if not dur_str.isdigit() or int(dur_str) <= 0:
            raise ValueError(
                f"Action segment {segment!r} has a non-positive duration {dur_str!r}."
            )
        n = int(dur_str)
        keys_lower = keys_part.lower()
        if keys_lower == "none":
            keys: list[str] = []
        else:
            bad = sorted({c for c in keys_lower if c not in ALLOWED_ACTION_KEYS})
            if bad:
                raise ValueError(
                    f"Action segment {segment!r} contains unknown keys {bad}; "
                    f"allowed: {''.join(sorted(ALLOWED_ACTION_KEYS))}."
                )
            keys = sorted(set(keys_lower))
        per_frame.extend([list(keys) for _ in range(n)])
    return per_frame


def action_string_to_c2w(
    action: str,
    translation_speed: float = DEFAULT_TRANSLATION_SPEED,
    rotation_speed_deg: float = DEFAULT_ROTATION_SPEED_DEG,
    pitch_limit_deg: float = DEFAULT_PITCH_LIMIT_DEG,
) -> np.ndarray:
    """Roll out an ``(N+1, 4, 4)`` camera-to-world trajectory from an action string.

    The DSL groups segments as ``<keys>-<frames>`` joined by commas. ``"none"``
    means no keys held. Movement keys (``wasd``) translate on the world XZ
    plane; rotation keys (``ijkl``) apply pitch / yaw.
    """
    per_frame = parse_action_string(action)
    rotate_rad = math.radians(rotation_speed_deg)
    pitch_limit_rad = math.radians(pitch_limit_deg)
    current = np.eye(4, dtype=np.float64)
    poses = [current.copy()]
    current_pitch = 0.0

    for keys in per_frame:
        held = set(keys)
        R = current[:3, :3]
        T_ = current[:3, 3]

        pitch_delta = (rotate_rad if "i" in held else 0.0) - (
            rotate_rad if "k" in held else 0.0
        )
        new_pitch = current_pitch + pitch_delta
        if not (-pitch_limit_rad <= new_pitch <= pitch_limit_rad):
            pitch_delta = 0.0
        else:
            current_pitch = new_pitch

        yaw_delta = (rotate_rad if "l" in held else 0.0) - (
            rotate_rad if "j" in held else 0.0
        )
        R_new = _rot_y(yaw_delta) @ R @ _rot_x(pitch_delta)

        forward = R_new[:, 2].copy()
        forward[1] = 0.0
        right = R_new[:, 0].copy()
        right[1] = 0.0
        fn = float(np.linalg.norm(forward))
        rn = float(np.linalg.norm(right))
        if fn > 0:
            forward /= fn + 1e-6
        if rn > 0:
            right /= rn + 1e-6
        move = np.zeros(3, dtype=np.float64)
        if "w" in held:
            move += forward * translation_speed
        if "s" in held:
            move -= forward * translation_speed
        if "d" in held:
            move += right * translation_speed
        if "a" in held:
            move -= right * translation_speed

        current = np.eye(4, dtype=np.float64)
        current[:3, :3] = R_new
        current[:3, 3] = T_ + move
        poses.append(current.copy())

    return np.stack(poses, axis=0).astype(np.float32)
