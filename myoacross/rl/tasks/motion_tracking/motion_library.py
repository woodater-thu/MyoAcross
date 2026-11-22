"""Validated, packed reference clips for one source subject."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch


_CPU_MOTION_CLIPS: dict[Path, dict[str, np.ndarray]] = {}


def _read_motion_clip(path: Path) -> dict[str, np.ndarray]:
    """Decode and screen each CPU clip once at its input boundary."""
    with np.load(path, allow_pickle=False) as archive:
        fps_values = np.asarray(archive["fps"])
        if fps_values.size != 1:
            raise ValueError(f"{path}: motion fps must contain one value")
        fps = float(fps_values.reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path}: invalid motion fps")
        clip = {"fps": fps_values}
        for name in (
            "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        ):
            values = np.asarray(archive[name], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {name}: nonfinite motion values")
            values.setflags(write=False)
            clip[name] = values
    joint_pos = clip["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[0] < 2:
        raise ValueError(f"{path}: motion needs at least two joint frames")
    return clip


def clear_motion_clip_cache() -> None:
    """Release the current training process's CPU reference arrays."""
    _CPU_MOTION_CLIPS.clear()


def preload_motion_clips(paths: Iterable[str | Path]) -> None:
    """Cache only the selected training clips; keep GPU libraries separate."""
    clear_motion_clip_cache()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path not in _CPU_MOTION_CLIPS:
            _CPU_MOTION_CLIPS[path] = _read_motion_clip(path)


class MotionLibrary:
    """Store each clip once; environments index it by clip and local frame."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        body_indexes: torch.Tensor,
        *,
        num_joints: int,
        num_bodies: int,
        step_dt: float,
        device: str,
    ) -> None:
        paths = tuple(Path(path).expanduser().resolve() for path in paths)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Motion paths must be nonempty and unique")
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Environment step_dt must be positive")

        shapes = {
            "joint_pos": (num_joints,),
            "joint_vel": (num_joints,),
            "body_pos_w": (num_bodies, 3),
            "body_quat_w": (num_bodies, 4),
            "body_lin_vel_w": (num_bodies, 3),
            "body_ang_vel_w": (num_bodies, 3),
        }
        clips = {name: [] for name in shapes}
        lengths = []
        selected = body_indexes.detach().cpu().numpy()
        for path in paths:
            data = _CPU_MOTION_CLIPS.get(path)
            if data is None:
                data = _read_motion_clip(path)
            fps = float(data["fps"].reshape(-1)[0])
            if not np.isclose(fps * step_dt, 1.0, atol=1e-8, rtol=1e-6):
                raise ValueError(f"{path}: motion fps differs from control frequency")
            count = data["joint_pos"].shape[0]
            for name, suffix in shapes.items():
                values = data[name]
                expected = (count, *suffix)
                if values.shape != expected:
                    raise ValueError(f"{path}: {name}: expected {expected}, got {values.shape}")
                if name.startswith("body_"):
                    values = values[:, selected]
                clips[name].append(values)
            lengths.append(count)

        # Validate every CPU clip before allocating the replacement library.
        for name, values in clips.items():
            setattr(self, name, torch.as_tensor(
                np.concatenate(values), dtype=torch.float32, device=device,
            ))
        self.paths = paths
        self.num_motions = len(paths)
        self.fps = 1.0 / step_dt
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.offsets = self.lengths.cumsum(0) - self.lengths
