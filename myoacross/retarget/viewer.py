"""Replay exported motion using an explicitly supplied embodiment."""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from myoacross.retarget.export import get_mjlab_layout


def view_motion(
    xml_path: str | Path,
    motion_path: str | Path,
    *,
    fps: float = 60.0,
    speed: float = 1.0,
    loop: bool = True,
) -> None:
    """
    Replay motion configurations without advancing dynamics.

    Args:
        xml_path: The model used to export the motion.
        motion_path: Exported motion.npz in mjlab Entity order.
        fps: Display refresh rate in Hz.
        speed: Playback speed multiplier.
        loop: Restart after the last frame.

    The model and mjlab ordering must match those used during export.
    """

    xml_path = Path(xml_path).expanduser().resolve()
    motion_path = Path(motion_path).expanduser().resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(f"Model XML not found: {xml_path}")
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion not found: {motion_path}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Display FPS must be finite and positive")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be finite and positive")

    # Reconstruct the same Entity ordering used by the exporter.
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    layout = get_mjlab_layout(xml_path)
    joint_names = layout["joint_names"]
    body_names = layout["body_names"]

    with np.load(motion_path, allow_pickle=False) as motion:
        required = {
            "fps",
            "joint_pos",
            "body_pos_w",
            "body_quat_w",
        }
        missing = required - set(motion.files)
        if missing:
            raise ValueError(f"Missing motion fields: {sorted(missing)}")

        motion_fps = float(motion["fps"].item())
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64).copy()
        body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64).copy()
        body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64).copy()

    if not math.isfinite(motion_fps) or motion_fps <= 0:
        raise ValueError("Motion FPS must be finite and positive")

    if joint_pos.ndim != 2 or len(joint_pos) == 0:
        raise ValueError("joint_pos must be a nonempty 2D array")

    num_frames = len(joint_pos)
    num_bodies = len(body_names)

    if (
        joint_pos.shape != (num_frames, len(joint_names))
        or body_pos.shape != (num_frames, num_bodies, 3)
        or body_quat.shape != (num_frames, num_bodies, 4)
    ):
        raise ValueError("Motion dimensions do not match the model Entity")

    for values in (joint_pos, body_pos, body_quat):
        if not np.isfinite(values).all():
            raise ValueError("Motion contains non-finite values")

    if not np.allclose(
        np.linalg.norm(body_quat, axis=-1),
        1.0,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("Body quaternions must have unit norm")

    # Locate the root freejoint and its body column.
    free_joint_ids = [
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError("Playback requires exactly one freejoint")

    root_joint_id = free_joint_ids[0]
    root_body_id = int(model.jnt_bodyid[root_joint_id])
    root_qadr = int(model.jnt_qposadr[root_joint_id])

    if model.body_parentid[root_body_id] != 0:
        raise ValueError("Root body must be attached to world")

    root_column = body_names.index(model.body(root_body_id).name)

    # Map Entity scalar-joint columns to MuJoCo qpos addresses.
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    scalar_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    expected_joint_ids = {
        i for i in range(model.njnt) if int(model.jnt_type[i]) in scalar_types
    }

    if (
        any(i < 0 for i in joint_ids)
        or len(joint_ids) != len(set(joint_ids))
        or set(joint_ids) != expected_joint_ids
        or len(expected_joint_ids) + 1 != model.njnt
    ):
        raise ValueError("Expected one freejoint and complete scalar-joint coverage")

    q_addresses = model.jnt_qposadr[joint_ids]

    qpos = np.tile(model.qpos0, (num_frames, 1))
    qpos[:, q_addresses] = joint_pos
    qpos[:, root_qadr : root_qadr + 3] = body_pos[:, root_column]
    qpos[:, root_qadr + 3 : root_qadr + 7] = body_quat[:, root_column]

    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)

    print(
        f"Playing {num_frames} frames at {motion_fps:g} Hz; "
        f"display={fps:g} Hz, speed={speed:g}, loop={loop}",
        flush=True,
    )

    frame_interval = 1.0 / fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.perf_counter()

        while viewer.is_running():
            frame_start = time.perf_counter()
            frame_index = int((frame_start - start_time) * speed * motion_fps)

            if loop:
                frame_index %= num_frames
            else:
                frame_index = min(frame_index, num_frames - 1)

            with viewer.lock():
                data.qpos[:] = qpos[frame_index]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

            viewer.sync()

            remaining = frame_interval - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
