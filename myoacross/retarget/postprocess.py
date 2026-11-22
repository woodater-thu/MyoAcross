"""Postprocessing and storage of retargeted trajectories."""

import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def max_penetration_with_floor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    floor_name: str = "floor",
) -> tuple[float, int | None, int]:
    """
    Return the deepest detected floor contact penetration.

    Call mujoco.mj_kinematics and mujoco.mj_collision (or mj_forward)
    after setting data.qpos and before calling this function.

    Returns:
        penetration: Most negative contact distance, in meters;
                    zero if no penetrating floor contact is detected.
        geom_id: Non-floor geom involved in the deepest penetration,
                or None.
        floor_id: Floor geom ID.
    """

    floor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        floor_name,
    )
    if floor_id < 0:
        raise ValueError(f"No floor geom named {floor_name!r}")

    penetration = 0.0
    geom_id = None

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]

        if contact.geom1 == floor_id:
            other_geom_id = int(contact.geom2)
        elif contact.geom2 == floor_id:
            other_geom_id = int(contact.geom1)
        else:
            continue

        if contact.dist < penetration:
            penetration = float(contact.dist)
            geom_id = other_geom_id

    return penetration, geom_id, floor_id


def correct_ground(
    model: mujoco.MjModel,
    qpos: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Apply MuscleMimic's ground correction to a MyoFullBody motion.

    Assumes a horizontal floor at world z=0 and a leading root freejoint.
    Returns a corrected copy and a summary of vertical adjustments.
    """

    corrected = np.array(qpos, dtype=np.float64, copy=True)

    if corrected.ndim != 2 or corrected.shape[1] != model.nq or corrected.shape[0] == 0:
        raise ValueError(
            f"Expected nonempty qpos with shape (T, {model.nq}), got {corrected.shape}"
        )
    if not np.isfinite(corrected).all():
        raise ValueError("qpos contains non-finite values")

    # Match the root layout used by the upstream MyoFullBody pipeline.
    if (
        model.njnt == 0
        or int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE)
        or model.jnt_qposadr[0] != 0
    ):
        raise ValueError("Ground correction requires a leading root freejoint")

    floor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_id < 0:
        raise ValueError("No floor geom named 'floor'")

    non_floor_ids = [geom_id for geom_id in range(model.ngeom) if geom_id != floor_id]
    if not non_floor_ids:
        raise ValueError("No non-floor geoms found")

    # Moves down to eliminate the suspension
    data = mujoco.MjData(model)

    # Only positions and contacts are read here. mj_kinematics (+ mj_collision)
    # gives the same geom poses and contacts as mj_forward without evaluating
    # the 400+ muscles, tendons and constraint solver.
    lowest_geom_z = float("inf")
    for frame_qpos in corrected:
        data.qpos[:] = frame_qpos
        mujoco.mj_kinematics(model, data)
        lowest_geom_z = min(
            lowest_geom_z,
            float(np.min(data.geom_xpos[non_floor_ids, 2])),
        )

    downward_shift = max(0.0, lowest_geom_z)
    corrected[:, 2] -= downward_shift

    # Moves up to eliminate the penetration
    lifted_frames = 0
    max_upward_shift = 0.0

    for frame_index in range(len(corrected)):
        data.qpos[:] = corrected[frame_index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)

        penetration, _, _ = max_penetration_with_floor(model, data)

        if penetration < 0.0:
            upward_shift = -penetration
            corrected[frame_index, 2] += upward_shift
            lifted_frames += 1
            max_upward_shift = max(max_upward_shift, upward_shift)

    report = {
        "global_lowest_geom_z": lowest_geom_z,
        "downward_shift": downward_shift,
        "lifted_frames": lifted_frames,
        "max_upward_shift": max_upward_shift,
    }

    return corrected, report


def compute_qvel(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute velocities using MuscleMimic's central-difference scheme.

    Args:
        model: The embodiment model corresponding to qpos.
        qpos: Final postprocessed configurations, shape (T, nq).
        fps: Actual aligned frame rate returned by the motion loader.

    Returns:
        Trimmed qpos, shape (T - 2, nq).
        Central-difference qvel, shape (T - 2, nv).
    """

    qpos = np.asarray(qpos, dtype=np.float64)
    fps = float(fps)

    if qpos.ndim != 2 or qpos.shape[1] != model.nq or qpos.shape[0] < 3:
        raise ValueError(
            f"Expected qpos with shape (T, {model.nq}) and T >= 3, got {qpos.shape}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains non-finite values")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")

    dt = 1.0 / fps
    num_frames = len(qpos)

    forward_qvel = np.empty(
        (num_frames - 1, model.nv),
        dtype=np.float64,
    )

    for frame_index in range(num_frames - 1):
        mujoco.mj_differentiatePos(
            model,
            forward_qvel[frame_index],
            dt,
            qpos[frame_index],
            qpos[frame_index + 1],
        )

    qvel = 0.5 * (forward_qvel[:-1] + forward_qvel[1:])
    trimmed_qpos = qpos[1:-1].copy()

    if not np.isfinite(qvel).all():
        raise ValueError("Computed qvel contains non-finite values")

    return trimmed_qpos, qvel


def resample_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """
    Resample MyoFullBody configurations on a uniform time grid.

    Root translation and scalar joints use linear interpolation.
    Root orientation uses quaternion SLERP.

    Assumes one leading freejoint followed by hinge/slide joints.
    Source timing follows the supplied source_fps.
    """

    qpos = np.asarray(qpos, dtype=np.float64)
    source_fps = float(source_fps)
    target_fps = float(target_fps)

    # Validate inputs
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 3:
        raise ValueError(f"Expected qpos shape (T, {model.nq}) with T >= 3")

    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains non-finite values")

    if (
        not np.isfinite(source_fps)
        or source_fps <= 0
        or not np.isfinite(target_fps)
        or target_fps <= 0
    ):
        raise ValueError("Source and target FPS must be finite and positive")

    if (
        model.njnt == 0
        or int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE)
        or int(model.jnt_qposadr[0]) != 0
    ):
        raise ValueError("Expected a leading root freejoint")

    scalar_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    if any(int(kind) not in scalar_types for kind in model.jnt_type[1:]):
        raise ValueError("Only scalar joints may follow the root freejoint")

    if source_fps == target_fps:
        return qpos.copy()

    source_times = np.arange(len(qpos), dtype=np.float64) / source_fps
    duration = source_times[-1]

    output_count = int(np.floor(duration * target_fps)) + 1
    target_times = np.arange(output_count, dtype=np.float64) / target_fps

    # Avoid extrapolation, including floating-point endpoint overshoot.
    target_times = target_times[target_times <= duration]
    if len(target_times) < 3:
        raise ValueError("Resampled motion must contain at least three frames")

    output = np.empty((len(target_times), model.nq), dtype=np.float64)

    linear_columns = np.r_[:3, 7 : model.nq]
    interpolate = interp1d(
        source_times,
        qpos[:, linear_columns],
        axis=0,
        bounds_error=True,
    )
    output[:, linear_columns] = interpolate(target_times)

    # MuJoCo stores wxyz; SciPy defaults to xyzw.
    source_rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    interpolated_xyzw = Slerp(source_times, source_rotations)(target_times).as_quat()

    output[:, 3:7] = interpolated_xyzw[:, [3, 0, 1, 2]]

    return output
