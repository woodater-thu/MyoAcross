"""Export retargeted motion in mjlab tracking format."""

from collections.abc import Sequence
from functools import lru_cache

import mujoco
import numpy as np
from copy import deepcopy
import json
import os
import tempfile
from pathlib import Path
from mjlab.entity import EntityCfg
from importlib.metadata import version


THRESHOLDS = dict(
    foot_ik_p95_m=0.03,
    max_upward_shift_m=0.20,
    max_body_linear_speed=15.0,
    max_body_angular_speed=40.0,
    max_pelvis_step_m=0.08,
)


def get_mjlab_layout(robot_xml_path: str | Path) -> dict:
    """Read joint and body ordering from the official mjlab Entity."""

    robot_xml_path = Path(robot_xml_path).expanduser().resolve()
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Model XML not found: {robot_xml_path}")

    # Building the Entity compiles the XML; cache the ordering per file.
    return deepcopy(
        _get_mjlab_layout(
            str(robot_xml_path),
            robot_xml_path.stat().st_mtime_ns,
        )
    )


def load_model(robot_xml_path: str | Path) -> mujoco.MjModel:
    """Compile an embodiment XML once per process (read-only use)."""
    robot_xml_path = Path(robot_xml_path).expanduser().resolve()
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Model XML not found: {robot_xml_path}")
    return _load_model(str(robot_xml_path), robot_xml_path.stat().st_mtime_ns)


@lru_cache(maxsize=8)
def _get_mjlab_layout(robot_xml_path: str, mtime_ns: int) -> dict:
    robot_xml_path = Path(robot_xml_path)

    entity_cfg = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(robot_xml_path)),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )
    entity = entity_cfg.build()

    layout = {
        "joint_names": list(entity.joint_names),
        "body_names": list(entity.body_names),
        "mjlab_version": version("mjlab"),
        "mujoco_version": mujoco.__version__,
    }

    for field in ("joint_names", "body_names"):
        names = layout[field]
        if (
            not names
            or any(not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(f"Invalid mjlab layout field: {field}")

    return layout


@lru_cache(maxsize=2)
def _load_model(robot_xml_path: str, mtime_ns: int) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(robot_xml_path)


def build_mjlab_motion(
    model: mujoco.MjModel,
    result: dict,
    *,
    joint_names: Sequence[str],
    body_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """
    Build standard mjlab motion arrays in explicit Entity order.

    joint_names and body_names must come from the target mjlab Entity.
    The model must be the same subject model used for retargeting.
    """

    qpos = np.asarray(result["qpos"], dtype=np.float64)
    qvel = np.asarray(result["qvel"], dtype=np.float64)
    fps = float(result["fps"])

    # Validate inputs
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) == 0:
        raise ValueError(f"Expected nonempty qpos with width {model.nq}")
    if qvel.shape != (len(qpos), model.nv):
        raise ValueError(f"Expected qvel shape {(len(qpos), model.nv)}")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("Non-finite trajectory values")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")

    joint_names = tuple(joint_names)
    body_names = tuple(body_names)

    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("Joint names must be nonempty and unique")
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("Body names must be nonempty and unique")

    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ],
        dtype=np.int64,
    )

    body_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        ],
        dtype=np.int64,
    )

    if np.any(joint_ids < 0):
        raise ValueError("An Entity joint name is missing from the model")
    if np.any(body_ids <= 0):
        raise ValueError("Entity bodies must exist and exclude world")

    scalar_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    expected_joint_ids = {
        i for i in range(model.njnt) if int(model.jnt_type[i]) in scalar_types
    }

    if set(joint_ids.tolist()) != expected_joint_ids:
        raise ValueError("Entity joint names must cover all scalar joints")
    if set(body_ids.tolist()) != set(range(1, model.nbody)):
        raise ValueError("Entity body names must cover all non-world bodies")

    q_addresses = model.jnt_qposadr[joint_ids]
    v_addresses = model.jnt_dofadr[joint_ids]

    num_frames = len(qpos)
    num_bodies = len(body_ids)

    # arrays for motion.npz
    arrays = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": qpos[:, q_addresses].astype(np.float32),
        "joint_vel": qvel[:, v_addresses].astype(np.float32),
        "body_pos_w": np.empty((num_frames, num_bodies, 3), dtype=np.float32),
        "body_quat_w": np.empty((num_frames, num_bodies, 4), dtype=np.float32),
        "body_lin_vel_w": np.empty((num_frames, num_bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.empty((num_frames, num_bodies, 3), dtype=np.float32),
    }

    data = mujoco.MjData(model)
    root_ids = model.body_rootid[body_ids]

    for frame_index in range(num_frames):
        data.qpos[:] = qpos[frame_index]
        data.qvel[:] = qvel[frame_index]
        # xpos/xquat (kinematics), subtree_com (comPos) and cvel (comVel)
        # match mj_forward exactly; muscles and constraints are not needed.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)

        positions = data.xpos[body_ids]
        spatial_velocities = data.cvel[body_ids]
        angular_velocities = spatial_velocities[:, :3]

        # Match mjlab EntityData.compute_velocity_from_cvel:
        # shift velocity from subtree COM to each body link origin.
        offsets = data.subtree_com[root_ids] - positions
        linear_velocities = spatial_velocities[:, 3:] - np.cross(
            angular_velocities, offsets
        )

        arrays["body_pos_w"][frame_index] = positions
        arrays["body_quat_w"][frame_index] = data.xquat[body_ids]
        arrays["body_lin_vel_w"][frame_index] = linear_velocities
        arrays["body_ang_vel_w"][frame_index] = angular_velocities

    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite exported values in {name}")

    return arrays


def build_report(
    result: dict,
    motion: dict[str, np.ndarray],
    *,
    joint_names: Sequence[str],
    body_names: Sequence[str],
) -> dict:
    """
    Build a JSON-serializable retarget report.

    IK errors describe original IK frames before resampling and
    ground correction, not the final exported body positions.
    """

    errors = np.asarray(result["position_errors"], dtype=np.float64)
    error_names = list(result["error_body_names"])
    joint_names = list(joint_names)
    body_names = list(body_names)

    if errors.ndim != 2 or errors.shape[0] == 0 or errors.shape[1] != len(error_names):
        raise ValueError("Invalid IK error array or body names")
    if not np.isfinite(errors).all():
        raise ValueError("IK errors contain non-finite values")

    if motion["joint_pos"].shape[1] != len(joint_names):
        raise ValueError("Exported joint count does not match joint names")
    if motion["body_pos_w"].shape[1] != len(body_names):
        raise ValueError("Exported body count does not match body names")

    fps = float(np.asarray(motion["fps"]).item())
    num_frames = int(motion["joint_pos"].shape[0])

    per_body = {}
    for column, name in enumerate(error_names):
        values = errors[:, column]
        per_body[name] = {
            "mean_m": float(values.mean()),
            "p95_m": float(np.percentile(values, 95)),
            "max_m": float(values.max()),
        }

    metadata = deepcopy(result["metadata"])

    return {
        "format": "mjlab_tracking_npz",
        "metadata": metadata,
        "motion": {
            "file": "motion.npz",
            "fps": fps,
            "num_frames": num_frames,
            "sample_span_seconds": (num_frames - 1) / fps,
            "joint_names": joint_names,
            "body_names": body_names,
            "quaternion_order": "wxyz",
            "body_position_reference": "link_origin",
            "body_velocity_frame": "world",
            "array_shapes": {
                name: list(values.shape) for name, values in motion.items()
            },
        },
        "ik_position_errors": {
            "sampling": "original_ik_frames",
            "stage": "before_resampling_and_ground_correction",
            "num_frames": int(errors.shape[0]),
            "mean_m": float(errors.mean()),
            "p95_m": float(np.percentile(errors, 95)),
            "max_m": float(errors.max()),
            "per_body": per_body,
            "note": (
                "Unweighted body position distances, including tasks "
                "with zero position weight; not the IK objective value."
            ),
        },
    }


def save_result(
    motion: dict[str, np.ndarray],
    report: dict,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Save motion.npz and report.json in a new output directory.

    Write both files in a staging directory before publishing them.
    Existing output directories are not reused.
    """

    output_dir = Path(output_dir).expanduser().absolute()

    if os.path.lexists(output_dir):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError(f"Output must be a regular directory: {output_dir}")
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")

    # Validate inputs
    expected_fields = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    if set(motion) != expected_fields:
        raise ValueError("Motion must contain exactly the seven mjlab fields")

    for name, values in motion.items():
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if values.dtype.kind not in "fi":
            raise TypeError(f"{name} must contain real numeric values")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    motion_info = report["motion"]
    num_frames = motion_info["num_frames"]
    num_joints = len(motion_info["joint_names"])
    num_bodies = len(motion_info["body_names"])

    if num_frames <= 0 or num_joints <= 0 or num_bodies <= 0:
        raise ValueError("Motion dimensions must be positive")

    expected_shapes = {
        "fps": (1,),
        "joint_pos": (num_frames, num_joints),
        "joint_vel": (num_frames, num_joints),
        "body_pos_w": (num_frames, num_bodies, 3),
        "body_quat_w": (num_frames, num_bodies, 4),
        "body_lin_vel_w": (num_frames, num_bodies, 3),
        "body_ang_vel_w": (num_frames, num_bodies, 3),
    }
    for name, shape in expected_shapes.items():
        if motion[name].shape != shape:
            raise ValueError(
                f"{name}: expected shape {shape}, got {motion[name].shape}"
            )

    fps = float(motion["fps"].item())
    if fps <= 0 or fps != motion_info["fps"]:
        raise ValueError("Motion FPS must be positive and match the report")

    quaternion_norms = np.linalg.norm(motion["body_quat_w"], axis=-1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError("Body quaternions must have unit norm")

    # Serialize before creating output files; reject NaN/Infinity in JSON.
    report_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Same parent filesystem permits publishing via directory rename.
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-",
        dir=output_dir.parent,
    ) as temporary_dir:
        staging_dir = Path(temporary_dir) / "result"
        staging_dir.mkdir()

        np.savez_compressed(staging_dir / "motion.npz", **motion)

        with (staging_dir / "report.json").open("w", encoding="utf-8") as file:
            file.write(report_text)
            file.write("\n")

        # Recheck after computation and serialization.
        if os.path.lexists(output_dir):
            if output_dir.is_symlink() or not output_dir.is_dir():
                raise ValueError(f"Output must be a regular directory: {output_dir}")
            if not overwrite:
                raise FileExistsError(f"Output already exists: {output_dir}")

            filenames = ("motion.npz", "report.json")

            # Validate both destinations before replacing either file.
            for filename in filenames:
                destination = output_dir / filename
                if destination.is_symlink() or (
                    destination.exists() and not destination.is_file()
                ):
                    raise ValueError(
                        f"Output target must be a regular file: {destination}"
                    )

            for filename in filenames:
                (staging_dir / filename).replace(output_dir / filename)
        else:
            staging_dir.rename(output_dir)

    return output_dir


def evaluate_retarget_quality(report, motion_path):
    reasons = []
    metrics = {}
    metadata = report["metadata"]
    fit = metadata.get("shape_fitting")
    if not fit or not fit["metadata"]["metrics"].get("converged", False):
        reasons.append("shape_fit_not_converged")
    feet = report.get("ik_position_errors", {}).get("per_body", {})
    if not all(name in feet for name in ("calcn_l", "calcn_r")):
        reasons.append("missing_foot_ik_errors")
    else:
        metrics["foot_ik_p95_m"] = max(
            feet[name]["p95_m"] for name in ("calcn_l", "calcn_r")
        )
    ground = metadata.get("ground_correction")
    if not ground:
        reasons.append("missing_ground_correction_report")
    else:
        metrics["max_upward_shift_m"] = ground["max_upward_shift"]
    _collect_motion_metrics(report, motion_path, metrics, reasons)
    for name, value in metrics.items():
        if not np.isfinite(value) or value > THRESHOLDS[name]:
            reasons.append(name)
    return dict(
        status="failed" if reasons else "passed",
        reasons=reasons,
        metrics=metrics,
        thresholds=THRESHOLDS,
        scope="numerical_screen_only",
    )


def _collect_motion_metrics(report, motion_path, metrics, reasons):
    """Collect array metrics and append structural failures in screening order."""
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    with np.load(motion_path, allow_pickle=False) as arrays:
        if set(required) - set(arrays.files):
            raise ValueError("Missing exported motion arrays")
        if not all(np.isfinite(arrays[name]).all() for name in required):
            reasons.append("nonfinite_arrays")
        if arrays["joint_pos"].shape[0] < 3:
            reasons.append("insufficient_frames")
        metrics["max_body_linear_speed"] = float(
            np.linalg.norm(arrays["body_lin_vel_w"], axis=-1).max()
        )
        metrics["max_body_angular_speed"] = float(
            np.linalg.norm(arrays["body_ang_vel_w"], axis=-1).max()
        )
        pelvis = report["motion"]["body_names"].index("pelvis")
        metrics["max_pelvis_step_m"] = float(
            np.linalg.norm(
                np.diff(arrays["body_pos_w"][:, pelvis], axis=0), axis=-1
            ).max()
        )


def annotate_exported_result(directory):
    """Atomically attach numerical screening after export completes."""
    directory = Path(directory)
    path = directory / "report.json"
    report = json.loads(path.read_text())
    report["quality"] = evaluate_retarget_quality(report, directory / "motion.npz")
    report.pop("provenance", None)
    tmp = path.with_suffix(".quality.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)
    return report["quality"]
