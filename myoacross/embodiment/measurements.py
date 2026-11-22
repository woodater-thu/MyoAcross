"""measurement methods for embodiment(MyoFullBody)"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import mujoco
from collections.abc import Iterable

from myoacross.utils.paths import MEASUREMENTS_DIR
from myoacross.embodiment.smplh_to_measurements import (
    JOINT_MEASUREMENTS,
    JOINT_HEIGHT_MEASUREMENTS,
)

# MyoFullBody landmarks corresponding to SKEL joints
MYOFULLBODY_LANDMARKS = {
    # Lower limbs
    "femur_r": (mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r"),
    "tibia_r": (mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r"),
    "talus_r": (mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_r"),
    "calcn_r": (mujoco.mjtObj.mjOBJ_JOINT, "subtalar_angle_r"),
    "toes_r": (mujoco.mjtObj.mjOBJ_JOINT, "mtp_angle_r"),
    "femur_l": (mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_l"),
    "tibia_l": (mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_l"),
    "talus_l": (mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_l"),
    "calcn_l": (mujoco.mjtObj.mjOBJ_JOINT, "subtalar_angle_l"),
    "toes_l": (mujoco.mjtObj.mjOBJ_JOINT, "mtp_angle_l"),
    # Upper limbs
    "humerus_r": (mujoco.mjtObj.mjOBJ_JOINT, "shoulder_rot_r"),
    "ulna_r": (mujoco.mjtObj.mjOBJ_JOINT, "elbow_flex_r"),
    "radius_r": (mujoco.mjtObj.mjOBJ_JOINT, "pro_sup_r"),
    "hand_r": (mujoco.mjtObj.mjOBJ_JOINT, "flexion_r"),
    "humerus_l": (mujoco.mjtObj.mjOBJ_JOINT, "shoulder_rot_l"),
    "ulna_l": (mujoco.mjtObj.mjOBJ_JOINT, "elbow_flex_l"),
    "radius_l": (mujoco.mjtObj.mjOBJ_JOINT, "pro_sup_l"),
    "hand_l": (mujoco.mjtObj.mjOBJ_JOINT, "flexion_l"),
}


def load_subject_measurements(
    subject_id: str,
    *,
    measurements_dir: str | Path = MEASUREMENTS_DIR,
) -> dict[str, float]:
    """Load measurements for one subject."""

    # Validate inputs
    subject_id = str(subject_id).strip()
    if not subject_id:
        raise ValueError("Subject ID must not be empty")
    if Path(subject_id).name != subject_id:
        raise ValueError(f"Subject ID must not contain path separators: {subject_id!r}")

    measurements_path = (
        Path(measurements_dir).expanduser().resolve() / f"{subject_id}.json"
    )
    if not measurements_path.is_file():
        raise FileNotFoundError(
            f"Subject measurement file does not exist: {measurements_path}"
        )

    # Load measurements from JSON file
    with measurements_path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    # Validate outputs
    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping in measurement file: {measurements_path}")
    if document.get("subject_id") != subject_id:
        raise ValueError(
            f"Subject ID mismatch in measurement file: {measurements_path}"
        )

    raw_measurements = document.get("measurements")
    if not isinstance(raw_measurements, dict):
        raise ValueError(
            f"Measurement file is missing a 'measurements' mapping: {measurements_path}"
        )

    measurements = {str(name): float(value) for name, value in raw_measurements.items()}

    invalid_names = [
        name
        for name, value in measurements.items()
        if not np.isfinite(value) or value <= 0.0
    ]
    if invalid_names:
        raise ValueError(
            f"Measurements must be finite and positive: {sorted(invalid_names)}"
        )

    return measurements


def compute_myofullbody_landmarks(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compute MyoFullBody landmarks in an equality-consistent measurement T-pose."""

    # Set the measurement pose without changing the model's default configuration
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    shoulder_positions: dict[int, float] = {}
    for side in ("r", "l"):
        for name, angle in (
            (f"elv_angle_{side}", 0.0),
            (f"shoulder_elv_{side}", np.pi / 2),
            (f"shoulder_rot_{side}", 0.0),
        ):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise KeyError(f"MyoFullBody is missing required joint {name!r}")
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected a hinge joint for {name!r}")
            address = int(model.jnt_qposadr[joint_id])
            shoulder_positions[address] = angle
            data.qpos[address] = angle

    # Resolve all active joint equalities, including spine, knee and locked joints
    pending: dict[int, int] = {}
    for equality_id in np.flatnonzero(data.eq_active):
        if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        joint_id = int(model.eq_obj1id[equality_id])
        if joint_id in pending:
            raise ValueError(f"Multiple active equalities constrain joint {joint_id}")
        pending[joint_id] = int(equality_id)

    while pending:
        resolved: list[int] = []
        for joint_id, equality_id in pending.items():
            source_id = int(model.eq_obj2id[equality_id])
            if source_id in pending:
                continue
            offset = 0.0
            if source_id >= 0:
                source_address = int(model.jnt_qposadr[source_id])
                offset = data.qpos[source_address] - model.qpos0[source_address]
            address = int(model.jnt_qposadr[joint_id])
            data.qpos[address] = model.qpos0[
                address
            ] + np.polynomial.polynomial.polyval(offset, model.eq_data[equality_id, :5])
            resolved.append(joint_id)
        if not resolved:
            raise ValueError(
                "Cyclic joint equalities prevent constructing the measurement T-pose"
            )
        for joint_id in resolved:
            del pending[joint_id]

    # Forward dynamics evaluates constraints but does not project qpos onto them
    if not np.isfinite(data.qpos).all():
        raise ValueError("Measurement T-pose contains non-finite joint positions")
    for address, angle in shoulder_positions.items():
        if not np.isclose(data.qpos[address], angle, rtol=0.0, atol=1e-10):
            raise ValueError("Joint equalities conflict with the measurement T-pose")
    mujoco.mj_forward(model, data)

    # Check every active equality, including any non-joint constraints
    equality_rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    active_equalities = set(np.flatnonzero(data.eq_active).tolist())
    evaluated_equalities = set(data.efc_id[equality_rows].tolist())
    if evaluated_equalities != active_equalities:
        raise ValueError("Not all active measurement-pose equalities were evaluated")
    residuals = data.efc_pos[equality_rows]
    if not np.isfinite(residuals).all() or np.any(np.abs(residuals) > 1e-10):
        raise ValueError("Measurement T-pose does not satisfy all active equalities")

    # Extract measurement-pose landmark positions
    landmark_names = tuple(MYOFULLBODY_LANDMARKS)
    landmark_positions: list[np.ndarray] = []

    for landmark_name in landmark_names:
        object_type, model_name = MYOFULLBODY_LANDMARKS[landmark_name]

        object_id = mujoco.mj_name2id(
            model,
            object_type,
            model_name,
        )
        if object_id < 0:
            raise KeyError(
                f"MyoFullBody is missing required {object_type.name.lower()} {model_name!r}"
            )

        if object_type == mujoco.mjtObj.mjOBJ_JOINT:
            position = data.xanchor[object_id]
        elif object_type == mujoco.mjtObj.mjOBJ_SITE:
            position = data.site_xpos[object_id]
        else:
            raise RuntimeError(f"Unsupported landmark object type: {object_type}")

        landmark_positions.append(
            np.asarray(
                position,
                dtype=np.float64,
            ).copy()
        )

    landmarks_reference = np.stack(
        landmark_positions,
        axis=0,
    )

    # Validate outputs
    expected_shape = (len(landmark_names), 3)
    if landmarks_reference.shape != expected_shape:
        raise RuntimeError(
            f"Expected reference landmarks with shape {expected_shape}, got {landmarks_reference.shape}"
        )
    if not np.isfinite(landmarks_reference).all():
        raise RuntimeError("MyoFullBody reference landmarks contain non-finite values")

    return landmarks_reference, landmark_names


def compute_landmark_group_center(
    landmarks_reference: np.ndarray,
    landmark_indices: dict[str, int],
    landmark_group: tuple[str, ...],
) -> np.ndarray:
    """Compute the center of one or more MyoFullBody landmarks."""

    # Validate inputs
    if not landmark_group:
        raise ValueError("Landmark group must contain at least one landmark")

    missing_landmarks = set(landmark_group) - set(landmark_indices)
    if missing_landmarks:
        raise KeyError(
            f"MyoFullBody is missing required landmarks: {sorted(missing_landmarks)}"
        )

    # Compute the center of the landmark group
    indices = [landmark_indices[name] for name in landmark_group]
    center = np.mean(
        landmarks_reference[indices],
        axis=0,
    )

    # Validate output
    if center.shape != (3,):
        raise RuntimeError(
            f"Expected landmark-group center with shape (3,), got {center.shape}"
        )
    if not np.isfinite(center).all():
        raise ValueError(
            f"Landmark group {landmark_group!r} produced a non-finite center"
        )

    return center


def compute_myofullbody_measurements(
    landmarks_reference: np.ndarray,
    landmark_names: Iterable[str],
) -> dict[str, float]:
    """Compute joint-based measurements from MyoFullBody landmarks."""

    landmarks_reference = np.asarray(
        landmarks_reference,
        dtype=np.float64,
    )
    landmark_names = tuple(str(name) for name in landmark_names)

    # Validate inputs
    expected_shape = (len(landmark_names), 3)
    if landmarks_reference.shape != expected_shape:
        raise ValueError(
            f"Expected reference landmarks with shape {expected_shape}, got {landmarks_reference.shape}"
        )
    if not np.isfinite(landmarks_reference).all():
        raise ValueError("Reference landmarks contain non-finite values")
    if len(set(landmark_names)) != len(landmark_names):
        raise ValueError("MyoFullBody landmark names contain duplicates")

    landmark_indices = {name: index for index, name in enumerate(landmark_names)}

    required_landmarks = {
        landmark_name
        for measurement_table in (
            JOINT_MEASUREMENTS,
            JOINT_HEIGHT_MEASUREMENTS,
        )
        for endpoint_groups in measurement_table.values()
        for landmark_group in endpoint_groups
        for landmark_name in landmark_group
    }
    missing_landmarks = required_landmarks - set(landmark_indices)
    if missing_landmarks:
        raise KeyError(
            f"MyoFullBody is missing required landmarks: {sorted(missing_landmarks)}"
        )

    # Compute distances between landmark-group centers
    measurements: dict[str, float] = {}

    for measurement_name, (proximal_group, distal_group) in JOINT_MEASUREMENTS.items():
        proximal_center = compute_landmark_group_center(
            landmarks_reference,
            landmark_indices,
            proximal_group,
        )
        distal_center = compute_landmark_group_center(
            landmarks_reference,
            landmark_indices,
            distal_group,
        )
        distance = float(np.linalg.norm(distal_center - proximal_center))
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(
                f"Invalid value for measurement {measurement_name!r}: {distance}"
            )

        measurements[measurement_name] = distance

    # Compute vertical height between landmark-group centers
    for measurement_name, (
        upper_group,
        lower_group,
    ) in JOINT_HEIGHT_MEASUREMENTS.items():
        upper_center = compute_landmark_group_center(
            landmarks_reference,
            landmark_indices,
            upper_group,
        )
        lower_center = compute_landmark_group_center(
            landmarks_reference,
            landmark_indices,
            lower_group,
        )
        height = float(upper_center[2] - lower_center[2])
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError(
                f"Invalid value for measurement {measurement_name!r}: {height}"
            )

        measurements[measurement_name] = height

    return measurements


def compute_myofullbody_height(
    model: mujoco.MjModel,
) -> float:
    """Compute the world-Z span of model meshes in the reference pose."""

    # Evaluate the model in its reference pose
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Find vertical bounds of mesh geometry
    minimum_z = np.inf
    maximum_z = -np.inf

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        if vertex_count == 0:
            continue

        vertices = np.asarray(
            model.mesh_vert[vertex_start : vertex_start + vertex_count],
            dtype=np.float64,
        )

        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_z = vertices @ rotation[2, :] + data.geom_xpos[geom_id, 2]

        minimum_z = min(minimum_z, float(world_z.min()))
        maximum_z = max(maximum_z, float(world_z.max()))

    # Validate output
    height = float(maximum_z - minimum_z)
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError(
            f"Expected a finite, positive MyoFullBody mesh height, got {height}"
        )

    return height
