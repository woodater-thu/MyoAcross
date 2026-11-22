"""Extract subject-specific body measurements from SMPL data using SKEL."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import json
import numpy as np
import argparse

from myoacross.utils.paths import (
    DATASET_ROOT,
    SKEL_MODEL_DIR,
    BMLMOVI_SUBJECTS_PATH,
    KIT_SUBJECTS_PATH,
    MEASUREMENTS_DIR,
)
from myoacross.utils.datalists import (
    get_subject_reference_motion,
    get_target_subject_ids,
)

JointGroup = tuple[str, ...]
JointMeasurement = tuple[JointGroup, JointGroup]

# Joint-based Euclidean distance measurements table
JOINT_MEASUREMENTS: dict[str, JointMeasurement] = {
    # Lower limbs
    "thigh_length_r": (("femur_r",), ("tibia_r",)),
    "shank_length_r": (("tibia_r",), ("talus_r",)),
    "thigh_length_l": (("femur_l",), ("tibia_l",)),
    "shank_length_l": (("tibia_l",), ("talus_l",)),

    # Feet
    "ankle_subtalar_length_r":  (("talus_r",), ("calcn_r",)),
    "subtalar_mtp_length_r":    (("calcn_r",), ("toes_r",)),
    "ankle_subtalar_length_l":  (("talus_l",), ("calcn_l",)),
    "subtalar_mtp_length_l":    (("calcn_l",), ("toes_l",)),

    # Upper limbs
    "upper_arm_length_r":   (("humerus_r",), ("ulna_r",)),
    "forearm_length_r":     (("radius_r",), ("hand_r",)),
    "upper_arm_length_l":   (("humerus_l",), ("ulna_l",)),
    "forearm_length_l":     (("radius_l",), ("hand_l",)),

    # Body widths
    "hip_width":        (("femur_r",), ("femur_l",)),
    "shoulder_width":   (("humerus_r",), ("humerus_l",)),

    # Distance between bilateral joint-center midpoints
    "trunk_length": (
        ("femur_r", "femur_l"),
        ("humerus_r", "humerus_l"),
    ),
}
# Joint-based vertical height meausrements table
JOINT_HEIGHT_MEASUREMENTS: dict[str, JointMeasurement] = {
    "shoulder_ankle_height_r": (("humerus_r",), ("talus_r",)),
    "shoulder_ankle_height_l": (("humerus_l",), ("talus_l",)),
}


def load_subject_metadata(
    subject_id: str,
    *,
    metadata_path: str | Path | None = None,
) -> dict[str, str | int | float]:
    """Load one subject's demographic and physical metadata."""

    # Validate inputs
    subject_id = str(subject_id).strip()
    if not subject_id:
        raise ValueError("Subject ID must not be empty")

    if metadata_path is None:
        dataset_name = subject_id.partition("_")[0]
        metadata_paths = {
            "BMLmovi": BMLMOVI_SUBJECTS_PATH,
            "KIT": KIT_SUBJECTS_PATH,
        }
        try:
            metadata_path = metadata_paths[dataset_name]
        except KeyError as error:
            raise KeyError(
                f"No subject metadata file is registered for dataset {dataset_name!r}"
            ) from error

    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Subject metadata file does not exist: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping in subject metadata file: {metadata_path}")

    subjects = document.get("subjects")
    if not isinstance(subjects, dict):
        raise ValueError(f"Metadata file is missing a 'subjects' mapping: {metadata_path}")
    if subject_id not in subjects:
        raise KeyError(f"Subject {subject_id!r} is not present in {metadata_path}")

    metadata = subjects[subject_id]
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata for {subject_id!r} to be a mapping")

    required_fields = {
        "gender",
        "age_years",
        "height_m",
        "body_mass_kg",
    }
    missing_fields = required_fields - set(metadata)
    if missing_fields:
        raise KeyError(f"Subject {subject_id!r} is missing metadata fields: {sorted(missing_fields)}")

    # Load subject metadata fields
    gender = str(metadata["gender"]).strip().lower()
    age_years = int(metadata["age_years"])
    height_m = float(metadata["height_m"])
    body_mass_kg = float(metadata["body_mass_kg"])

    # Validate outputs
    if gender not in {"male", "female"}:
        raise ValueError(f"Invalid gender for {subject_id!r}: {gender!r}")
    if age_years <= 0:
        raise ValueError(f"Invalid age for {subject_id!r}: {age_years}")
    if not np.isfinite(height_m) or height_m <= 0.0:
        raise ValueError(f"Invalid height for {subject_id!r}: {height_m}")
    if not np.isfinite(body_mass_kg) or body_mass_kg <= 0.0:
        raise ValueError(f"Invalid body mass for {subject_id!r}: {body_mass_kg}")

    return {
        "gender": gender,
        "age_years": age_years,
        "height_m": height_m,
        "body_mass_kg": body_mass_kg,
    }


def load_smpl_identity(
    motion_path: str | Path,
) -> tuple[str, np.ndarray]:
    """Load gender and the first 10 SMPL shape coefficients from a SMPL motion."""

    motion_path = Path(motion_path).expanduser().resolve()

    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
    if motion_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an NPZ motion file: {motion_path}")

    with np.load(motion_path, allow_pickle=False) as motion:
        missing_keys = {"gender", "betas"} - set(motion.files)
        if missing_keys:
            raise KeyError(f"{motion_path} is missing required keys: {sorted(missing_keys)}")

        gender_value = np.asarray(motion["gender"])
        if gender_value.size != 1:
            raise ValueError(f"Expected one gender value in {motion_path}, got shape {gender_value.shape}")
        gender = gender_value.item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        gender = str(gender).strip().lower()
        if gender not in {"male", "female"}:
            raise ValueError(f"Expected gender to be 'male' or 'female' in {motion_path}, got {gender!r}")

        betas = np.asarray(motion["betas"], dtype=np.float64).reshape(-1)
        if betas.size < 10:
            raise ValueError(f"Expected at least 10 SMPL betas in {motion_path}, got {betas.size}")
        betas = betas[:10].copy()
        if not np.isfinite(betas).all():
            raise ValueError(f"SMPL betas contain non-finite values: {motion_path}")
        
    return gender, betas


def load_subject_identity(
    motion_paths: Iterable[str | Path],
    *,
    beta_atol: float = 1e-6,
) -> tuple[str, np.ndarray]:
    """Validate and return the shared identity parameters of one subject."""

    motion_paths = tuple(motion_paths)
    if not motion_paths:
        raise ValueError("Expected at least one motion path")

    reference_path = Path(motion_paths[0]).expanduser().resolve()
    reference_gender, reference_betas = load_smpl_identity(reference_path)

    for motion_path in motion_paths[1:]:
        motion_path = Path(motion_path).expanduser().resolve()
        gender, betas = load_smpl_identity(motion_path)

        if gender != reference_gender:
            raise ValueError(f"Inconsistent gender for one subject: {reference_path} has {reference_gender!r}, but {motion_path} has {gender!r}")

        if not np.allclose(
            betas, reference_betas,
            rtol=0.0, atol=beta_atol,
        ):
            max_difference = float(np.max(np.abs(betas - reference_betas)))
            raise ValueError(f"Inconsistent SMPL betas for one subject: {reference_path} and {motion_path} differ by as much as {max_difference:.6g}, exceeding beta_atol={beta_atol}")

    return reference_gender, reference_betas


def compute_tpose_geometry(
    gender: str,
    betas: np.ndarray,
    *,
    model_dir: str | Path = SKEL_MODEL_DIR,
    device: str = "cpu",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Infer subject-specific SKEL joint positions in the T-pose."""

    import torch
    from skel.skel_model import SKEL

    # Validate inputs
    gender = str(gender).strip().lower()
    if gender not in {"male", "female"}:
        raise ValueError(f"Expected gender to be 'male' or 'female', got {gender!r}")

    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas.shape != (10,):
        raise ValueError(f"Expected exactly 10 SMPL betas, got shape {betas.shape}")
    if not np.isfinite(betas).all():
        raise ValueError(f"SMPL betas contain non-finite values: {betas}")

    model_dir = Path(model_dir).expanduser().resolve()
    model_file = model_dir / f"skel_{gender}.pkl"
    if not model_file.is_file():
        raise FileNotFoundError(f"SKEL model file does not exist: {model_file}")

    # Load the SKEL model
    torch_device = torch.device(device)
    model = SKEL(
        gender=gender,
        model_path=str(model_dir),
    ).to(torch_device)
    model.eval()

    beta_tensor = torch.as_tensor(
        betas,
        dtype=torch.float32,
        device=torch_device,
    ).unsqueeze(0)
    pose_tensor = torch.zeros(
        (1, model.num_q_params),
        dtype=torch.float32,
        device=torch_device,
    )
    translation_tensor = torch.zeros(
        (1, 3),
        dtype=torch.float32,
        device=torch_device,
    )

    # Infer the subject-specific T-pose geometry
    with torch.inference_mode():
        output = model(
            poses=pose_tensor,
            betas=beta_tensor,
            trans=translation_tensor,
            poses_type="skel",
            skelmesh=False,
        )

    # Extract T-pose joint centers
    joints_tpose = (
        output.joints_tpose[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )


    # Validate outputs
    bone_names = tuple(str(name) for name in model.bone_names)
    expected_joint_shape = (len(bone_names), 3)
    if joints_tpose.shape != expected_joint_shape:
        raise RuntimeError(f"Expected T-pose joints with shape {expected_joint_shape}, got {joints_tpose.shape}")
    if not np.isfinite(joints_tpose).all():
        raise RuntimeError("T-pose joints contain non-finite values")

    return joints_tpose, bone_names


def compute_joint_group_center(
    joints_tpose: np.ndarray,
    bone_indices: dict[str, int],
    joint_group: JointGroup,
) -> np.ndarray:
    """Compute the center of one or more SKEL joint positions."""

    # Validate inputs
    if not joint_group:
        raise ValueError("Joint group must contain at least one joint")
    missing_bones = set(joint_group) - set(bone_indices)
    if missing_bones:
        raise KeyError(f"SKEL model is missing required bones: {sorted(missing_bones)}")

    # Compute the center of the joint group
    indices = [bone_indices[name] for name in joint_group]
    center = np.mean(joints_tpose[indices], axis=0)

    # Validate output
    if center.shape != (3,):
        raise RuntimeError(f"Expected joint-group center with shape (3,), got {center.shape}")
    if not np.isfinite(center).all():
        raise ValueError(f"Joint group {joint_group!r} produced a non-finite center")

    return center


def compute_measurements(
    joints_tpose: np.ndarray,
    bone_names: Iterable[str],
) -> dict[str, float]:
    """Compute joint-based body measurements from SKEL T-pose joints."""

    joints_tpose = np.asarray(
        joints_tpose,
        dtype=np.float64,
    )
    bone_names = tuple(str(name) for name in bone_names)

    # Validate inputs
    expected_shape = (len(bone_names), 3)
    if joints_tpose.shape != expected_shape:
        raise ValueError(f"Expected T-pose joints with shape {expected_shape}, got {joints_tpose.shape}")
    if not np.isfinite(joints_tpose).all():
        raise ValueError("T-pose joints contain non-finite values")
    if len(set(bone_names)) != len(bone_names):
        raise ValueError("SKEL bone names contain duplicates")

    bone_indices = {
        name: index
        for index, name in enumerate(bone_names)
    }

    required_bones = {
        bone_name
        for measurement_table in (
            JOINT_MEASUREMENTS,
            JOINT_HEIGHT_MEASUREMENTS,
        )
        for endpoint_groups in measurement_table.values()
        for joint_group in endpoint_groups
        for bone_name in joint_group
    }
    missing_bones = required_bones - set(bone_indices)
    if missing_bones:
        raise KeyError(f"SKEL model is missing required bones: {sorted(missing_bones)}")

    # Compute distances between joint-group centers
    measurements: dict[str, float] = {}
    for measurement_name, (proximal_group, distal_group,) in JOINT_MEASUREMENTS.items():
        proximal_center = compute_joint_group_center(
            joints_tpose,
            bone_indices,
            proximal_group,
        )
        distal_center = compute_joint_group_center(
            joints_tpose,
            bone_indices,
            distal_group,
        )

        distance = float(np.linalg.norm(distal_center - proximal_center))
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"Invalid value for measurement {measurement_name!r}: {distance}")

        measurements[measurement_name] = distance

    # Compute vertical height between joint groups.
    for measurement_name, (upper_group, lower_group,) in JOINT_HEIGHT_MEASUREMENTS.items():
        upper_center = compute_joint_group_center(
            joints_tpose,
            bone_indices,
            upper_group,
        )
        lower_center = compute_joint_group_center(
            joints_tpose,
            bone_indices,
            lower_group,
        )

        height = float(upper_center[1] - lower_center[1])
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError(f"Invalid value for measurement {measurement_name!r}: {height}")

        measurements[measurement_name] = height
    
    return measurements


def compute_subject_measurements(
    subject_id: str,
    motion_paths: Iterable[str | Path],
    *,
    metadata_path: str | Path | None = None,
    beta_atol: float = 1e-6,
    model_dir: str | Path = SKEL_MODEL_DIR,
    device: str = "cpu",
) -> tuple[str, np.ndarray, dict[str, float]]:
    """Compute identity and body measurements for one subject."""

    # Load subject metadata and identity
    subject_metadata = load_subject_metadata(
        subject_id,
        metadata_path=metadata_path,
    )
    gender, betas = load_subject_identity(
        motion_paths,
        beta_atol=beta_atol,
    )

    # Validate inputs
    metadata_gender = str(subject_metadata["gender"])
    if gender != metadata_gender:
        raise ValueError(f"Gender mismatch for {subject_id!r}: motion data has {gender!r}, but metadata has {metadata_gender!r}")

    # Compute measurements
    joints_tpose, bone_names = compute_tpose_geometry(
        gender,
        betas,
        model_dir=model_dir,
        device=device,
    )
    measurements = compute_measurements(
        joints_tpose,
        bone_names,
    )

    measurements["height"] = float(subject_metadata["height_m"])
    measurements["body_mass"] = float(subject_metadata["body_mass_kg"])

    return gender, betas, measurements


def save_subject_measurements(
    subject_id: str,
    gender: str,
    betas: np.ndarray,
    measurements: dict[str, float],
    *,
    output_dir: str | Path = MEASUREMENTS_DIR,
    overwrite: bool = False,
) -> Path:
    """Save one subject's identity and measurements as JSON."""

    # Validate subject ID
    subject_id = str(subject_id).strip()
    if not subject_id:
        raise ValueError("Subject ID must not be empty")
    if Path(subject_id).name != subject_id:
        raise ValueError(f"Subject ID must not contain path separators: {subject_id!r}")

    # Resolve output path
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{subject_id}.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Measurement file already exists: {output_path}")

    payload = {
        "subject_id": subject_id,
        "gender": gender,
        "betas": betas.tolist(),
        "measurement_units": {
            "joint_distances": "m",
            "joint_heights": "m",
            "height": "m",
            "body_mass": "kg",
        },
        "measurements": {
            name: float(value)
            for name, value in sorted(measurements.items())
        },
    }

    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_path


def generate_measurements(
    motions_by_subject: Mapping[str, Iterable[str | Path]],
    *,
    metadata_path: str | Path | None = None,
    output_dir: str | Path = MEASUREMENTS_DIR,
    beta_atol: float = 1e-6,
    model_dir: str | Path = SKEL_MODEL_DIR,
    device: str = "cpu",
    overwrite: bool = False,
    dataset_root: str | Path = DATASET_ROOT,
) -> dict[str, Path]:
    """Compute and save measurements for multiple subjects."""

    if not motions_by_subject:
        raise ValueError("Expected at least one subject")

    output_paths: dict[str, Path] = {}

    for subject_id, motion_paths in motions_by_subject.items():
        motion_paths = tuple(motion_paths)
        if not motion_paths:
            raise ValueError(f"Subject {subject_id!r} has no motion paths")

        gender, betas, measurements = (
            compute_subject_measurements(
                subject_id,
                motion_paths,
                metadata_path=(
                    metadata_path if metadata_path is not None else
                    Path(dataset_root).expanduser() / subject_id.partition("_")[0] / "subjects.json"
                ),
                beta_atol=beta_atol,
                model_dir=model_dir,
                device=device,
            )
        )

        output_path = save_subject_measurements(
            subject_id,
            gender,
            betas,
            measurements,
            output_dir=output_dir,
            overwrite=overwrite,
        )

        output_paths[subject_id] = output_path

    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate measurements from each registered target subject's own SMPL shape."
        )
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Target subject split (KIT/BMLmovi); default: train.",
    )    
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help=(
            "Process only one subject, such as KIT_3. By default, all target subjects in the selected split are processed."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for SKEL inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MEASUREMENTS_DIR,
        help="Directory used to save measurement JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing measurement files.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root containing KIT/BMLmovi subject shape references and subjects.json.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate measurements for the selected product target subjects."""

    args = parse_args(argv)
    subject_ids = get_target_subject_ids(split=args.split)
    if args.subject is not None:
        subject_id = args.subject.strip()
        if subject_id not in subject_ids:
            raise ValueError(
                f"Subject {subject_id!r} is not registered in target split {args.split!r}"
            )
        subject_ids = (subject_id,)

    # Resolve all references before any measurement outputs are written.
    motions_by_subject = {
        subject_id: (get_subject_reference_motion(
            subject_id, dataset_root=args.dataset_root,
        ),)
        for subject_id in subject_ids
    }

    # Generate measurements for each subject
    print("Generating measurements for target subjects...")
    output_paths = generate_measurements(
        motions_by_subject,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.overwrite,
    )
    print("Done generating measurements.")

    for subject_id, output_path in output_paths.items():
        print(f"{subject_id}: {output_path}")


if __name__ == "__main__":
    main()
