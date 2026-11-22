"""GMR-Fit integration for subject-specific embodiments."""

from collections import OrderedDict
from functools import wraps

import mink
import mujoco
from pathlib import Path
import numpy as np
import torch
from copy import deepcopy
import json
import os
import tempfile


from general_motion_retargeting import (
    GeneralMotionRetargeting,
    IK_CONFIG_DICT,
)
from general_motion_retargeting.utils.smpl import (
    load_smplh_file,
    get_smplh_data_offline_fast,
    SMPLH_Parser,
)
from general_motion_retargeting.utils.shape_fitting import (
    get_robot_tpose_targets,
    get_smpl_tpose_indices,
    compute_alignment_offsets,
)

from myoacross.utils.paths import SMPLH_MODEL_DIR
from myoacross.retarget import acceleration
from myoacross.retarget.shape_fitting import (
    ensure_fitted_shape,
    validate_fitted_shape,
    _digest,
)
from myoacross.retarget.postprocess import (
    correct_ground,
    compute_qvel,
    resample_qpos,
)


# Per-process caches. Retargeters are reset before each motion, and
# offsets depend only on (XML, SMPLH gender, betas), so reuse is exact.
_RETARGETER_CACHE: OrderedDict = OrderedDict()
_RETARGETER_CACHE_SIZE = 2
_OFFSETS_CACHE: OrderedDict = OrderedDict()
_OFFSETS_CACHE_SIZE = 64
_SMPLH_PARSERS: dict = {}


def cached_smplh_parser(*args, **kwargs) -> SMPLH_Parser:
    """
    Return one SMPLH_Parser per constructor arguments in this process.

    Used for subject offsets only; the parser is evaluated with explicit
    pose, betas and translation, so its parameters are never modified.
    GMR's load_smplh_file() still builds its own parser: its methods call
    super(SMPLH_Parser, self) through the module global, so that name must
    not be replaced.
    """
    key = (args, tuple(sorted(kwargs.items())))
    if key not in _SMPLH_PARSERS:
        _SMPLH_PARSERS[key] = SMPLH_Parser(*args, **kwargs)
    return _SMPLH_PARSERS[key]


def install_mink_patch(fast_ik: bool = False) -> None:
    """
    Apply MuscleMimic's GMR/Mink calling compatibility behavior.

    Preserve IK limits while disabling early termination for an out-of-range current configuration.
    With fast_ik, the same QP is solved with box-bounded, reduced DAQP.
    """
    acceleration.ENABLED["ik"] = bool(fast_ik)
    acceleration.install_equality_mask_patch()

    if getattr(mink.solve_ik, "_myoacross_compat", False):
        return

    original_solve_ik = mink.solve_ik

    @wraps(original_solve_ik)
    def solve_ik_compat(
        configuration,
        tasks,
        dt,
        solver,
        damping=1e-12,
        safety_break=False,
        limits=None,
        constraints=None,
        **kwargs,
    ):
        # GMR passes its limits list as the sixth positional argument.
        if isinstance(safety_break, list):
            limits = safety_break

        if (
            acceleration.ENABLED["ik"]
            and not kwargs
            and acceleration.supports_box_qp(solver, limits, constraints)
        ):
            return acceleration.solve_ik_box(configuration, tasks, dt, damping, limits[0])

        return original_solve_ik(
            configuration,
            tasks,
            dt,
            solver,
            damping=damping,
            safety_break=False,
            limits=limits,
            constraints=constraints,
            **kwargs,
        )

    solve_ik_compat._myoacross_compat = True
    mink.solve_ik = solve_ik_compat


def initialize_retargeter(
    retargeter: GeneralMotionRetargeting,
) -> list[dict]:
    """
    Initialize out-of-range scalar joints to their range midpoints.

    Follow MuscleMimic's MyoFullBody initialization strategy without modifying the model's qpos0 or XML.
    """
    configuration = retargeter.configuration
    model = configuration.model
    qpos = configuration.data.qpos.copy()

    adjustments = []

    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue

        # Joint ranges represent scalar positions for hinge/slide joints.
        if int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue

        qpos_address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        previous = float(qpos[qpos_address])

        if lower <= previous <= upper:
            continue

        initialized = float((lower + upper) / 2)
        qpos[qpos_address] = initialized

        adjustments.append(
            {
                "joint_name": model.joint(joint_id).name,
                "previous": previous,
                "initialized": initialized,
            }
        )

    if adjustments:
        configuration.update(q=qpos)

    return adjustments


def create_retargeter(
    robot_xml_path: str | Path,
    config: dict,
    *,
    fitted_shape_path: str | Path | None = None,
) -> tuple[GeneralMotionRetargeting, list[dict]]:
    """
    Create and initialize GMR retargeter for an explicit MyoFullBody XML.

    Args:
        robot_xml_path: Generated embodiment XML.
        config: Validated configuration returned by load_config().

    Returns:
        Retargeter and initial joint adjustment records.
    """

    # Validate myofullbody.xml and IK configs
    robot_xml_path = Path(robot_xml_path).expanduser().resolve()
    if not robot_xml_path.is_file():
        raise FileNotFoundError(f"Embodiment XML not found: {robot_xml_path}")

    ik_config_path = Path(IK_CONFIG_DICT["smplh"]["myofullbody"])
    if not ik_config_path.is_file():
        raise FileNotFoundError(f"GMR IK configuration not found: {ik_config_path}")

    # Patch for out-of-range
    fast_ik = config.get("performance", {}).get("fast_ik", False)
    install_mink_patch(fast_ik)

    # GMR retargeter
    options = config["gmr"]
    use_fitted_shape = config["gmr"]["use_fitted_shape"]
    if use_fitted_shape != (fitted_shape_path is not None):
        raise ValueError("Fitted shape flag and path must agree")
    if fitted_shape_path is not None:
        validate_fitted_shape(fitted_shape_path)
    retargeter = GeneralMotionRetargeting(
        src_human="smplh",
        tgt_robot="myofullbody",
        robot_xml_path=str(robot_xml_path),
        ik_config_path=str(ik_config_path),
        actual_human_height=None,
        use_fitted_shape=use_fitted_shape,
        fitted_shape_path=str(fitted_shape_path)
        if fitted_shape_path is not None
        else None,
        solver=options["solver"],
        damping=options["damping"],
        use_velocity_limit=options["use_velocity_limit"],
        verbose=options["verbose"],
    )

    if use_fitted_shape and (
        not hasattr(retargeter, "fitted_shape") or retargeter.learned_offsets is None
    ):
        raise RuntimeError("GMR did not load the fitted shape and local offsets")

    # Fitted scale is already applied in the SMPL loader; do not scale twice.
    if any(scale != 1.0 for scale in retargeter.human_scale_table.values()):
        raise ValueError(
            "Subject-specific retargeting requires all GMR human scale factors to equal 1.0"
        )

    if fast_ik:
        acceleration.install_vectorized_targets(retargeter)

    # Adjust initial qpos
    adjustments = initialize_retargeter(retargeter)

    return retargeter, adjustments


def get_retargeter(
    robot_xml_path: str | Path,
    config: dict,
    *,
    fitted_shape_path: str | Path | None = None,
) -> tuple[GeneralMotionRetargeting, list[dict]]:
    """
    Return a retargeter for this XML in the state create_retargeter() gives.

    Compiling MyoFullBody takes about a second, so retargeters are cached
    per process and reset instead: a new mink Configuration at qpos0, the
    original fitted offsets (or None for source-shape mode), zero ground
    offset, then the same out-of-range initialization. Fit contents are part
    of the key, including their metadata where GMR stores local offsets.
    """
    robot_xml_path = Path(robot_xml_path).expanduser().resolve()
    options = config["gmr"]
    fast_ik = config.get("performance", {}).get("fast_ik", False)
    if options["use_fitted_shape"] != (fitted_shape_path is not None):
        raise ValueError("Fitted shape flag and path must agree")
    fit_key = None
    if fitted_shape_path is not None:
        fitted_shape_path = Path(fitted_shape_path).expanduser().resolve()
        validate_fitted_shape(fitted_shape_path)
        metadata_path = fitted_shape_path.with_name(
            fitted_shape_path.stem + "_metadata.json"
        )
        fit_key = (
            str(fitted_shape_path),
            _digest(fitted_shape_path),
            _digest(metadata_path),
        )
    key = (
        str(robot_xml_path),
        robot_xml_path.stat().st_mtime_ns if robot_xml_path.is_file() else None,
        options["solver"],
        options["damping"],
        options["use_velocity_limit"],
        options["verbose"],
        options["use_fitted_shape"],
        fit_key,
        fast_ik,
    )

    retargeter = _RETARGETER_CACHE.pop(key, None)
    if retargeter is None:
        retargeter, adjustments = create_retargeter(
            robot_xml_path,
            config,
            fitted_shape_path=fitted_shape_path,
        )
        retargeter._initial_learned_offsets = deepcopy(retargeter.learned_offsets)
    else:
        install_mink_patch(fast_ik)
        retargeter.configuration = mink.Configuration(retargeter.model)
        retargeter.learned_offsets = deepcopy(retargeter._initial_learned_offsets)
        retargeter.ground_offset = 0.0
        adjustments = initialize_retargeter(retargeter)

    _RETARGETER_CACHE[key] = retargeter
    while len(_RETARGETER_CACHE) > _RETARGETER_CACHE_SIZE:
        _RETARGETER_CACHE.popitem(last=False)

    return retargeter, adjustments


def load_source_motion(
    motion_path: str | Path,
    config: dict,
    *,
    smplh_model_dir: str | Path = SMPLH_MODEL_DIR,
    fitted_shape_path: str | Path | None = None,
) -> tuple[list[dict], float, dict]:
    """
    Load subject SMPLH targets using GMR's AMASS processing.

    Returns:
        Per-frame targets, GMR-aligned FPS, and source metadata.
    """

    motion_path = Path(motion_path).expanduser().resolve()
    smplh_model_dir = Path(smplh_model_dir).expanduser().resolve()

    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    if not smplh_model_dir.is_dir():
        raise FileNotFoundError(f"SMPLH model directory not found: {smplh_model_dir}")

    # Validate AMASS input before invoking the upstream loader.
    with np.load(motion_path, allow_pickle=True) as data:
        required = {
            "poses",
            "trans",
            "betas",
            "gender",
            "mocap_framerate",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing motion fields: {sorted(missing)}")

        poses = np.asarray(data["poses"])
        trans = np.asarray(data["trans"])
        betas = np.asarray(data["betas"])

        if poses.ndim != 2 or poses.shape[1] != 156:
            raise ValueError("AMASS poses must have shape (T, 156)")

        num_frames = poses.shape[0]
        if num_frames < 2:
            raise ValueError("Motion must contain at least two frames")
        if trans.shape != (num_frames, 3):
            raise ValueError("AMASS trans must have shape (T, 3)")

        num_betas = config["smplh"]["num_betas"]
        if betas.shape not in ((num_betas,), (1, num_betas)):
            raise ValueError(
                f"Expected one subject shape with {num_betas} betas, got {betas.shape}"
            )

        for name, values in (
            ("poses", poses),
            ("trans", trans),
            ("betas", betas),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {name}")

        source_fps = float(np.asarray(data["mocap_framerate"]).item())
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Source FPS must be finite and positive")

        source_gender = np.asarray(data["gender"]).item()
        if isinstance(source_gender, bytes):
            source_gender = source_gender.decode("utf-8")
        source_gender = str(source_gender).strip().lower()

        if source_gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Invalid motion gender: {source_gender!r}")

        source_betas = betas.reshape(-1).tolist()

    if config["gmr"]["use_fitted_shape"] != (fitted_shape_path is not None):
        raise ValueError("Fitted shape flag and loader path must agree")
    fit_metadata = (
        validate_fitted_shape(fitted_shape_path)
        if fitted_shape_path is not None
        else None
    )
    gender_mode = config["smplh"]["gender_mode"]
    if fit_metadata is not None and gender_mode != "neutral":
        raise ValueError("Target-fitted motion loading requires neutral SMPL-H")
    body_model_gender = source_gender if gender_mode == "motion" else "neutral"

    target_fps = config["gmr"]["target_fps"]

    # Upstream does not actually upsample when target_fps > source_fps.
    if target_fps > source_fps:
        raise ValueError(
            f"Upsampling is not supported by this integration: {source_fps} -> {target_fps} FPS"
        )

    if target_fps < source_fps:
        frame_skip = int(source_fps / target_fps)
        if num_frames // frame_skip < 2:
            raise ValueError("Motion is too short for the requested FPS")

    with torch.inference_mode():
        smplh_data, body_model, smplh_output, _ = load_smplh_file(
            str(motion_path),
            str(smplh_model_dir),
            fitted_shape_path=str(fitted_shape_path)
            if fitted_shape_path is not None
            else None,
            body_model_gender=body_model_gender,
        )

        extract_frames = (
            acceleration.smplh_frames
            if config.get("performance", {}).get("fast_frames", False)
            else get_smplh_data_offline_fast
        )
        frames, aligned_fps = extract_frames(
            smplh_data,
            body_model,
            smplh_output,
            tgt_fps=target_fps,
        )

    # Upstream interpolation spans both endpoints of the source motion.
    if target_fps < source_fps:
        sampling_fps = (len(frames) - 1) * source_fps / (num_frames - 1)
    else:
        sampling_fps = source_fps

    metadata = {
        "motion_path": str(motion_path),
        "source_gender": source_gender,
        "body_model_gender": body_model_gender,
        "betas": source_betas,  # Source provenance; not the effective fitted shape.
        "shape_mode": "target_fitted" if fit_metadata is not None else "source",
        "fitted_shape_path": str(fitted_shape_path)
        if fitted_shape_path is not None
        else None,
        "source_fps": source_fps,
        "requested_fps": target_fps,
        "aligned_fps": float(aligned_fps),
        "source_num_frames": num_frames,
        "aligned_num_frames": len(frames),
        "sampling_fps": float(sampling_fps),
    }

    return frames, float(aligned_fps), metadata


def retarget_frames(
    retargeter: GeneralMotionRetargeting,
    frames: list[dict],
    *,
    offset_to_ground: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Retarget consecutive SMPLH frames with a shared IK state.

    The retargeter must already be initialized for this motion.

    Returns:
        qpos: MuJoCo configurations, shape (T, nq).
        position_errors: Stage-two body position errors in meters,
            shape (T, num_error_bodies), before postprocessing.
        error_body_names: Body names corresponding to error columns.
    """

    if not frames:
        raise ValueError("Cannot retarget an empty motion")

    # Match the exact task order used by GMR when returning errors.
    frame_tasks = [
        task for task in retargeter.tasks2 if isinstance(task, mink.FrameTask)
    ]
    if not frame_tasks:
        raise ValueError("GMR requires stage-two frame tasks")

    error_body_names = [task.frame_name for task in frame_tasks]

    num_frames = len(frames)
    num_bodies = len(error_body_names)

    qpos = np.empty(
        (num_frames, retargeter.model.nq),
        dtype=np.float64,
    )
    position_errors = np.empty(
        (num_frames, num_bodies),
        dtype=np.float64,
    )

    # retarget
    for frame_index, frame in enumerate(frames):
        try:
            frame_qpos, frame_errors = retargeter.retarget(
                frame,
                offset_to_ground=offset_to_ground,
            )
        except Exception as error:
            raise RuntimeError(
                f"GMR failed at frame {frame_index} (zero-based, total={num_frames})"
            ) from error

        frame_qpos = np.asarray(frame_qpos)
        frame_errors = np.asarray(frame_errors)

        # Validate outputs
        if frame_qpos.shape != (retargeter.model.nq,):
            raise ValueError(
                f"Unexpected qpos shape at frame {frame_index}: {frame_qpos.shape}"
            )
        if frame_errors.shape != (num_bodies,):
            raise ValueError(
                f"Unexpected error shape at frame {frame_index}: {frame_errors.shape}"
            )

        if not np.isfinite(frame_qpos).all():
            raise ValueError(f"Non-finite qpos at frame {frame_index}")
        if not np.isfinite(frame_errors).all():
            raise ValueError(f"Non-finite position errors at frame {frame_index}")

        qpos[frame_index] = frame_qpos
        position_errors[frame_index] = frame_errors

    return qpos, position_errors, error_body_names


def compute_subject_offsets(
    robot_xml_path: str | Path,
    source_metadata: dict,
    *,
    smplh_model_dir: str | Path = SMPLH_MODEL_DIR,
) -> dict:
    """
    Align the original SMPLH shape to an embodiment without shape fitting.

    Results are cached per process by XML, SMPLH gender and betas; every
    motion of a subject therefore reuses one computation.
    """
    robot_xml_path = Path(robot_xml_path).expanduser().resolve()
    smplh_model_dir = Path(smplh_model_dir).expanduser().resolve()

    key = (
        str(robot_xml_path),
        robot_xml_path.stat().st_mtime_ns,
        str(smplh_model_dir),
        source_metadata["body_model_gender"],
        tuple(float(beta) for beta in source_metadata["betas"]),
    )
    if key in _OFFSETS_CACHE:
        _OFFSETS_CACHE.move_to_end(key)
        return deepcopy(_OFFSETS_CACHE[key])

    offsets = _compute_subject_offsets(robot_xml_path, source_metadata, smplh_model_dir)

    _OFFSETS_CACHE[key] = deepcopy(offsets)
    while len(_OFFSETS_CACHE) > _OFFSETS_CACHE_SIZE:
        _OFFSETS_CACHE.popitem(last=False)

    return offsets


def _compute_subject_offsets(
    robot_xml_path: Path,
    source_metadata: dict,
    smplh_model_dir: Path,
) -> dict:
    """Run GMR's reference-pose alignment for one subject."""

    ik_config_path = Path(IK_CONFIG_DICT["smplh"]["myofullbody"])
    with ik_config_path.open("r", encoding="utf-8") as file:
        ik_config = json.load(file)

    positions, joint_names, rotations, human_names = get_robot_tpose_targets(
        str(robot_xml_path),
        "myofullbody",
        ik_config,
        include_rotations=True,
    )

    parser = cached_smplh_parser(
        model_path=str(smplh_model_dir),
        gender=source_metadata["body_model_gender"],
        num_betas=16,
        use_pca=False,
    )
    betas = torch.as_tensor(
        source_metadata["betas"],
        dtype=torch.float32,
    ).reshape(1, 16)

    with torch.inference_mode():
        offsets = compute_alignment_offsets(
            smpl_parser=parser,
            shape=betas,
            scale=torch.ones(1, dtype=torch.float32),
            smpl_joint_indices=get_smpl_tpose_indices(joint_names),
            human_joint_names=human_names,
            target_positions=positions,
            target_rotations=rotations,
            robot_type="myofullbody",
        )

    return offsets


def retarget_motion(
    motion_path: str | Path,
    robot_xml_path: str | Path,
    config: dict,
    *,
    smplh_model_dir: str | Path = SMPLH_MODEL_DIR,
) -> dict:
    """
    Retarget one complete AMASS motion to an explicit embodiment.

    Args:
        motion_path: Source subject's AMASS NPZ file.
        robot_xml_path: Corresponding embodiment XML.
        config: Validated configuration returned by load_config().
        smplh_model_dir: Directory containing SMPLH model files.

    Returns:
        Trajectory arrays, model indexing information, and metadata.
        No files are written.
    """

    motion_path = Path(motion_path).expanduser().resolve()
    robot_xml_path = Path(robot_xml_path).expanduser().resolve()
    smplh_model_dir = Path(smplh_model_dir).expanduser().resolve()

    fitted_shape_path = None
    fit_metadata = None
    if config["gmr"]["use_fitted_shape"]:
        fitted_shape_path = ensure_fitted_shape(
            robot_xml_path,
            Path(IK_CONFIG_DICT["smplh"]["myofullbody"]),
            smplh_model_dir,
            config["shape_fitting"],
        )
        fit_metadata = validate_fitted_shape(fitted_shape_path)

    # Reuse the model while restoring this fit and resetting motion state.
    retargeter, adjustments = get_retargeter(
        robot_xml_path,
        config,
        fitted_shape_path=fitted_shape_path,
    )
    model = retargeter.model

    frames, fps, source_metadata = load_source_motion(
        motion_path,
        config,
        smplh_model_dir=smplh_model_dir,
        fitted_shape_path=fitted_shape_path,
    )

    # Central-difference velocities require at least three frames.
    if len(frames) < 3:
        raise ValueError(
            f"At least three aligned frames are required, got {len(frames)} for {motion_path}"
        )

    # Compute subject offsets
    subject_offsets = None

    if (
        fitted_shape_path is None
        and config["gmr"]["offsets_mode"] == "subject_specific"
    ):
        subject_offsets = compute_subject_offsets(
            robot_xml_path,
            source_metadata,
            smplh_model_dir=smplh_model_dir,
        )
        retargeter.learned_offsets = subject_offsets

    # Retarget
    qpos, position_errors, error_body_names = retarget_frames(
        retargeter,
        frames,
        offset_to_ground=config["gmr"]["offset_to_ground"],
    )

    # Resample to export.fps
    ik_fps = fps
    ik_num_frames = len(qpos)

    qpos = resample_qpos(
        model,
        qpos,
        source_fps=source_metadata["sampling_fps"],
        target_fps=config["export"]["fps"],
    )
    fps = float(config["export"]["fps"])

    # Postprocess
    ground_report = None
    if config["postprocess"]["ground_correction"]:
        qpos, ground_report = correct_ground(model, qpos)

    # Differentiate the final positions, not the pre-correction positions.
    qpos, qvel = compute_qvel(model, qpos, fps)

    metadata = {
        "motion_path": str(motion_path),
        "robot_xml_path": str(robot_xml_path),
        "smplh_model_dir": str(smplh_model_dir),
        "config": deepcopy(config),
        "source": source_metadata,
        "shape_fitting": {"path": str(fitted_shape_path), "metadata": fit_metadata}
        if fitted_shape_path is not None
        else None,
        "initialization_adjustments": adjustments,
        "ik_fps": ik_fps,
        "ik_num_frames": ik_num_frames,
        "output_fps": fps,
        "position_error_sampling": "original_ik_frames",
        "ground_correction": ground_report,
        "output_num_frames": len(qpos),
        "trimmed_start_frames": 1,
        "trimmed_end_frames": 1,
        "quaternion_order": "wxyz",
        "position_error_stage": "before_ground_correction",
        "alignment": {
            "mode": config["gmr"]["offsets_mode"],
            "reference_pose": (
                "gmr_upstream_tpose"
                if subject_offsets is not None or fitted_shape_path is not None
                else None
            ),
            "offsets": retargeter.learned_offsets,
        },
    }

    return {
        "qpos": qpos,
        "qvel": qvel,
        "fps": fps,
        "position_errors": position_errors,
        "error_body_names": error_body_names,
        "joint_names": [model.joint(joint_id).name for joint_id in range(model.njnt)],
        "jnt_qposadr": model.jnt_qposadr.copy(),
        "jnt_dofadr": model.jnt_dofadr.copy(),
        "body_names": [model.body(body_id).name for body_id in range(model.nbody)],
        "metadata": metadata,
    }
