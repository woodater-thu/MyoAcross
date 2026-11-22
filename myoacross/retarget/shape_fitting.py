"""Fit neutral SMPL-H to an explicit target model, once per model/configuration."""

import hashlib
import inspect
import json
import os
import tempfile
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.utils.smpl import SMPLH_Parser
from general_motion_retargeting.utils.shape_fitting import (
    get_robot_tpose_targets,
    get_smpl_tpose_indices,
    fit_smpl_shape_to_robot,
    compute_alignment_offsets,
    save_fitted_shape,
    load_fitted_shape,
)
from myoacross.embodiment.model import model_sha256
from myoacross.utils.paths import RETARGETS_DIR


@lru_cache(maxsize=64)
def _file_digest(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    # Stat values invalidate the in-process digest cache after file changes.
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(path: Path) -> str:
    stat = path.stat()
    return _file_digest(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    )


def fitting_fingerprint(
    robot_xml_path, ik_config_path, smplh_model_dir, options
) -> dict:
    """Hash compiled model (including includes/assets), SMPL files and fitting inputs."""
    compiled_model_hash = model_sha256(robot_xml_path)
    root = Path(smplh_model_dir).resolve()
    model_files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in (".npz", ".pkl")
    )
    if not model_files:
        raise FileNotFoundError(f"No SMPL-H model files under {root}")
    return {
        "schema": 1,
        "compiled_model_sha256": compiled_model_hash,
        "ik_sha256": _digest(Path(ik_config_path)),
        "smplh_files": {str(p.relative_to(root)): _digest(p) for p in model_files},
        "gender": "neutral",
        "num_betas": 16,
        "robot_type": "myofullbody",
        "device": "cpu",
        "options": dict(options),
        "gmr_version": version("general-motion-retargeting"),
        "mujoco_version": mujoco.__version__,
        # Package versions alone may not distinguish different forks.
        "gmr_shape_code_sha256": _digest(
            Path(inspect.getfile(fit_smpl_shape_to_robot))
        ),
    }


def validate_fitted_shape(path: str | Path) -> dict:
    """Reject broken caches and upstream's fallback to global offsets."""
    path = Path(path)
    metadata = json.loads(path.with_name(path.stem + "_metadata.json").read_text())
    offsets = metadata.get("local_offsets")
    names = metadata.get("human_joint_names")
    if not names or not isinstance(offsets, dict):
        raise ValueError(f"Missing local fitted offsets: {path}")
    for field, width in (("pos_offsets", 3), ("rot_offsets", 4)):
        values = offsets.get(field, {})
        for name in names:
            array = np.asarray(values.get(name, []), dtype=float)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {field} for {name}: {path}")
            if width == 4 and not np.isclose(np.linalg.norm(array), 1.0, atol=1e-3):
                raise ValueError(f"Invalid offset quaternion for {name}: {path}")
    shape, scale, offset_z, height_scale, _ = load_fitted_shape(str(path))
    if np.asarray(shape).shape not in ((16,), (1, 16)) or not np.isfinite(shape).all():
        raise ValueError(f"Invalid fitted betas: {path}")
    if (
        np.asarray(scale).size != 1
        or not np.isfinite(scale).all()
        or float(np.asarray(scale).item()) <= 0
    ):
        raise ValueError(f"Invalid fitted scale: {path}")
    if not np.isfinite(offset_z) or not np.isfinite(height_scale) or height_scale <= 0:
        raise ValueError(f"Invalid fitted translation scaling: {path}")
    metrics = metadata.get("metrics", {})
    loss = metrics.get("final_loss")
    if not isinstance(loss, (int, float)) or not np.isfinite(loss) or loss < 0:
        raise ValueError(f"Invalid fitting metrics: {path}")
    return metadata


def _fit_and_save(robot_xml_path, ik_config_path, smplh_model_dir, options, save_path):
    ik = json.loads(Path(ik_config_path).read_text())
    positions, joints, rotations, human_names = get_robot_tpose_targets(
        str(robot_xml_path),
        "myofullbody",
        ik,
        include_rotations=True,
    )
    indices = get_smpl_tpose_indices(joints)
    parser = SMPLH_Parser(
        model_path=str(smplh_model_dir), gender="neutral", num_betas=16, use_pca=False
    )
    # Optimization must run with gradients; do not wrap this in inference_mode.
    shape, scale, pos, rot, offset_z, height_scale, metrics = fit_smpl_shape_to_robot(
        smpl_parser=parser,
        target_positions=positions,
        smpl_joint_indices=indices,
        target_rotations=rotations,
        iterations=options["iterations"],
        lr=options["lr"],
        device="cpu",
        robot_type="myofullbody",
        verbose=False,
    )
    offsets = compute_alignment_offsets(
        smpl_parser=parser,
        shape=shape,
        scale=scale,
        smpl_joint_indices=indices,
        human_joint_names=human_names,
        target_positions=positions,
        target_rotations=rotations,
        robot_type="myofullbody",
    )
    save_fitted_shape(
        shape=shape,
        scale=scale,
        smpl2robot_pos=pos,
        smpl2robot_rot_mat=rot,
        offset_z=offset_z,
        height_scale=height_scale,
        metrics=metrics,
        save_path=str(save_path),
        human_joint_names=human_names,
        local_offsets=offsets,
    )


def ensure_fitted_shape(
    robot_xml_path: str | Path,
    ik_config_path: str | Path,
    smplh_model_dir: str | Path,
    options: dict,
    *,
    cache_root: str | Path = RETARGETS_DIR / "fitted_shapes",
) -> Path:
    """Return a validated cache; publish each complete fit atomically.

    Equal compiled models share a cache, including the common baseline.
    A finite non-converged fit is recorded with a warning, matching MuscleMimic;
    it is not a declaration that the resulting RL motion has been accepted.
    """
    fingerprint = fitting_fingerprint(
        robot_xml_path, ik_config_path, smplh_model_dir, options
    )
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    root = Path(cache_root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / key
    path = folder / "shape.pkl"
    # Lock per fingerprint: different bodies may fit concurrently, identical ones cannot.
    import fcntl

    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if folder.exists():
            if json.loads((folder / "fingerprint.json").read_text()) != fingerprint:
                raise ValueError(f"Fitted cache identity mismatch: {folder}")
            validate_fitted_shape(path)
            return path
        with tempfile.TemporaryDirectory(prefix=".fit-", dir=root) as tmp:
            staging = Path(tmp) / "result"
            staging.mkdir()
            _fit_and_save(
                robot_xml_path,
                ik_config_path,
                smplh_model_dir,
                options,
                staging / "shape.pkl",
            )
            metadata = validate_fitted_shape(staging / "shape.pkl")
            (staging / "fingerprint.json").write_text(
                json.dumps(fingerprint, indent=2) + "\n"
            )
            os.rename(staging, folder)
        if not metadata["metrics"].get("converged", False):
            print(
                f"WARNING: shape fit did not converge: {path}; metrics={metadata['metrics']}",
                flush=True,
            )
    return path
