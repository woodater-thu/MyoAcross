"""Load and validate retargeting configurations."""

from pathlib import Path
import math

from myoacross.utils.config import load_config as compose_config


def load_config(config_path: str | Path) -> dict:
    """Load and validate the GMR-Fit configuration."""

    config = compose_config(config_path)

    allowed_sections = {
        "name",
        "smplh",
        "gmr",
        "postprocess",
        "export",
        "shape_fitting",
        "performance",
    }
    unknown = set(config) - allowed_sections
    if unknown:
        raise ValueError(f"Unknown configuration fields: {list(unknown)}")

    if config.get("name") != "gmr_fit":
        raise ValueError("Retarget configuration name must be 'gmr_fit'")

    defaults = {
        "smplh": {
            "gender_mode": "motion",
            "num_betas": 16,
        },
        "gmr": {
            "use_fitted_shape": False,
            "solver": "daqp",
            "damping": 0.5,
            "target_fps": 50,
            "offsets_mode": "default",
            "use_velocity_limit": False,
            "offset_to_ground": False,
            "verbose": False,
        },
        "shape_fitting": {
            "iterations": 500,
            "lr": 0.001,
        },
        "postprocess": {
            "ground_correction": True,
        },
        "export": {
            "fps": 50,
        },
        # Exact reimplementations; set false to run the upstream code paths.
        "performance": {
            "fast_ik": True,
            "fast_frames": True,
        },
    }

    for section, section_defaults in defaults.items():
        values = config.get(section, {})

        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")

        unknown = set(values) - set(section_defaults)
        if unknown:
            raise ValueError(f"Unknown {section} options: {list(unknown)}")

        config[section] = {**section_defaults, **values}

    smplh = config["smplh"]
    if smplh["gender_mode"] not in ("motion", "neutral"):
        raise ValueError("smplh.gender_mode must be 'motion' or 'neutral'")

    # The current GMR-Fit fork constructs SMPLH with 16 betas.
    if type(smplh["num_betas"]) is not int or smplh["num_betas"] != 16:
        raise ValueError("The current integration requires smplh.num_betas=16")

    gmr = config["gmr"]
    if not isinstance(gmr["solver"], str) or not gmr["solver"].strip():
        raise ValueError("gmr.solver must be a nonempty string")
    if gmr["offsets_mode"] not in ("default", "subject_specific", "fitted"):
        raise ValueError(
            "gmr.offsets_mode must be 'default', 'subject_specific', or 'fitted'"
        )

    damping = gmr["damping"]
    if type(damping) not in (int, float) or not math.isfinite(damping) or damping < 0:
        raise ValueError("gmr.damping must be finite and nonnegative")

    target_fps = gmr["target_fps"]
    if type(target_fps) is not int or target_fps <= 0:
        raise ValueError("gmr.target_fps must be a positive integer")

    for field in (
        "use_velocity_limit",
        "offset_to_ground",
        "verbose",
        "use_fitted_shape",
    ):
        if type(gmr[field]) is not bool:
            raise ValueError(f"gmr.{field} must be a boolean")

    if type(config["postprocess"]["ground_correction"]) is not bool:
        raise ValueError("postprocess.ground_correction must be a boolean")

    for field in ("fast_ik", "fast_frames"):
        if type(config["performance"][field]) is not bool:
            raise ValueError(f"performance.{field} must be a boolean")

    export_fps = config["export"]["fps"]
    if type(export_fps) is not int or export_fps <= 0:
        raise ValueError("export.fps must be a positive integer")

    if gmr["use_fitted_shape"]:
        if smplh["gender_mode"] != "neutral" or gmr["offsets_mode"] != "fitted":
            raise ValueError(
                "Fitted shape requires neutral SMPL-H and offsets_mode=fitted"
            )
    elif gmr["offsets_mode"] == "fitted":
        raise ValueError("offsets_mode=fitted requires use_fitted_shape=true")
    fit = config["shape_fitting"]
    if type(fit["iterations"]) is not int or fit["iterations"] <= 0:
        raise ValueError("shape_fitting.iterations must be a positive integer")
    if (
        type(fit["lr"]) not in (int, float)
        or not math.isfinite(fit["lr"])
        or fit["lr"] <= 0
    ):
        raise ValueError("shape_fitting.lr must be finite and positive")
    return config
