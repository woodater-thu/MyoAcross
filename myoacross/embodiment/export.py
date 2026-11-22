"""
Format embodiment reports and export validated model artifacts.

Each subject directory contains model.xml and report.json on success,
or only report.json on optimization failure.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from myoacross.utils.paths import ASSETS_DIR, EMBODIMENTS_DIR


def build_report(result: dict) -> dict:
    """Build the report.json."""

    # Basic fields
    failed = result.get("status") == "optimization_failed"
    optimization = result.get("optimization", {})
    document = {
        "strategy": result["strategy"],
        "subject_id": result["subject_id"],
        "status": "failed" if failed else "passed",
        "units": {"length": "m", "mass": "kg", "scale": "dimensionless"},
    }
    for key in ("baseline_path", "config_path", "config", "timestamp_utc"):
        if key in result:
            document[key] = deepcopy(result[key])

    # Scales and optiminzation field
    if optimization:
        if "uniform_scale" in optimization:
            document["scales"] = {
                "uniform": optimization["uniform_scale"],
                "fixed_bodies": {"Full Body": 1.0},
                "bound_active": optimization["bound_active"],
            }
        else:
            document["scales"] = {
                "regions": deepcopy(optimization["region_scales"]),
                "bound_regions": list(optimization["bound_regions"]),
            }
        document["optimization"] = {
            key: deepcopy(optimization[key])
            for key in (
                "solver_success",
                "message",
                "nfev",
                "regularization",
                "prior_scale",
                "prior_scale_basis",
                "quality",
                "method",
                "cost",
                "optimality",
            )
            if key in optimization
        }
    else:
        scales = result["body_scales"]
        values = {float(value) for name, value in scales.items() if name != "Full Body"}
        if len(values) == 1:
            document["scales"] = {
                "uniform": values.pop(),
                "fixed_bodies": {"Full Body": 1.0},
            }
        else:
            # Preserve all scales for any nonuniform strategy without regional diagnostics.
            document["scales"] = {"bodies": deepcopy(scales)}

    # Measurement field
    if failed:
        document["error"] = result["error"]
        document["measurement_stage"] = "optimization_candidate"
        source_measurements = optimization["measurements"]
    else:
        validation = result["validation"]

        document["checks"] = {
            "mass_inertia_passed": validation["mass_inertia_checks_passed"],
            "tendon_muscle_passed": validation["tendon_muscle_checks_passed"],
            **(
                {"measurement_quality_passed": optimization["quality"]["passed"]}
                if "quality" in optimization
                else {}
            ),
        }
        document["mass"] = {
            "actual_kg": validation["body_mass_kg"],
            "target_kg": validation["target_body_mass_kg"],
            "error_kg": validation["body_mass_error_kg"],
            "density_scale": validation["density_scale"],
        }
        document["measurement_stage"] = "final_compiled_model"
        source_measurements = validation["measurements"]

    # Format measurements consistently for both outcomes.
    fit_measurements = optimization.get("measurements", {})
    measurements = {}

    for name, values in source_measurements.items():
        actual = float(values["predicted"])
        target = values.get("target")
        target = None if target is None else float(target)
        error = None if target is None else actual - target

        entry = {
            "actual_m": actual,
            "target_m": target,
            "error_m": error,
            "used_for_scaling": (
                name in fit_measurements
                if optimization
                else result["strategy"] == "height_scale" and name == "height"
            ),
        }

        fit = fit_measurements.get(name)
        if fit is not None:
            tolerance = float(fit["tolerance_m"])
            entry.update(
                {
                    "tolerance_m": tolerance,
                    "normalized_abs_error": abs(error) / tolerance,
                }
            )

        if fit is not None and "acceptance_limit_m" in fit:
            entry["acceptance_limit_m"] = float(fit["acceptance_limit_m"])

        measurements[name] = entry

    document["measurements"] = measurements
    return document


def save(
    spec: mujoco.MjSpec,
    report: dict,
    *,
    output_root: str | Path = EMBODIMENTS_DIR,
    overwrite: bool = False,
) -> Path:
    """Save model XML and report under the strategy and subject directory."""

    strategy = report["strategy"]
    subject_id = report["subject_id"]

    if strategy not in {"baseline", "height_scale", "segment_scale"}:
        raise ValueError(f"Unknown embodiment strategy: {strategy!r}")

    # Resolve output dir
    output_dir = Path(output_root) / strategy
    if strategy != "baseline":
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError(f"{strategy} requires a nonempty subject ID")
        if subject_id in {".", ".."} or "/" in subject_id or "\\" in subject_id:
            raise ValueError(f"Invalid subject ID for output directory: {subject_id!r}")

        output_dir /= subject_id

    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")

    # Resolve asset directories relative to the original model
    source_dir = Path(report["baseline_path"]).resolve().parent
    export_spec = spec.copy()
    export_spec.compiler.meshdir = str(
        (source_dir / export_spec.compiler.meshdir).resolve()
    )
    export_spec.compiler.texturedir = str(
        (source_dir / export_spec.compiler.texturedir).resolve()
    )

    # Replace the existing skybox with the MyoAcross background
    background_path = ASSETS_DIR / "myoacross_background.png"
    if not background_path.is_file():
        raise FileNotFoundError(f"Background image not found: {background_path}")

    skybox = export_spec.texture("sky")
    if skybox.type != mujoco.mjtTexture.mjTEXTURE_SKYBOX:
        raise ValueError("Texture 'sky' is not a skybox")

    skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
    skybox.file = str(background_path.resolve())
    skybox.gridsize[:] = [1, 1]

    # Compile model
    model = export_spec.compile()
    root = ET.fromstring(export_spec.to_xml())

    # Explicitly write the inertial parameters to preserve precision.
    for body in root.findall(".//body"):
        inertial = body.find("inertial")
        if inertial is None:
            continue

        body_id = model.body(body.attrib["name"]).id
        inertial.set("mass", format(model.body_mass[body_id], ".17g"))
        inertial.set(
            "pos", " ".join(format(value, ".17g") for value in model.body_ipos[body_id])
        )
        inertial.set(
            "quat",
            " ".join(format(value, ".17g") for value in model.body_iquat[body_id]),
        )
        inertial.set(
            "diaginertia",
            " ".join(format(value, ".17g") for value in model.body_inertia[body_id]),
        )

    xml_text = ET.tostring(root, encoding="unicode")

    # Check the serialized XML before writing any output
    reloaded = mujoco.MjModel.from_xml_string(xml_text)

    if reloaded.nbody != model.nbody:
        raise ValueError("Reloaded model body count differs from the generated model")
    if not np.allclose(reloaded.body_mass, model.body_mass, rtol=1e-9, atol=1e-12):
        raise ValueError("Reloaded model masses differ from the generated model")
    if not np.allclose(
        reloaded.body_inertia, model.body_inertia, rtol=1e-9, atol=1e-12
    ):
        raise ValueError("Reloaded model inertias differ from the generated model")

    # Write report.json
    report_text = json.dumps(
        build_report(report), indent=2, ensure_ascii=False, allow_nan=False
    )

    # Overwrite only model.xml and report.json when explicitly enabled
    output_dir.mkdir(parents=True, exist_ok=overwrite)

    model_path = output_dir / "model.xml"
    model_path.write_text(xml_text, encoding="utf-8")
    (output_dir / "report.json").write_text(report_text + "\n", encoding="utf-8")

    return model_path


def save_failure_report(
    report: dict,
    *,
    config: dict,
    config_path: str | Path,
    output_root: str | Path = EMBODIMENTS_DIR,
    overwrite: bool = False,
) -> Path:
    """Save an unsuccessful optimization without exporting a model."""

    # Validate inputs
    if report.get("status") != "optimization_failed":
        raise ValueError("Expected an optimization failure report")

    strategy = report["strategy"]
    subject_id = report["subject_id"]

    if strategy not in {"segment_scale", "height_scale"}:
        raise ValueError(f"Unsupported failed optimization strategy: {strategy!r}")
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("Failure report requires a subject ID")
    if subject_id in {".", ".."} or "/" in subject_id or "\\" in subject_id:
        raise ValueError(f"Invalid subject ID: {subject_id!r}")

    output_dir = Path(output_root) / strategy / subject_id
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")

    # Do not leave an old successful XML beside a failed report.
    if (output_dir / "model.xml").exists():
        raise FileExistsError(
            f"Cannot save failure over an existing model: {output_dir}; use a new output root"
        )

    # Save failure reports (use build_report())
    document = {
        **report,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config": config,
    }
    text = json.dumps(
        build_report(document), indent=2, ensure_ascii=False, allow_nan=False
    )

    output_dir.mkdir(parents=True, exist_ok=overwrite)
    report_path = output_dir / "report.json"

    with report_path.open("w" if overwrite else "x", encoding="utf-8") as file:
        file.write(text + "\n")

    return report_path
