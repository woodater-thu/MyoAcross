"""Region-based scaling for subject-specific MyoFullBody models."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from myoacross.embodiment.smplh_to_measurements import (
    JOINT_MEASUREMENTS,
    JOINT_HEIGHT_MEASUREMENTS,
)
from myoacross.utils.paths import MEASUREMENTS_DIR

from myoacross.embodiment.model import EmbodimentModel, OptimizationFailure
from myoacross.embodiment.measurements import (
    load_subject_measurements,
    compute_myofullbody_landmarks,
    compute_myofullbody_measurements,
)


# Mapping MyoFullBody bodies to scaling regions
BODY_REGION_MAP: dict[str, str] = {
    # Fixed global root
    "Full Body": "global_root",
    # Pelvis and sacrum
    "sacrum": "pelvis_sacrum",
    "pelvis": "pelvis_sacrum",
    # Trunk: lumbar spine, abdomen, thorax and attachment bodies
    "lumbar5": "trunk",
    "Abdomen": "trunk",
    "lumbar4": "trunk",
    "lumbar3": "trunk",
    "lumbar2": "trunk",
    "lumbar1": "trunk",
    "torso": "trunk",
    "bimanual_attachment": "trunk",
    "root": "trunk",
    "thorax": "trunk",
    "cervical_spine": "trunk",
    "head_attach": "trunk",
    "neck": "trunk",
    "head": "trunk",
    # Shoulder girdle: shared bilateral scale
    "clavicle_r": "shoulder_girdle",
    "clavphant_r": "shoulder_girdle",
    "scapula_r": "shoulder_girdle",
    "scapphant_r": "shoulder_girdle",
    "humphant_r": "shoulder_girdle",
    "humphant1_r": "shoulder_girdle",
    "clavicle_l": "shoulder_girdle",
    "clavphant_l": "shoulder_girdle",
    "scapula_l": "shoulder_girdle",
    "scapphant_l": "shoulder_girdle",
    "humphant_l": "shoulder_girdle",
    "humphant1_l": "shoulder_girdle",
    # Upper arms
    "humerus_r": "upper_arm_r",
    "humerus_l": "upper_arm_l",
    # Right forearm and hand
    "ulna_r": "forearm_r",
    "radius_r": "forearm_r",
    "lunate_r": "forearm_r",
    "scaphoid_r": "forearm_r",
    "pisiform_r": "forearm_r",
    "triquetrum_r": "forearm_r",
    "capitate_r": "forearm_r",
    "trapezium_r": "forearm_r",
    "firstmc_r": "forearm_r",
    "proximal_thumb_r": "forearm_r",
    "distal_thumb_r": "forearm_r",
    "trapezoid_r": "forearm_r",
    "hamate_r": "forearm_r",
    "secondmc_r": "forearm_r",
    "2proxph_r": "forearm_r",
    "2midph_r": "forearm_r",
    "2distph_r": "forearm_r",
    "thirdmc_r": "forearm_r",
    "3proxph_r": "forearm_r",
    "3midph_r": "forearm_r",
    "3distph_r": "forearm_r",
    "fourthmc_r": "forearm_r",
    "4proxph_r": "forearm_r",
    "4midph_r": "forearm_r",
    "4distph_r": "forearm_r",
    "fifthmc_r": "forearm_r",
    "5proxph_r": "forearm_r",
    "5midph_r": "forearm_r",
    "5distph_r": "forearm_r",
    # Left forearm and hand
    "ulna_l": "forearm_l",
    "radius_l": "forearm_l",
    "lunate_l": "forearm_l",
    "scaphoid_l": "forearm_l",
    "pisiform_l": "forearm_l",
    "triquetrum_l": "forearm_l",
    "capitate_l": "forearm_l",
    "trapezium_l": "forearm_l",
    "firstmc_l": "forearm_l",
    "proximal_thumb_l": "forearm_l",
    "distal_thumb_l": "forearm_l",
    "trapezoid_l": "forearm_l",
    "hamate_l": "forearm_l",
    "secondmc_l": "forearm_l",
    "2proxph_l": "forearm_l",
    "2midph_l": "forearm_l",
    "2distph_l": "forearm_l",
    "thirdmc_l": "forearm_l",
    "3proxph_l": "forearm_l",
    "3midph_l": "forearm_l",
    "3distph_l": "forearm_l",
    "fourthmc_l": "forearm_l",
    "4proxph_l": "forearm_l",
    "4midph_l": "forearm_l",
    "4distph_l": "forearm_l",
    "fifthmc_l": "forearm_l",
    "5proxph_l": "forearm_l",
    "5midph_l": "forearm_l",
    "5distph_l": "forearm_l",
    # Thighs and patellae
    "femur_r": "thigh_r",
    "patella_r": "thigh_r",
    "femur_l": "thigh_l",
    "patella_l": "thigh_l",
    # Shanks
    "tibia_r": "shank_r",
    "tibia_l": "shank_l",
    # Feet
    "talus_r": "foot_r",
    "talus_l": "foot_l",
    "calcn_r": "foot_r",
    "toes_r": "foot_r",
    "calcn_l": "foot_l",
    "toes_l": "foot_l",
}


# MyoFullBody optimization regions
OPTIMIZATION_REGIONS: tuple[str, ...] = (
    "pelvis_sacrum",
    "trunk",
    "shoulder_girdle",
    "upper_arm_r",
    "upper_arm_l",
    "forearm_r",
    "forearm_l",
    "thigh_r",
    "thigh_l",
    "shank_r",
    "shank_l",
    "foot_r",
    "foot_l",
)


class SegmentScaleModel(EmbodimentModel):
    """Fit regional MyoFullBody scales to subject measurements."""

    def __init__(
        self,
        baseline_spec: mujoco.MjSpec,
        subject_id: str,
        *,
        measurements_dir: str | Path = MEASUREMENTS_DIR,
    ) -> None:
        """Initialize the baseline and load subject measurements."""

        self.subject_id = str(subject_id).strip()
        self.measurements = load_subject_measurements(
            self.subject_id,
            measurements_dir=measurements_dir,
        )

        super().__init__(baseline_spec)

    def solve(
        self,
        *,
        measurement_tolerances: dict[str, float] | None = None,
        scale_bounds: tuple[float, float] = (0.7, 1.3),
        prior_weight: float = 1.0,
        sigma_symmetry: float = 0.03,
        sigma_adjacency: float = 0.15,
        max_normalized_measurement_error: float = 1.0,
        max_nfev: int = 300,
    ) -> dict[str, float]:
        """Fit regional scales and record the final measurement residuals."""

        # Validate targets and measurement tolerances
        measurement_names = (
            *JOINT_MEASUREMENTS,
            *JOINT_HEIGHT_MEASUREMENTS,
        )
        missing = set(measurement_names) - set(self.measurements)
        if missing:
            raise KeyError(
                f"Subject {self.subject_id!r} is missing measurements: {sorted(missing)}"
            )

        targets = {name: float(self.measurements[name]) for name in measurement_names}
        if any(not np.isfinite(value) or value <= 0.0 for value in targets.values()):
            raise ValueError("Target measurements must be finite and positive")

        if measurement_tolerances is None:
            tolerances = {name: 0.01 for name in measurement_names}
        else:
            if set(measurement_tolerances) != set(measurement_names):
                raise ValueError(
                    "Measurement tolerances must cover exactly "
                    f"the {len(measurement_names)} optimization measurements"
                )
            tolerances = {
                name: float(measurement_tolerances[name]) for name in measurement_names
            }

        if any(not np.isfinite(value) or value <= 0.0 for value in tolerances.values()):
            raise ValueError("Measurement tolerances must be finite and positive")

        if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")

        # Validate optimization settings and region coverage
        lower, upper = scale_bounds
        if not np.isfinite([lower, upper]).all() or not 0.0 < lower < 1.0 < upper:
            raise ValueError(
                "Scale bounds must be finite and satisfy 0 < lower < 1 < upper"
            )
        if not np.isfinite(prior_weight) or prior_weight < 0.0:
            raise ValueError("prior_weight must be finite and nonnegative")

        positive_options = {
            "sigma_symmetry": sigma_symmetry,
            "sigma_adjacency": sigma_adjacency,
            "max_normalized_measurement_error": max_normalized_measurement_error,
        }
        for name, value in positive_options.items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        model_names = {
            self.baseline_model.body(i).name
            for i in range(1, self.baseline_model.nbody)
        }
        if set(BODY_REGION_MAP) != model_names:
            raise ValueError("Body-region mapping does not match the baseline model")
        if len(set(OPTIMIZATION_REGIONS)) != len(OPTIMIZATION_REGIONS):
            raise ValueError("Optimization regions contain duplicates")
        if set(BODY_REGION_MAP.values()) != set(OPTIMIZATION_REGIONS) | {"global_root"}:
            raise ValueError(
                "Body-region mapping and optimization regions are inconsistent"
            )
        if BODY_REGION_MAP["Full Body"] != "global_root":
            raise ValueError("'Full Body' must belong to the fixed global_root region")

        region_indices = {
            name: index for index, name in enumerate(OPTIMIZATION_REGIONS)
        }

        # Similar bilateral scaling, without enforcing exact symmetry.
        symmetry_pairs = [
            (f"{region}_r", f"{region}_l")
            for region in (
                "upper_arm",
                "forearm",
                "thigh",
                "shank",
                "foot",
            )
        ]

        # Smooth scale changes between neighboring regions.
        adjacency_pairs = [
            ("pelvis_sacrum", "trunk"),
            ("trunk", "shoulder_girdle"),
        ]
        for side in ("r", "l"):
            adjacency_pairs.extend(
                [
                    ("shoulder_girdle", f"upper_arm_{side}"),
                    (f"upper_arm_{side}", f"forearm_{side}"),
                    ("pelvis_sacrum", f"thigh_{side}"),
                    (f"thigh_{side}", f"shank_{side}"),
                    (f"shank_{side}", f"foot_{side}"),
                ]
            )

        # Initialize and regularize scales using shoulder–ankle heights.
        baseline_landmarks, baseline_names = compute_myofullbody_landmarks(
            self.baseline_model
        )
        baseline_measurements = compute_myofullbody_measurements(
            baseline_landmarks,
            baseline_names,
        )

        height_ratios = np.asarray(
            [
                targets[name] / baseline_measurements[name]
                for name in JOINT_HEIGHT_MEASUREMENTS
            ],
            dtype=np.float64,
        )
        if not np.isfinite(height_ratios).all() or np.any(height_ratios <= 0.0):
            raise ValueError(f"Invalid shoulder–ankle height ratios: {height_ratios}")

        # Average in log space, consistent with the optimization variables.
        prior_center = float(np.mean(np.log(height_ratios)))
        prior_scale = float(np.exp(prior_center))

        body_scales: dict[str, float] = {}
        predicted: dict[str, float] = {}

        # Evaluate every candidate from the original baseline
        def residual(z: np.ndarray) -> np.ndarray:
            nonlocal body_scales, predicted

            region_scales = dict(zip(OPTIMIZATION_REGIONS, np.exp(z)))
            region_scales["global_root"] = 1.0
            body_scales = {
                body_name: float(region_scales[region])
                for body_name, region in BODY_REGION_MAP.items()
            }

            model = self.scale_geometry(body_scales).compile()
            landmarks, landmark_names = compute_myofullbody_landmarks(model)
            predicted = compute_myofullbody_measurements(landmarks, landmark_names)

            measurement_errors = [
                (predicted[name] - targets[name]) / tolerances[name]
                for name in measurement_names
            ]

            symmetry_errors = np.asarray(
                [
                    (z[region_indices[first]] - z[region_indices[second]])
                    / sigma_symmetry
                    for first, second in symmetry_pairs
                ]
            )

            adjacency_errors = np.asarray(
                [
                    (z[region_indices[first]] - z[region_indices[second]])
                    / sigma_adjacency
                    for first, second in adjacency_pairs
                ]
            )

            return np.concatenate(
                (
                    measurement_errors,
                    np.sqrt(prior_weight) * (z - prior_center),
                    symmetry_errors,
                    adjacency_errors,
                )
            )

        # Solve optimization problem
        result = least_squares(
            residual,
            x0=np.full(
                len(OPTIMIZATION_REGIONS),
                np.clip(prior_center, np.log(lower), np.log(upper)),
            ),
            bounds=(np.log(lower), np.log(upper)),
            jac="3-point",
            diff_step=1e-3,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-5,
            max_nfev=max_nfev,
            verbose=2,
        )

        # Re-evaluate the solution, since the last call may be a Jacobian probe
        residual(result.x)
        normalized_errors = {
            name: float(abs(predicted[name] - targets[name]) / tolerances[name])
            for name in measurement_names
        }
        failed_measurements = [
            name
            for name, error in normalized_errors.items()
            if (not np.isfinite(error) or error > max_normalized_measurement_error)
        ]

        quality_passed = not failed_measurements
        accepted = bool(result.success) and quality_passed

        message = str(result.message)
        if not quality_passed:
            message += " Measurement quality check failed: " + ", ".join(
                failed_measurements
            )

        # Optimization reports
        self.report = {
            "success": accepted,
            "solver_success": bool(result.success),
            "message": message,
            "regularization": {
                "prior_weight": float(prior_weight),
                "sigma_symmetry": float(sigma_symmetry),
                "sigma_adjacency": float(sigma_adjacency),
            },
            "quality": {
                "passed": quality_passed,
                "max_normalized_measurement_error": float(
                    max_normalized_measurement_error
                ),
                "failed_measurements": failed_measurements,
            },
            "prior_scale": prior_scale,
            "prior_scale_basis": "shoulder_ankle_heights",
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "region_scales": {
                name: float(np.exp(result.x[index]))
                for index, name in enumerate(OPTIMIZATION_REGIONS)
            },
            "bound_regions": [
                name
                for index, name in enumerate(OPTIMIZATION_REGIONS)
                if result.active_mask[index] != 0
            ],
            "measurements": {
                name: {
                    "target": targets[name],
                    "predicted": predicted[name],
                    "error_m": predicted[name] - targets[name],
                    "tolerance_m": tolerances[name],
                    "normalized_abs_error": normalized_errors[name],
                    "acceptance_limit_m": (
                        tolerances[name] * max_normalized_measurement_error
                    ),
                }
                for name in measurement_names
            },
        }

        if not accepted:
            raise OptimizationFailure(self.failure_report())

        return body_scales

    def failure_report(self) -> dict:
        """Return diagnostics for the latest unsuccessful optimization."""

        if not hasattr(self, "report"):
            raise RuntimeError("No optimization report is available")
        if self.report["success"]:
            raise RuntimeError("The latest optimization did not fail")

        return {
            "status": "optimization_failed",
            "strategy": "segment_scale",
            "subject_id": self.subject_id,
            "error": f"Regional scale optimization failed: {self.report['message']}",
            "optimization": self.report,
        }
