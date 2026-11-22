"""Fit one global MyoFullBody scale to 17 subject measurements."""

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


class HeightScaleModel(EmbodimentModel):
    """Fit a shared body scale with the regional fitting residual convention."""

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
        max_nfev: int = 100,
    ) -> dict[str, float]:
        """Optimize one log scale. Accept every successful solver result."""
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

        # Validate optimization settings and body coverage.
        lower, upper = scale_bounds
        if not np.isfinite([lower, upper]).all() or not 0 < lower < 1 < upper:
            raise ValueError(
                "Scale bounds must be finite and satisfy 0 < lower < 1 < upper"
            )
        if not np.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")
        body_names = [
            self.baseline_model.body(i).name
            for i in range(1, self.baseline_model.nbody)
        ]
        if any(not name for name in body_names):
            raise ValueError("All MyoFullBody bodies must have names")
        if "Full Body" not in body_names:
            raise KeyError("MyoFullBody is missing the 'Full Body' root body")

        # Initialize and regularize the scale using shoulder–ankle heights.
        landmarks, names = compute_myofullbody_landmarks(self.baseline_model)
        baseline_measurements = compute_myofullbody_measurements(landmarks, names)
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

        # Evaluate every candidate from the original baseline.
        def residual(z: np.ndarray) -> np.ndarray:
            nonlocal body_scales, predicted
            scale = float(np.exp(z[0]))
            body_scales = {
                name: 1.0 if name == "Full Body" else scale for name in body_names
            }
            model = self.scale_geometry(body_scales).compile()
            landmarks, names = compute_myofullbody_landmarks(model)
            predicted = compute_myofullbody_measurements(landmarks, names)
            measurement_errors = [
                (predicted[name] - targets[name]) / tolerances[name]
                for name in measurement_names
            ]
            return np.concatenate(
                (
                    measurement_errors,
                    np.sqrt(prior_weight) * (z - prior_center),
                )
            )

        # Solve the single-scale optimization problem.
        result = least_squares(
            residual,
            x0=np.asarray([np.clip(prior_center, np.log(lower), np.log(upper))]),
            bounds=(np.log(lower), np.log(upper)),
            jac="3-point",
            diff_step=1e-3,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-5,
            max_nfev=max_nfev,
            verbose=2,
        )

        # Re-evaluate the solution after the final Jacobian probe.
        residual(result.x)

        # Record diagnostics without a measurement quality rejection threshold.
        self.report = {
            "method": "global_measurement_fit",
            "success": bool(result.success),
            "solver_success": bool(result.success),
            "message": str(result.message),
            "regularization": {"prior_weight": float(prior_weight)},
            "prior_scale": prior_scale,
            "prior_scale_basis": "shoulder_ankle_heights",
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "uniform_scale": float(np.exp(result.x[0])),
            "bound_active": bool(result.active_mask[0]),
            "measurements": {
                name: {
                    "target": targets[name],
                    "predicted": predicted[name],
                    "error_m": predicted[name] - targets[name],
                    "tolerance_m": tolerances[name],
                    "normalized_abs_error": (
                        abs(predicted[name] - targets[name]) / tolerances[name]
                    ),
                }
                for name in measurement_names
            },
        }

        if not result.success:
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
            "strategy": "height_scale",
            "subject_id": self.subject_id,
            "error": f"Global scale optimization failed: {self.report['message']}",
            "optimization": self.report,
        }
