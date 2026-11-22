"""Check single-scale fitting, solver failure, and uniform optimization reports."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import mujoco
import numpy as np

from myoacross.embodiment.height_scale.model import HeightScaleModel
from myoacross.embodiment.config import load_config
from myoacross.embodiment.export import build_report, save_failure_report
from myoacross.embodiment.segment_scale.model import OptimizationFailure
from myoacross.embodiment.smplh_to_measurements import (
    JOINT_MEASUREMENTS,
    JOINT_HEIGHT_MEASUREMENTS,
)

NAMES = (*JOINT_MEASUREMENTS, *JOINT_HEIGHT_MEASUREMENTS)
MODULE = "myoacross.embodiment.height_scale.model"


class HeightScaleTests(TestCase):
    def make_model(self):
        model = HeightScaleModel.__new__(HeightScaleModel)
        model.subject_id = "KIT_1487"
        model.measurements = {name: 1.2 for name in NAMES}
        # Height is deliberately inconsistent with the 17 fitting measurements.
        model.measurements["height"] = 2.0
        model.baseline_model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="Full Body"><body name="segment">'
            '<geom type="sphere" size="0.1"/></body></body></worldbody></mujoco>'
        )
        return model

    def fit(self, model, **options):
        def geometry(scales):
            self.assertEqual(scales["Full Body"], 1.0)
            return SimpleNamespace(compile=lambda: scales["segment"])

        with (
            patch.object(model, "scale_geometry", side_effect=geometry),
            patch(
                MODULE + ".compute_myofullbody_landmarks",
                create=True,
                side_effect=lambda value: (
                    1.0 if value is model.baseline_model else value,
                    None,
                ),
            ),
            patch(
                MODULE + ".compute_myofullbody_measurements",
                create=True,
                side_effect=lambda value, names: {name: value for name in NAMES},
            ),
        ):
            return model.solve(**options)

    def test_fit_uses_all_measurements_and_preserves_root(self):
        model = self.make_model()
        scales = self.fit(model, prior_weight=0.0)
        self.assertAlmostEqual(scales["segment"], 1.2, places=6)
        self.assertEqual(len(model.report["measurements"]), 17)
        self.assertTrue(model.report["solver_success"])
        self.assertNotIn("quality", model.report)

    def test_weighted_optimum_and_large_errors_do_not_reject(self):
        model = self.make_model()
        model.measurements[NAMES[0]] = 2.0
        tolerances = {name: 0.01 for name in NAMES}
        tolerances[NAMES[0]] = 0.1
        scales = self.fit(model, prior_weight=0.0, measurement_tolerances=tolerances)
        expected = (16 * 1.2 / 0.01**2 + 2.0 / 0.1**2) / (16 / 0.01**2 + 1 / 0.1**2)
        self.assertAlmostEqual(scales["segment"], expected, places=6)
        self.assertGreater(
            model.report["measurements"][NAMES[0]]["normalized_abs_error"], 4
        )
        self.assertTrue(model.report["success"])

    def test_success_at_scale_bound_is_exportable(self):
        model = self.make_model()
        model.measurements.update({name: 2.0 for name in NAMES})
        scales = self.fit(model, prior_weight=0.0)
        self.assertAlmostEqual(scales["segment"], 1.3, places=6)
        self.assertTrue(model.report["success"])
        self.assertTrue(model.report["bound_active"])

    def test_invalid_measurements_and_options_are_rejected(self):
        for change, options, exception in [
            ({NAMES[0]: 0.0}, {}, ValueError),
            ({NAMES[0]: float("nan")}, {}, ValueError),
            ({}, {"measurement_tolerances": {}}, ValueError),
            ({}, {"scale_bounds": (0.0, 1.3)}, ValueError),
            ({}, {"max_nfev": 0}, ValueError),
        ]:
            with self.subTest(change=change, options=options):
                model = self.make_model()
                model.measurements.update(change)
                with self.assertRaises(exception):
                    self.fit(model, **options)
        model = self.make_model()
        del model.measurements[NAMES[0]]
        with self.assertRaises(KeyError):
            self.fit(model)

    def test_solver_failure_saves_report_without_model(self):
        model = self.make_model()
        model.measurements[NAMES[0]] = 2.0
        with self.assertRaises(OptimizationFailure) as caught:
            self.fit(model, prior_weight=0.0, max_nfev=1)
        report = caught.exception.report
        with TemporaryDirectory() as temp:
            path = save_failure_report(
                report,
                config={"name": "height_scale"},
                config_path=Path(temp) / "config.yaml",
                output_root=temp,
            )
            self.assertTrue(path.is_file())
            self.assertFalse((path.parent / "model.xml").exists())

    def test_uniform_report_marks_all_fitting_measurements(self):
        model = self.make_model()
        scales = self.fit(model, prior_weight=0.0)
        report = build_report(
            {
                "strategy": "height_scale",
                "subject_id": model.subject_id,
                "body_scales": scales,
                "optimization": model.report,
                "validation": {
                    "mass_inertia_checks_passed": True,
                    "tendon_muscle_checks_passed": True,
                    "body_mass_kg": 70.0,
                    "target_body_mass_kg": 70.0,
                    "body_mass_error_kg": 0.0,
                    "density_scale": 1.0,
                    "measurements": {
                        name: {
                            "predicted": scales["segment"],
                            "target": model.measurements[name],
                        }
                        for name in NAMES
                    },
                },
            }
        )
        self.assertAlmostEqual(report["scales"]["uniform"], 1.2, places=6)
        self.assertTrue(
            all(entry["used_for_scaling"] for entry in report["measurements"].values())
        )
        self.assertNotIn("measurement_quality_passed", report["checks"])

    def test_configuration_accepts_global_options_only(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "height.yaml"
            path.write_text(
                "name: height_scale\noptimization:\n  prior_weight: 0\n  max_nfev: 100\n"
            )
            self.assertEqual(load_config(path)["optimization"]["prior_weight"], 0)
            path.write_text(
                "name: height_scale\noptimization:\n  sigma_symmetry: 0.03\n"
            )
            with self.assertRaises(ValueError):
                load_config(path)
