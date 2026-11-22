import json
import tempfile
from pathlib import Path
from unittest import TestCase
import numpy as np
from myoacross.retarget.export import (
    evaluate_retarget_quality,
    annotate_exported_result,
)


class QualityTests(TestCase):
    def test_large_ground_correction_fails_despite_finite_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "motion.npz"
            np.savez(
                path,
                fps=np.array([50]),
                joint_pos=np.zeros((4, 2)),
                joint_vel=np.zeros((4, 2)),
                body_pos_w=np.zeros((4, 1, 3)),
                body_quat_w=np.tile([1, 0, 0, 0], (4, 1, 1)),
                body_lin_vel_w=np.zeros((4, 1, 3)),
                body_ang_vel_w=np.zeros((4, 1, 3)),
            )
            report = {
                "metadata": {
                    "shape_fitting": {"metadata": {"metrics": {"converged": True}}},
                    "ground_correction": {"max_upward_shift": 0.200001},
                },
                "motion": {"body_names": ["pelvis"]},
                "ik_position_errors": {
                    "per_body": {"calcn_l": {"p95_m": 0.01}, "calcn_r": {"p95_m": 0.01}}
                },
            }
            quality = evaluate_retarget_quality(report, path)
            self.assertEqual(quality["status"], "failed")
            self.assertIn("max_upward_shift_m", quality["reasons"])
            for shift in (0.01, 0.15, 0.20):
                report["metadata"]["ground_correction"]["max_upward_shift"] = shift
                self.assertEqual(
                    evaluate_retarget_quality(report, path)["status"], "passed"
                )

            report_path = path.with_name("report.json")
            report_path.write_text(json.dumps(report))
            annotate_exported_result(temp)
            exported = json.loads(report_path.read_text())
            self.assertEqual(exported["quality"]["status"], "passed")
