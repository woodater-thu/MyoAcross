import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from myoacross.retarget import run
from myoacross.retarget.config import load_config
from myoacross.utils.datalists import MotionSpec


class ProductJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.motions = (
            MotionSpec("KIT__1__walk", "KIT/1/walk.npz", "walk_slow"),
            MotionSpec("KIT__2__walk", "KIT/2/walk.npz", "walk_slow"),
        )
        for motion in self.motions:
            path = self.root / "dataset" / motion.source_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for subject in ("KIT_1", "BMLmovi_2"):
            self.model("segment_scale", subject)
        self.model("baseline", None)
        for name, value in (
            ("EMBODIMENTS_DIR", self.root / "embodiments"),
            ("DATASET_ROOT", self.root / "dataset"),
        ):
            p = patch.object(run, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(
            run,
            "get_target_subject_ids",
            return_value=("KIT_1", "BMLmovi_2"),
            create=True,
        )
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(
            run, "get_motion_specs", return_value=self.motions, create=True
        )
        p.start()
        self.addCleanup(p.stop)

    def model(self, strategy, subject):
        folder = self.root / "embodiments" / strategy
        if subject:
            folder /= subject
        folder.mkdir(parents=True)
        (folder / "model.xml").touch()
        (folder / "report.json").write_text(
            json.dumps(dict(strategy=strategy, subject_id=subject, status="passed"))
        )

    def jobs(self, strategy="segment_scale", **kwargs):
        return run.collect_jobs(
            strategy, motion_set="example", output_root=self.root / "out", **kwargs
        )

    def test_cartesian_identity_and_colliding_stems(self):
        jobs = self.jobs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({j.output_dir for j in jobs}), 4)
        for job in jobs:
            self.assertEqual(
                job.output_dir,
                self.root / "out" / "segment_scale" / job.target_subject_id / job.motion_id,
            )
        self.assertEqual(
            {(j.target_subject_id, j.source_subject_id) for j in jobs},
            {
                ("KIT_1", "KIT_1"),
                ("KIT_1", "KIT_2"),
                ("BMLmovi_2", "KIT_1"),
                ("BMLmovi_2", "KIT_2"),
            },
        )

    def test_target_and_motion_filters(self):
        jobs = self.jobs(subject_id="BMLmovi_2", motion_ids=["KIT__2__walk"])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].target_subject_id, "BMLmovi_2")
        with self.assertRaises(ValueError):
            self.jobs(motion_ids=["unknown"])

    def test_baseline_computed_once_per_motion(self):
        jobs = self.jobs("baseline")
        self.assertEqual(len(jobs), 2)
        self.assertEqual({j.target_subject_id for j in jobs}, {None})
        for job in jobs:
            self.assertEqual(
                job.output_dir, self.root / "out" / "baseline" / job.motion_id
            )

    def test_missing_source_fails_before_execution(self):
        (self.root / "dataset" / self.motions[1].source_path).unlink()
        with self.assertRaises(FileNotFoundError):
            self.jobs()


class FitConfigurationTests(unittest.TestCase):
    def config(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text)
            return load_config(path)

    def test_fitted_config_and_conflicts(self):
        text = "name: gmr_fit\nsmplh:\n  gender_mode: neutral\ngmr:\n  use_fitted_shape: true\n  offsets_mode: fitted\n"
        self.assertTrue(self.config(text)["gmr"]["use_fitted_shape"])
        with self.assertRaises(ValueError):
            self.config(text.replace("neutral", "motion"))
        with self.assertRaises(ValueError):
            self.config(text.replace("fitted\n", "subject_specific\n"))
        with self.assertRaises(ValueError):
            self.config(text + "shape_fitting:\n  iterations: 0\n")


class FittedPlumbingTests(unittest.TestCase):
    def test_same_fit_reaches_loader_and_gmr_without_source_offset_override(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import numpy as np
        from myoacross.retarget import retargeter as rt

        config = {
            "gmr": {
                "use_fitted_shape": True,
                "offsets_mode": "fitted",
                "offset_to_ground": False,
            },
            "shape_fitting": {"iterations": 500, "lr": 0.001},
            "postprocess": {"ground_correction": False},
            "export": {"fps": 50},
        }
        fit_path = Path("/tmp/target-fit/shape.pkl")
        offsets = {"pos_offsets": {"pelvis": [0.0, 0.0, 0.0]}}
        model = SimpleNamespace(
            njnt=0, nbody=0, jnt_qposadr=np.array([]), jnt_dofadr=np.array([])
        )
        engine = SimpleNamespace(model=model, learned_offsets=offsets)
        patches = {
            "ensure_fitted_shape": Mock(return_value=fit_path),
            "validate_fitted_shape": Mock(
                return_value={"metrics": {"converged": True}}
            ),
            "get_retargeter": Mock(return_value=(engine, [])),
            "load_source_motion": Mock(
                return_value=([{}, {}, {}], 50.0, {"sampling_fps": 50.0})
            ),
            "compute_subject_offsets": Mock(
                side_effect=AssertionError("must not use source offsets")
            ),
            "retarget_frames": Mock(
                return_value=(np.zeros((3, 7)), np.zeros((3, 1)), ["pelvis"])
            ),
            "resample_qpos": Mock(return_value=np.zeros((3, 7))),
            "compute_qvel": Mock(return_value=(np.zeros((3, 7)), np.zeros((3, 6)))),
        }
        from contextlib import ExitStack

        with ExitStack() as stack:
            for name, mock in patches.items():
                stack.enter_context(patch.object(rt, name, mock))
            result = rt.retarget_motion("/tmp/source.npz", "/tmp/target.xml", config)
        self.assertEqual(
            patches["get_retargeter"].call_args.kwargs["fitted_shape_path"], fit_path
        )
        self.assertEqual(
            patches["load_source_motion"].call_args.kwargs["fitted_shape_path"],
            fit_path,
        )
        patches["compute_subject_offsets"].assert_not_called()
        self.assertEqual(result["metadata"]["alignment"]["offsets"], offsets)


if __name__ == "__main__":
    unittest.main()
