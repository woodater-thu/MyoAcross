"""Regression coverage for fitted cache reuse and per-motion batch reporting."""

import json
import queue
import tempfile
import threading
import unittest
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from myoacross.retarget import retargeter as rt, run
from myoacross.retarget.config import load_config


class FittedRetargeterCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.xml = self.root / "model.xml"
        self.xml.write_text("model fixture")
        self.fit = self.root / "shape.pkl"
        self.fit.write_bytes(b"fit one")
        self.metadata = self.root / "shape_metadata.json"
        self.metadata.write_text("metadata one")
        self.config = {
            "gmr": {
                "solver": "daqp",
                "damping": 0.5,
                "verbose": False,
                "use_velocity_limit": False,
                "use_fitted_shape": True,
            },
            "performance": {"fast_ik": True},
        }
        self.offsets = {"pos_offsets": {"pelvis": [1.0, 2.0, 3.0]}}
        self.created = []

        def create(*args, **kwargs):
            engine = SimpleNamespace(
                model=object(),
                configuration=object(),
                learned_offsets=deepcopy(self.offsets),
                ground_offset=0.0,
            )
            self.created.append(engine)
            return engine, []

        for target, name, value in [
            (rt, "_RETARGETER_CACHE", OrderedDict()),
            (rt, "validate_fitted_shape", lambda p: {}),
            (rt, "create_retargeter", create),
            (rt, "install_mink_patch", lambda enabled: None),
            (rt, "initialize_retargeter", lambda engine: []),
            (rt.mink, "Configuration", lambda model: object()),
        ]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_reset_preserves_fit_but_not_mutated_motion_state(self):
        engine, _ = rt.get_retargeter(self.xml, self.config, fitted_shape_path=self.fit)
        old_configuration = engine.configuration
        engine.learned_offsets["pos_offsets"]["pelvis"][0] = 999
        engine.ground_offset = 42.0
        reused, _ = rt.get_retargeter(self.xml, self.config, fitted_shape_path=self.fit)
        self.assertIs(engine, reused)
        self.assertIsNot(reused.configuration, old_configuration)
        self.assertEqual(reused.learned_offsets, self.offsets)
        self.assertEqual(reused.ground_offset, 0.0)
        self.assertEqual(len(self.created), 1)

    def test_metadata_content_and_fit_path_invalidate_cache(self):
        first, _ = rt.get_retargeter(self.xml, self.config, fitted_shape_path=self.fit)
        self.metadata.write_text("changed offset metadata")
        second, _ = rt.get_retargeter(self.xml, self.config, fitted_shape_path=self.fit)
        self.assertIsNot(first, second)
        other = self.root / "other.pkl"
        other.write_bytes(self.fit.read_bytes())
        other.with_name("other_metadata.json").write_text(self.metadata.read_text())
        third, _ = rt.get_retargeter(self.xml, self.config, fitted_shape_path=other)
        self.assertIsNot(second, third)

    def test_fit_flag_mismatch_rejected_before_cache_lookup(self):
        rt.get_retargeter(self.xml, self.config, fitted_shape_path=self.fit)
        with self.assertRaises(ValueError):
            rt.get_retargeter(self.xml, self.config)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text("name: gmr_fit\n")
        self.config = load_config(self.config_path)
        self.jobs = [
            run.RetargetJob(
                "target",
                "source",
                str(i),
                "walk",
                "set",
                "segment_scale",
                "train",
                self.root / "source.npz",
                self.root / "model.xml",
                self.root / str(i),
            )
            for i in range(3)
        ]

    def execute(self, job, *args):
        if job.motion_id == "1":
            raise ValueError("bad input")
        return {
            **job.identity(),
            "output_dir": str(job.output_dir),
            "status": "exported",
        }

    def test_sequential_failure_preserves_count_and_continue_semantics(self):
        with patch.object(run, "_execute_job", side_effect=self.execute):
            observed = []
            records = run.execute_jobs(
                self.jobs, self.config_path, on_result=observed.append
            )
            self.assertEqual([r["status"] for r in records], ["exported", "failed"])
            self.assertEqual(records, observed)
            records = run.execute_jobs(
                self.jobs, self.config_path, continue_on_error=True
            )
            self.assertEqual(
                [r["status"] for r in records], ["exported", "failed", "exported"]
            )

    def test_worker_reports_success_before_failure_and_stops(self):
        for keep_going in [False, True]:
            events = queue.Queue()
            stop = threading.Event()
            with (
                patch.object(run, "_RESULT_QUEUE", events),
                patch.object(run, "_STOP_EVENT", stop),
                patch.object(run, "_execute_job", side_effect=self.execute),
            ):
                records = run._run_group(
                    self.jobs, self.config_path, self.config, False, False, keep_going
                )
            self.assertEqual(len(records), 3 if keep_going else 2)
            self.assertEqual([events.get_nowait() for _ in records], records)
            self.assertEqual(stop.is_set(), not keep_going)

    def test_skip_requires_motion_identity_config_and_quality(self):
        job = self.jobs[0]
        job.output_dir.mkdir()
        report = {
            "metadata": {"identity": job.identity(), "config": self.config},
            "quality": {"status": "passed"},
        }
        (job.output_dir / "report.json").write_text(json.dumps(report))
        (job.output_dir / "motion.npz").touch()
        with patch.object(run, "run_motion") as execute:
            result = run._execute_job(job, self.config_path, self.config, False, True)
            self.assertEqual(result["status"], "skipped_complete")
            execute.assert_not_called()
        report["metadata"]["config"] = {"changed": True}
        (job.output_dir / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "Stale/incomplete"):
            run._execute_job(job, self.config_path, self.config, False, True)
        report["metadata"]["config"] = self.config
        (job.output_dir / "report.json").write_text(json.dumps(report))
        (job.output_dir / "motion.npz").unlink()
        with self.assertRaisesRegex(ValueError, "Stale/incomplete"):
            run._execute_job(job, self.config_path, self.config, False, True)

    def test_main_manifest_keeps_success_before_failed_motion(self):
        manifest = self.root / "manifest.json"
        with (
            patch.object(run, "collect_jobs", return_value=self.jobs),
            patch.object(run, "_execute_job", side_effect=self.execute),
        ):
            with self.assertRaises(SystemExit) as error:
                run.main(
                    [
                        "--config",
                        str(self.config_path),
                        "--embodiment",
                        "segment_scale",
                        "--manifest",
                        str(manifest),
                    ]
                )
        self.assertEqual(error.exception.code, 1)
        records = json.loads(manifest.read_text())
        self.assertEqual([r["status"] for r in records], ["exported", "failed"])
        self.assertEqual(sum(r["status"] == "exported" for r in records), 1)

    def test_dry_run_never_starts_execution_or_fitting(self):
        with (
            patch.object(run, "collect_jobs", return_value=self.jobs),
            patch.object(run, "execute_jobs") as execute,
            patch.object(rt, "ensure_fitted_shape") as fit,
        ):
            run.main(
                [
                    "--config",
                    str(self.config_path),
                    "--embodiment",
                    "segment_scale",
                    "--workers",
                    "2",
                    "--dry-run",
                ]
            )
            execute.assert_not_called()
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
