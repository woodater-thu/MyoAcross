"""Disable inherited virtual settings explicitly for subject-only generation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from myoacross.rl.runners import generate_embodiment_bundles as generation
from myoacross.utils.config import load_config
from mjlab.sim.sim import SimulationCfg


class GenerationConfigurationTests(TestCase):
    def test_generation_passes_worker_count(self):
        with TemporaryDirectory() as temp:
            with patch.object(generation, "generate_embodiment_bundles", return_value=[]) as export:
                generation.run_generation({
                    "data": {"embodiment": "baseline", "bundle_root": temp},
                    "workers": 4,
                })
            self.assertEqual(export.call_args.kwargs["workers"], 4)

    def test_generation_rejects_invalid_worker_count_before_export(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers):
                with patch.object(generation, "generate_embodiment_bundles") as export:
                    with self.assertRaisesRegex(ValueError, "workers must be a positive integer"):
                        generation.run_generation({
                            "data": {"embodiment": "baseline", "bundle_root": "/tmp/unused-bundle"},
                            "workers": workers,
                        })
                export.assert_not_called()

    def test_generation_requires_explicit_bundle_root(self):
        with patch.object(generation, "generate_embodiment_bundles") as export:
            with self.assertRaisesRegex(ValueError, "bundle_root"):
                generation.run_generation({"data": {"embodiment": "baseline"}})
        export.assert_not_called()

    def test_subject_export_accepts_explicit_null_virtual_settings(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "generate.yaml"
            config_path.write_text(
                "# @package _global_\n"
                "defaults:\n  - /rl/generate@_global_\n  - _self_\n"
                "data:\n  embodiment: baseline\n"
                f"  bundle_root: {root / 'bundles'}\n"
                "sampling: null\ncount_per_subject: null\nseed: null\n"
            )
            expected = [root / "bundles/subject/index.json"]
            with patch.object(generation, "generate_embodiment_bundles", return_value=expected) as export:
                self.assertEqual(generation.run_generation(load_config(config_path)), expected)
            export.assert_called_once()
            options = export.call_args.kwargs
            self.assertEqual(options["morphology_source"], "subject")
            self.assertEqual(options["count_per_subject"], 1)
            self.assertEqual(options["seed"], 42)
            self.assertIsNone(options["sampling"])
            self.assertEqual(options["output_dir"], root / "bundles/subject")
            self.assertEqual(options["workers"], 16)

    def test_subject_export_rejects_non_null_virtual_only_settings(self):
        for key, value in (("seed", 42), ("count_per_subject", 10)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "require sampling"):
                generation.run_generation(
                    {
                        "data": {"embodiment": "baseline", "bundle_root": "/tmp/unused-bundle"},
                        key: value,
                    }
                )

    def test_failed_subject_does_not_mark_bank_complete(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline.xml"
            baseline.touch()
            sources = {}
            for subject_id in ("KIT_10", "KIT_11"):
                directory = root / subject_id
                directory.mkdir()
                (directory / "model.xml").touch()
                (directory / "report.json").write_text(json.dumps({
                    "status": "passed", "strategy": "segment_scale",
                    "subject_id": subject_id, "baseline_path": str(baseline),
                }))
                sources[subject_id] = directory
            for results, error in (
                ([(root / "first.json", "a"), (root / "second.json", "b")], ValueError),
                ([(root / "first.json", "a"), RuntimeError("generation failed")], RuntimeError),
            ):
                with self.subTest(error=error):
                    output = root / error.__name__
                    with patch.object(generation, "_generate_subject_bundles", side_effect=results):
                        with self.assertRaises(error):
                            generation.generate_embodiment_bundles(
                                sources, output, embodiment="segment_scale",
                                morphology_source="subject", simulation_config=SimulationCfg(),
                                workers=1,
                            )
                    self.assertFalse((output / "bank.json").exists())

            # Empty XML passes source inventory checks but fails inside a worker.
            output = root / "worker_failure"
            with self.assertRaises(ValueError):
                generation.generate_embodiment_bundles(
                    sources, output, embodiment="segment_scale",
                    morphology_source="subject", simulation_config=SimulationCfg(),
                    workers=2,
                )
            self.assertFalse((output / "bank.json").exists())
