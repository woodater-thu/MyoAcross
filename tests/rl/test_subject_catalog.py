import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from myoacross.rl.embodiments.subjects import DatasetRecord
from myoacross.utils.datalists import MotionSpec


class SubjectCatalogTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.specs = (
            MotionSpec("KIT__1__walk", "KIT/1/walk.npz", "walk_medium"),
            MotionSpec("KIT__2__walk", "KIT/2/walk.npz", "walk_medium"),
        )
        for spec in self.specs:
            p = self.root / "source" / spec.source_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        for subject in ("BMLmovi_A", "KIT_B"):
            folder = self.root / "models" / "segment_scale" / subject
            folder.mkdir(parents=True)
            (folder / "model.xml").write_text("<mujoco/>")
            (folder / "report.json").write_text(
                json.dumps(
                    dict(status="passed", strategy="segment_scale", subject_id=subject)
                )
            )
            for spec in self.specs:
                folder = (
                    self.root
                    / "retarget"
                    / "segment_scale"
                    / subject
                    / spec.motion_id
                )
                folder.mkdir(parents=True)
                (folder / "motion.npz").touch()
                report = {
                    "metadata": {
                        "config": {},
                        "robot_xml_path": str(
                            self.root
                            / "models"
                            / "segment_scale"
                            / subject
                            / "model.xml"
                        ),
                        "motion_path": str(self.root / "source" / spec.source_path),
                        "identity": dict(
                            target_subject_id=subject,
                            source_subject_id=spec.source_subject_id,
                            motion_id=spec.motion_id,
                            category=spec.category,
                            motion_set="testset",
                            embodiment="segment_scale",
                            target_split="train",
                        ),
                    },
                    "quality": {"status": "passed"},
                }
                (folder / "report.json").write_text(json.dumps(report))
        for name, value in [
            ("get_target_subject_ids", ("BMLmovi_A", "KIT_B")),
            ("get_motion_specs", self.specs),
        ]:
            p = patch(
                "myoacross.rl.embodiments.subjects." + name,
                return_value=value,
                create=True,
            )
            p.start()
            self.addCleanup(p.stop)

    def catalog(self, **kwargs):
        return DatasetRecord.from_split(
            split="train",
            embodiment="segment_scale",
            motion_set="testset",
            dataset_root=self.root / "source",
            embodiments_root=self.root / "models",
            retargets_root=self.root / "retarget",
            **kwargs,
        )

    def test_targets_share_same_motion_ids(self):
        catalog = self.catalog()
        self.assertEqual(catalog.num_motions, 4)
        self.assertEqual(
            [m.motion_id for m in catalog.get("BMLmovi_A").motions],
            [m.motion_id for m in catalog.get("KIT_B").motions],
        )
        self.assertNotEqual(
            catalog.get("BMLmovi_A").motions[0].retarget_path,
            catalog.get("KIT_B").motions[0].retarget_path,
        )

    def test_selection_before_file_validation(self):
        (self.root / "models" / "segment_scale" / "KIT_B" / "model.xml").unlink()
        self.assertEqual(len(self.catalog(subject_ids=["BMLmovi_A"])), 1)

    def test_wrong_target_report_rejected(self):
        p = (
            self.root
            / "retarget"
            / "segment_scale"
            / "BMLmovi_A"
            / self.specs[0].motion_id
            / "report.json"
        )
        report = json.loads(p.read_text())
        report["metadata"]["identity"]["target_subject_id"] = "KIT_B"
        p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "identity"):
            self.catalog()

    def test_failed_quality_is_not_silently_skipped(self):
        p = (
            self.root
            / "retarget"
            / "segment_scale"
            / "BMLmovi_A"
            / self.specs[0].motion_id
            / "report.json"
        )
        report = json.loads(p.read_text())
        report["quality"]["status"] = "failed"
        p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "quality"):
            self.catalog()
        self.assertEqual(self.catalog(require_quality_pass=False).num_motions, 4)
