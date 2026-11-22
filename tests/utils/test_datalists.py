"""Product catalogs keep target morphology separate from source motions."""

from collections import Counter
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from myoacross.utils import datalists as data


class TargetSubjectTests(TestCase):
    def test_split_is_disjoint_and_covers_both_datasets(self):
        train = data.get_target_subject_ids(split="train")
        test = data.get_target_subject_ids(split="test")
        self.assertEqual(len(train), 110)
        self.assertEqual(len(test), 31)
        self.assertEqual(len(set(train + test)), 141)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(
            Counter(s.split("_", 1)[0] for s in train + test),
            {"KIT": 55, "BMLmovi": 86},
        )
        for strategy in ("baseline", "height_scale", "segment_scale"):
            self.assertEqual(
                data.get_target_subject_ids(split="train", embodiment=strategy), train
            )

    def test_exclusion_does_not_reassign_other_subjects(self):
        train = data.get_target_subject_ids(split="train")
        test = data.get_target_subject_ids(split="test")
        with patch.dict(data.TARGET_SUBJECT_EXCLUSIONS, {train[0]: "bad measurements"}):
            self.assertEqual(data.get_target_subject_ids(split="train"), train[1:])
            self.assertEqual(data.get_target_subject_ids(split="test"), test)

    def test_invalid_selection_or_exclusion_fails(self):
        for kwargs in ({"split": "typo"}, {"split": "train", "embodiment": "typo"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                data.get_target_subject_ids(**kwargs)
        for exclusions in ({"KIT_typo": "bad"}, {"KIT_3": ""}):
            with patch.dict(data.TARGET_SUBJECT_EXCLUSIONS, exclusions, clear=True):
                with self.assertRaises(ValueError):
                    data.get_target_subject_ids(split="train")


class MotionCatalogTests(TestCase):
    def test_standard_selection_excludes_running_without_changing_catalog(self):
        self.assertEqual(len(data.KIT_FLAT_LOCOMOTION_72), 72)
        expected = tuple(m for m in data.KIT_FLAT_LOCOMOTION_72 if m.category != "run")
        selected = data.get_motion_specs(motion_set="kit_flat_locomotion_72")
        self.assertEqual(len(selected), 64)
        self.assertEqual(selected, expected)

    def test_motion_exclusion_preserves_other_entries_and_order(self):
        selected = data.get_motion_specs(motion_set="kit_flat_locomotion_72")
        with patch.dict(data.MOTION_EXCLUSIONS, {selected[0].motion_id: "test exclusion"}):
            self.assertEqual(
                data.get_motion_specs(motion_set="kit_flat_locomotion_72"), selected[1:]
            )

    def test_invalid_motion_exclusions_fail(self):
        first = data.KIT_FLAT_LOCOMOTION_72[0].motion_id
        for exclusions in ({"unknown_motion": "invalid"}, {first: ""}):
            with patch.dict(data.MOTION_EXCLUSIONS, exclusions, clear=True):
                with self.assertRaises(ValueError):
                    data.get_motion_specs(motion_set="kit_flat_locomotion_72")

    def test_source_subject_is_derived_independently_of_target_split(self):
        spec = data.MotionSpec("kit_3_walk", "KIT/3/walk_poses.npz", "walk")
        self.assertEqual(spec.source_path, Path("KIT/3/walk_poses.npz"))
        self.assertEqual(spec.source_subject_id, "KIT_3")
        with patch.dict(data.MOTION_SETS, {"example": (spec,)}):
            self.assertEqual(data.get_motion_specs(motion_set="example"), (spec,))

    def test_motion_identity_and_path_must_be_unambiguous(self):
        for motion_id, path, category in (
            ("../walk", "KIT/3/walk.npz", "walk"),
            ("walk", "/KIT/3/walk.npz", "walk"),
            ("walk", "KIT/../walk.npz", "walk"),
            ("walk", "UnknownDataset/S001/walk.npz", "walk"),
            ("walk", "KIT/3/walk.txt", "walk"),
            ("walk", "KIT/3/walk.npz", " "),
        ):
            with (
                self.subTest(path=path, motion_id=motion_id),
                self.assertRaises(ValueError),
            ):
                data.MotionSpec(motion_id, path, category)

    def test_reject_duplicate_ids_sources_and_empty_sets(self):
        a = data.MotionSpec("walk_a", "KIT/3/walk.npz", "walk")
        same_id = data.MotionSpec("walk_a", "KIT/4/walk.npz", "walk")
        same_source = data.MotionSpec("walk_b", "KIT/3/walk.npz", "walk")
        for specs in ((), (a, same_id), (a, same_source)):
            with patch.dict(data.MOTION_SETS, {"example": specs}):
                with self.assertRaises(ValueError):
                    data.get_motion_specs(motion_set="example")
        with self.assertRaises(ValueError):
            data.get_motion_specs(motion_set="not_registered")


class SubjectReferenceTests(TestCase):
    def test_prefers_shape_then_deterministic_own_motion(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "KIT" / "3"
            subject.mkdir(parents=True)
            (subject / "z_poses.npz").touch()
            (subject / "a_poses.npz").touch()
            (subject / "unrelated.npz").touch()
            self.assertEqual(
                data.get_subject_reference_motion("KIT_3", dataset_root=root),
                subject / "a_poses.npz",
            )
            (subject / "shape.npz").touch()
            self.assertEqual(
                data.get_subject_reference_motion("KIT_3", dataset_root=root),
                subject / "shape.npz",
            )

    def test_missing_reference_and_unknown_subject_fail(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                data.get_subject_reference_motion("KIT_3", dataset_root=root)
            for subject in ("KIT_../3", "UnknownDataset_S001", "KIT_typo"):
                with self.subTest(subject=subject), self.assertRaises(ValueError):
                    data.get_subject_reference_motion(subject, dataset_root=root)
