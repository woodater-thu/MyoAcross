"""Generation entry points select product targets independently of RL clips."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from myoacross import cli
from myoacross.utils import datalists
from myoacross.embodiment import smplh_to_measurements as measurements
from myoacross.embodiment import generate as embodiment


class SubjectGenerationTests(TestCase):
    def test_measure_command_forwards_arguments(self):
        arguments = ["--split", "test", "--subject", "KIT_3", "--device", "cpu"]
        with patch.object(measurements, "main") as measure:
            cli.main(["measure", *arguments])
        measure.assert_called_once_with(arguments)

    def test_measurements_use_own_references_for_both_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for sid in (
                datalists.TRAIN_TARGET_SUBJECT_IDS + datalists.TEST_TARGET_SUBJECT_IDS
            ):
                dataset, _, subject = sid.partition("_")
                folder = root / dataset / subject
                folder.mkdir(parents=True)
                (folder / "shape.npz").touch()
            for split, count in (("train", 110), ("test", 31)):
                with patch.object(
                    measurements, "generate_measurements", return_value={}
                ) as generate:
                    with redirect_stdout(StringIO()):
                        measurements.main(
                            ["--split", split, "--dataset-root", str(root)]
                        )
                    selected = generate.call_args.args[0]
                    self.assertEqual(len(selected), count)
                    self.assertEqual(
                        tuple(selected), datalists.get_target_subject_ids(split=split)
                    )
                    for sid, paths in selected.items():
                        dataset, _, subject = sid.partition("_")
                        self.assertEqual(
                            paths, (root / dataset / subject / "shape.npz",)
                        )

    def test_measurements_reject_subject_from_other_split_before_discovery(self):
        subject = datalists.TEST_TARGET_SUBJECT_IDS[0]
        with TemporaryDirectory() as directory, self.assertRaises(ValueError):
            measurements.main(
                ["--split", "train", "--subject", subject, "--dataset-root", directory]
            )

    def test_embodiment_generation_uses_target_split_without_source_motions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.xml"
            baseline.touch()
            config = root / "height.yaml"
            config.write_text("name: height_scale\n")
            report = {
                "validation": {
                    "measurements": {"height": {"predicted": 1.8}},
                    "body_mass_kg": 70,
                    "target_body_mass_kg": None,
                }
            }
            for split, count in (("train", 110), ("test", 31)):
                with (
                    patch.object(embodiment, "get_xml_path", return_value=baseline),
                    patch.object(embodiment, "EMBODIMENTS_DIR", root / "outputs"),
                    patch.object(
                        embodiment, "generate", return_value=(None, None, report)
                    ) as generate,
                    patch.object(embodiment, "save", return_value=root / "model.xml"),
                    redirect_stdout(StringIO()),
                ):
                    embodiment.main(["--config", str(config), "--split", split])
                selected = tuple(call.args[1] for call in generate.call_args_list)
                self.assertEqual(len(selected), count)
                self.assertEqual(
                    selected, datalists.get_target_subject_ids(split=split)
                )
