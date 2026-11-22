"""Check image-derived measurements and experiment dispatch without HSMR weights."""

import json
import importlib.util
from pathlib import Path
import tempfile
from unittest import TestCase, skipUnless
from unittest.mock import patch
import numpy as np

from myoacross.utils.image2msksim import reconstruct, run


class InputImageTests(TestCase):
    def test_image_input_keeps_existing_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "person.jpg"
            path, source = reconstruct._prepare_input_image(
                dict(image_path=str(image)), root / "output"
            )
            self.assertEqual(path, image)
            self.assertEqual(source, dict(image_path=str(image)))
            self.assertFalse((root / "output").exists())

    def test_requires_one_input_and_an_explicit_video_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            invalid = [
                {},
                dict(image_path="person.jpg", video_path="person.mp4"),
                dict(video_path="person.mp4"),
                *[dict(video_path="person.mp4", video_frame_index=value)
                  for value in (-1, 1.5, True)],
            ]
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    reconstruct._prepare_input_image(config, output)
            self.assertFalse(output.exists())


@skipUnless(importlib.util.find_spec("cv2"), "Run video tests in the HSMR environment")
class VideoFrameTests(TestCase):
    def test_extracts_requested_frame_and_preserves_color(self):
        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "person.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            colors = [(200, 20, 10), (20, 150, 230), (50, 60, 70)]
            try:
                for color in colors:
                    writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
            finally:
                writer.release()
            config = dict(video_path=str(video), video_frame_index=1)
            path, source = reconstruct._prepare_input_image(config, root / "output")
            self.assertEqual(path, root / "output/input.png")
            np.testing.assert_allclose(cv2.imread(str(path))[24, 32], colors[1], atol=5)
            self.assertEqual(source["video_path"], str(video))
            self.assertEqual(source["video_frame_index"], 1)
            self.assertAlmostEqual(source["video_fps"], 10.0)
            self.assertEqual(source["image_path"], "input.png")
            with self.assertRaises(FileExistsError):
                reconstruct._prepare_input_image(config, root / "output")
            with self.assertRaisesRegex(ValueError, "frame"):
                reconstruct._prepare_input_image(
                    dict(video_path=str(video), video_frame_index=10), root / "invalid"
                )
            self.assertFalse((root / "invalid").exists())


class ReconstructionTests(TestCase):
    def test_multi_person_requires_selection(self):
        with self.assertRaisesRegex(ValueError, "person_index"):
            reconstruct.select_person(2, None)
        self.assertEqual(reconstruct.select_person(1, None), 0)
        self.assertEqual(reconstruct.select_person(3, 2), 2)
        with self.assertRaises(ValueError):
            reconstruct.select_person(1, 2)
        with self.assertRaises(ValueError):
            reconstruct.select_person(0, None)

    def test_export_calibrates_lengths_and_preserves_shape_and_mass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            betas = np.arange(10, dtype=np.float32)
            joints = np.zeros((24, 3))
            with (
                patch(
                    "myoacross.embodiment.smplh_to_measurements.compute_tpose_geometry",
                    return_value=(joints, ("pelvis",)),
                ) as geometry,
                patch(
                    "myoacross.embodiment.smplh_to_measurements.compute_measurements",
                    return_value={"thigh_length_l": 0.42, "hip_width": 0.15,
                                  "shoulder_ankle_height_l": 1.25},
                ),
            ):
                path = reconstruct.export_measurements(
                    "image_person",
                    "male",
                    betas,
                    73.0,
                    directory,
                    height_m=1.8,
                    reconstructed_height_m=1.5,
                    model_dir=directory / "skel",
                    device="cpu",
                )
            document = json.loads(path.read_text())
            measurements = document["measurements"]
            self.assertAlmostEqual(measurements["thigh_length_l"], 0.504)
            self.assertAlmostEqual(measurements["hip_width"], 0.18)
            self.assertAlmostEqual(measurements["shoulder_ankle_height_l"], 1.5)
            self.assertEqual(measurements["body_mass"], 73.0)
            self.assertEqual(measurements["height"], 1.8)
            self.assertEqual(document["betas"], betas.tolist())
            np.testing.assert_array_equal(geometry.call_args.args[1], betas)
            self.assertEqual(geometry.call_args.kwargs["model_dir"], directory / "skel")
            with self.assertRaises(FileExistsError):
                reconstruct.export_measurements(
                    "image_person",
                    "male",
                    betas,
                    73.0,
                    directory,
                    height_m=1.8,
                    reconstructed_height_m=1.5,
                    model_dir=directory / "skel",
                    device="cpu",
                )

    def test_reconstructed_height_uses_shape_in_standard_pose(self):
        import torch
        from types import SimpleNamespace

        template = torch.tensor([[0., -0.1, 0.], [0., 1.6, 0.]])
        directions = torch.zeros(2, 3, 10)
        directions[1, 1, 0] = 0.1
        model = SimpleNamespace(skin_template_v=template, shapedirs=directions)
        betas = np.zeros(10, dtype=np.float32)
        betas[0] = 2
        self.assertAlmostEqual(reconstruct.compute_reconstructed_height(model, betas), 1.9, places=6)


class DispatchTests(TestCase):
    def test_pipeline_uses_separate_environments_and_fixed_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(
                subject_id="image_person",
                image_path=str(root / "person.jpg"),
                body_mass_kg=73.0,
                height_m=1.8,
                person_index=None,
                output_root=str(root / "experiment"),
                cache_root=str(root / "cache"),
                hsmr=dict(
                    python="tools/hsmr/.venv/bin/python",
                    model_root="data/experiments/image2msksim/resources",
                    device="cuda:0",
                ),
                retarget_python="tools/gmr_fit/.venv/bin/python",
                embodiment_config="myoacross/configs/embodiment/segment_scale.yaml",
                retarget_config="myoacross/configs/retarget/gmr_fit.yaml",
                checkpoint_path="data/myoacross/rl/checkpoint/final.pt",
                policy_scheme="segment_film",
                motion_ids=None,
                device="cuda:0",
                envs_per_subject=64,
            )
            resources = root / "experiment/resources"
            resources.mkdir(parents=True)
            (resources / "weights").write_text("existing weights")
            with patch.object(run.subprocess, "run") as launch:
                output = run.run_image2msksim(config)
            self.assertEqual(output, root / "experiment")
            self.assertEqual((resources / "weights").read_text(), "existing weights")
            from myoacross.utils.config import load_config

            self.assertEqual(load_config(output / "config.yaml"), config)
            commands = [call.args[0] for call in launch.call_args_list]
            self.assertEqual(len(commands), 4)
            self.assertTrue(commands[0][0].endswith("tools/hsmr/.venv/bin/python"))
            self.assertIn("myoacross.utils.image2msksim.reconstruct", commands[0])
            self.assertIn("tools/gmr_fit/.venv/bin/python", commands[2][0])
            self.assertEqual(commands[1][-1], "embodiment")
            self.assertEqual(commands[2][-1], "retarget")
            self.assertEqual(commands[3][-1], "evaluate")
            self.assertTrue(all(call.kwargs["check"] for call in launch.call_args_list))
            self.assertEqual(launch.call_args.kwargs["env"]["WARP_CACHE_PATH"],
                             str(root / "cache/warp"))
            with self.assertRaises(FileExistsError):
                run.run_image2msksim(config)

    def test_checkpoint_motion_selection_is_reused(self):
        from types import SimpleNamespace

        specs = (SimpleNamespace(motion_id="walk"), SimpleNamespace(motion_id="run"))
        checkpoint = {
            "config": {
                "data": {"motion_ids": ["walk"], "motion_set": "kit_flat_locomotion_72"}
            }
        }
        config = dict(checkpoint_path="unused.pt", motion_ids=None)
        with (
            patch("torch.load", return_value=checkpoint),
            patch("myoacross.utils.datalists.get_motion_specs", return_value=specs),
        ):
            selected, motion_set = run._motion_specs(config)
            self.assertEqual([spec.motion_id for spec in selected], ["walk"])
            self.assertEqual(motion_set, "kit_flat_locomotion_72")
            config["motion_ids"] = ["run"]
            with self.assertRaisesRegex(ValueError, "checkpoint motion selection"):
                run._motion_specs(config)

    def test_negative_mass_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            with self.assertRaisesRegex(ValueError, "body_mass_kg"):
                run.run_image2msksim(
                    dict(subject_id="person", body_mass_kg=-1, output_root=str(root))
                )
            self.assertFalse(root.exists())

    def test_invalid_height_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            for height in (0, -1, float("nan")):
                with self.assertRaisesRegex(ValueError, "height_m"):
                    run.run_image2msksim(dict(subject_id="person", body_mass_kg=73,
                                             height_m=height, output_root=str(root)))
                self.assertFalse(root.exists())


class BatchTests(TestCase):
    def test_batch_records_failures_and_summarizes_successful_subjects(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "person.png"
            image.touch()
            records = [dict(subject_id="image_a", image_path=str(image), body_mass_kg=63, height_m=1.7,
                            source_subject_id="original_a", source_split="train"),
                       dict(subject_id="image_b", image_path=str(image), body_mass_kg=70, height_m=1.8,
                            source_subject_id="original_b", source_split="test")]
            manifest = root / "subjects.json"
            manifest.write_text(json.dumps(dict(subjects=records)))
            config = dict(subjects_path=str(manifest), output_root=str(root / "output"),
                          checkpoint_path=str(root / "checkpoint.pt"), policy_scheme="segment_film",
                          cache_root=str(root / "cache"))
            (root / "checkpoint.pt").touch()
            def execute(subject):
                if subject["subject_id"] == "image_a":
                    raise subprocess.CalledProcessError(1, ["python", "--stage", "retarget"])
                folder = Path(subject["output_root"]) / "evaluations/segment_film/test_iter15000"
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(json.dumps(dict(num_subjects=1, num_trials=2,
                                                                      completed=1.0, coverage=1.0)))
                return Path(subject["output_root"])
            with patch.object(run, "run_image2msksim", side_effect=execute) as launch,                  patch("torch.load", return_value=dict(iteration=15000)):
                output = run.run_dataset(config)
                self.assertEqual(launch.call_count, 2)
                second = launch.call_args_list[1].args[0]
                self.assertEqual(second["cache_root"], config["cache_root"])
                self.assertEqual(second["image_path"], str(image))
                self.assertEqual(second["height_m"], 1.8)
                self.assertIsNone(second["video_path"])
                report = json.loads((output / "dataset_summary.json").read_text())
                self.assertEqual(report["num_failed_subjects"], 1)
                self.assertEqual(report["num_completed_subjects"], 1)
                self.assertEqual(report["subjects"][0]["failed_stage"], "retarget")
                self.assertEqual(report["by_source_split"]["test"]["completed"], 1.0)
                self.assertEqual(report["overall"]["num_completed_subjects"], 1)
                self.assertEqual(report["overall"]["num_requested_subjects"], 2)
                with self.assertRaises(FileExistsError):
                    run.run_dataset(config)

    def test_duplicate_subjects_are_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "subjects.json"
            manifest.write_text(json.dumps(dict(subjects=[dict(subject_id="same"), dict(subject_id="same")])))
            with patch.object(run, "run_image2msksim") as launch:
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    run.run_dataset(dict(subjects_path=str(manifest), output_root=str(root / "output")))
                launch.assert_not_called()


class InstalledPackageTests(TestCase):
    def test_prepare_adds_missing_modules_without_overwriting_installed_code(self):
        import io
        import zipfile
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "lib").mkdir()
            (root / "lib/__init__.py").write_text("original")
            archive_data = io.BytesIO()
            with zipfile.ZipFile(archive_data, "w") as archive:
                archive.writestr("HSMR-abc/lib/__init__.py", "replace")
                archive.writestr("HSMR-abc/lib/kits/hsmr_demo.py", "inference")
                archive.writestr("HSMR-abc/configs/base.yaml", "configuration")
                archive.writestr("HSMR-abc/demo.jpg", "unused")
            package = SimpleNamespace(
                locate_file=lambda name: root,
                read_text=lambda name: json.dumps({"vcs_info": {"commit_id": "abc"}}),
            )
            with (
                patch.object(reconstruct, "distribution", return_value=package),
                patch.object(
                    reconstruct.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(archive_data.getvalue()),
                ) as download,
            ):
                self.assertEqual(reconstruct.prepare_hsmr_package(), root)
                self.assertEqual((root / "lib/__init__.py").read_text(), "original")
                self.assertEqual(
                    (root / "lib/kits/hsmr_demo.py").read_text(), "inference"
                )
                self.assertTrue((root / "configs/base.yaml").exists())
                self.assertTrue((root / "data_inputs").exists())
                self.assertFalse((root / "demo.jpg").exists())
                reconstruct.prepare_hsmr_package()
                self.assertEqual(download.call_count, 1)
