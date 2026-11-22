"""Subject selection and block switching use complete registered clip groups."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from myoacross.rl.embodiments.training_data import TrainingData, TrainingSubject
from myoacross.rl.tasks.motion_tracking.environment import switch_training_embodiment
from myoacross.rl.embodiments.features import MorphologyFeatures
from tests.rl.test_motion_command import make_command
from tests.rl.test_motion_library import write_motion


class MotionTrainingTests(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.paths = tuple(Path(tmp.name) / f"{i}.npz" for i in range(3))
        for i, path in enumerate(self.paths):
            write_motion(path, 5 + i, i * 100)
        self.command, self.env = make_command(self.paths, mode="start")
        self.env.sim.wp_model = SimpleNamespace(body_mass=SimpleNamespace(device="cpu"))
        self.events = []

        def reset():
            self.events.append("reset")
            self.command.reset(torch.arange(8))
            return {}, {}

        self.env.reset = reset

    def selection(self, paths, subject_id="S"):
        motions = tuple(SimpleNamespace(retarget_path=path) for path in paths)
        return SimpleNamespace(
            subject=SimpleNamespace(subject_id=subject_id, motions=motions),
            variant_id="V",
            bundle_path=Path("bundle"),
            motion=motions[0],
        )

    def switch(self, selection, *, baseline=False, inject=True):
        bundle = SimpleNamespace(
            embodiment="baseline" if baseline else "segment_scale",
            source_subject_id=None if baseline else selection.subject.subject_id,
            variant_id="V",
            features=MorphologyFeatures((2.0,) * 18),
            subject_height_m=2.0,
        )
        switcher = SimpleNamespace(inject=lambda prepared: self.events.append("inject"))
        with (
            patch(
                "myoacross.rl.tasks.motion_tracking.environment.wp.synchronize_device"
            ),
            patch(
                "myoacross.rl.tasks.motion_tracking.environment.refresh_joint_limit_buffers"
            ),
        ):
            switch_training_embodiment(
                self.env,
                switcher,
                SimpleNamespace(bundle=bundle),
                selection,
                inject_model=inject,
            )

    def test_selection_contains_subject_not_single_motion(self):
        subject = self.selection(self.paths).subject
        data = object.__new__(TrainingData)
        data.subjects = (TrainingSubject(subject, (("V", Path("bundle")),)),)
        data.rng = np.random.default_rng(42)
        selection = data.sample()
        self.assertEqual(selection.subject.motions, subject.motions)
        self.assertFalse(
            hasattr(selection, "motion"), "block still selects only one motion"
        )

    def test_same_subject_variant_reuses_library(self):
        selection = self.selection(self.paths[:2])
        self.command.set_motions(self.paths[:2])
        before = self.command.motion_library
        self.command.bin_failed_count[0].fill_(3)
        self.switch(selection)
        self.assertIs(self.command.motion_library, before)
        self.assertEqual(float(self.command.bin_failed_count[0].sum()), 0.0)
        self.assertEqual(self.events, ["inject", "reset"])
        self.assertTrue(torch.all(self.command.subject_height_m == 2.0))

    def test_shared_baseline_bundle_subject_switch(self):
        self.switch(self.selection(self.paths[:2], "A"), baseline=True, inject=False)
        self.assertEqual(self.command.motion_paths, self.paths[:2])
        self.switch(self.selection(self.paths[1:], "B"), baseline=True, inject=False)
        self.assertEqual(self.command.motion_paths, self.paths[1:])
        self.assertEqual(self.events, ["reset", "reset"])

    def test_invalid_group_aborts_before_model_injection(self):
        selection = self.selection((self.paths[0], self.paths[2]))
        write_motion(self.paths[2], fps=2)
        old = self.command.motion_library
        with self.assertRaises(ValueError):
            self.switch(selection)
        self.assertIs(self.command.motion_library, old)
        self.assertEqual(self.events, [])
