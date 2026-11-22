"""Packed references must never confuse clip boundaries or body order."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import importlib.util

import numpy as np
import torch


def write_motion(path, frames=5, offset=0.0, fps=10.0, **changes):
    values = np.arange(frames, dtype=np.float32) + offset
    pos = np.zeros((frames, 3, 3), dtype=np.float32)
    pos[:, :, 0] = values[:, None] + np.array([0, 10, 20])
    quat = np.zeros((frames, 3, 4), dtype=np.float32)
    quat[:, :, 0] = 1
    data = dict(
        fps=np.array(fps),
        joint_pos=values[:, None].repeat(2, 1),
        joint_vel=np.zeros((frames, 2)),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros_like(pos),
        body_ang_vel_w=np.zeros_like(pos),
    )
    data.update(changes)
    np.savez(path, **data)


class MotionLibraryTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = tuple(Path(self.tmp.name) / f"{i}.npz" for i in range(2))
        write_motion(self.paths[0], 2, 0)
        write_motion(self.paths[1], 5, 100)

    def load(self, paths=None):
        name = "myoacross.rl.tasks.motion_tracking.motion_library"
        self.assertIsNotNone(
            importlib.util.find_spec(name), "MotionLibrary not implemented"
        )
        from myoacross.rl.tasks.motion_tracking.motion_library import MotionLibrary

        return MotionLibrary(
            self.paths if paths is None else paths,
            torch.tensor([2, 0]),
            num_joints=2,
            num_bodies=3,
            step_dt=0.1,
            device="cpu",
        )

    def test_offsets_and_selected_body_order(self):
        lib = self.load()
        self.assertEqual(lib.lengths.tolist(), [2, 5])
        self.assertEqual(lib.offsets.tolist(), [0, 2])
        self.assertEqual(lib.num_motions, 2)
        self.assertEqual(lib.joint_pos[[1, 2, 6], 0].tolist(), [1, 100, 104])
        self.assertEqual(lib.body_pos_w[2, :, 0].tolist(), [120, 100])

    def test_rejects_empty_or_duplicate_paths(self):
        for paths in ((), (self.paths[0], self.paths[0])):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                self.load(paths)

    def test_rejects_bad_fps_shape_nonfinite_and_short_clip(self):
        for changes in (
            dict(fps=20),
            dict(frames=1),
            dict(joint_pos=np.zeros((5, 3))),
            dict(body_pos_w=np.full((5, 3, 3), np.nan)),
            dict(fps=np.array([10.0, 10.0])),
        ):
            with self.subTest(changes=list(changes)):
                write_motion(self.paths[1], **changes)
                with self.assertRaises(ValueError):
                    self.load()

    def test_failed_construction_does_not_mutate_existing_library(self):
        lib = self.load()
        before = lib.joint_pos.clone()
        write_motion(self.paths[1], fps=-1)
        with self.assertRaises(ValueError):
            self.load()
        torch.testing.assert_close(lib.joint_pos, before)

    def test_preloaded_cpu_clips_reuse_arrays_without_reopening_files(self):
        from myoacross.rl.tasks.motion_tracking.motion_library import (
            clear_motion_clip_cache,
            preload_motion_clips,
        )

        self.addCleanup(clear_motion_clip_cache)
        original = self.load()
        preload_motion_clips(self.paths)
        with patch("numpy.load", side_effect=AssertionError("Cached clip reopened")):
            cached = self.load()
        for name in ("joint_pos", "body_pos_w", "body_quat_w"):
            torch.testing.assert_close(getattr(cached, name), getattr(original, name))
        # Mutating a packed library must not change the cached CPU arrays.
        cached.joint_pos.zero_()
        torch.testing.assert_close(self.load().joint_pos, original.joint_pos)
        clear_motion_clip_cache()
        write_motion(self.paths[1], 5, 200)
        self.assertEqual(float(self.load().joint_pos[2, 0]), 200)

    def test_preloading_replaces_the_previous_training_selection(self):
        from myoacross.rl.tasks.motion_tracking.motion_library import (
            clear_motion_clip_cache,
            preload_motion_clips,
        )

        self.addCleanup(clear_motion_clip_cache)
        preload_motion_clips(self.paths)
        write_motion(self.paths[1], 5, 200)
        preload_motion_clips(self.paths[:1])
        self.assertEqual(float(self.load().joint_pos[2, 0]), 200)
