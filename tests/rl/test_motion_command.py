"""Exercise real command indexing/sampling with an in-memory entity boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from myoacross.rl.embodiments.features import MorphologyFeatures
from myoacross.rl.tasks.motion_tracking.mdp.commands import (
    MyoFullBodyMotionCommand,
    MyoFullBodyMotionCommandCfg,
)
from myoacross.rl.tasks.motion_tracking.mdp.observations import motion_phase
from myoacross.rl.tasks.motion_tracking.mdp.terminations import motion_finished
from tests.rl.test_motion_library import write_motion


class Scene(dict):
    def __init__(self, robot, n):
        super().__init__(robot=robot)
        self.env_origins = torch.zeros(n, 3)


class Robot:
    """Only the simulation boundary is fake; command reset logic is real."""

    def __init__(self, n):
        self.body_names = ["Full Body", "anchor", "tip"]
        self.joint_names = ["j0", "j1"]
        self.num_actuators = 2
        quat = torch.zeros(n, 3, 4)
        quat[..., 0] = 1
        self.data = SimpleNamespace(
            indexing=SimpleNamespace(ctrl_ids=torch.arange(2)),
            soft_joint_pos_limits=torch.tensor([-1000.0, 1000.0]).repeat(n, 2, 1),
            joint_pos=torch.zeros(n, 2),
            joint_vel=torch.zeros(n, 2),
            body_link_pos_w=torch.zeros(n, 3, 3),
            body_link_quat_w=quat,
            body_link_lin_vel_w=torch.zeros(n, 3, 3),
            body_link_ang_vel_w=torch.zeros(n, 3, 3),
        )
        self.ctrl = torch.ones(n, 2)

    def find_bodies(self, names, preserve_order):
        return [self.body_names.index(name) for name in names], list(names)

    def write_joint_state_to_sim(self, pos, vel, env_ids):
        self.data.joint_pos[env_ids] = pos
        self.data.joint_vel[env_ids] = vel

    def write_root_state_to_sim(self, state, env_ids):
        self.data.body_link_pos_w[env_ids, 0] = state[:, :3]
        self.data.body_link_quat_w[env_ids, 0] = state[:, 3:7]

    def reset(self, env_ids):
        pass

    def write_ctrl_to_sim(self, ctrl, env_ids):
        self.ctrl[env_ids] = ctrl


def make_command(paths, n=8, mode="uniform"):
    robot = Robot(n)
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=0.1,
        common_step_counter=0,
        scene=Scene(robot, n),
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(n, dtype=torch.bool)
        ),
        sim=SimpleNamespace(
            mj_model=SimpleNamespace(actuator_actadr=[0, 1], actuator_actnum=[1, 1]),
            data=SimpleNamespace(act=torch.ones(n, 2)),
            forward=lambda: None,
        ),
    )
    cfg = MyoFullBodyMotionCommandCfg(
        motion_file=str(paths[0]),
        entity_name="robot",
        body_names=("Full Body", "anchor", "tip"),
        anchor_body_name="anchor",
        subject_features=MorphologyFeatures((1.0,) * 18),
        subject_height_m=1.8,
        sampling_mode=mode,
        resampling_time_range=(1e9, 1e9),
    )
    command = MyoFullBodyMotionCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return command, env


class MotionCommandTests(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.paths = tuple(Path(tmp.name) / f"{i}.npz" for i in range(2))
        write_motion(self.paths[0], 2, 0)
        write_motion(self.paths[1], 25, 100)
        self.command, self.env = make_command(self.paths)
        self.assertTrue(
            hasattr(self.command, "set_motions"), "multi-motion command not implemented"
        )
        self.command.set_motions(self.paths)

    def test_per_env_reference_and_lookahead(self):
        c = self.command
        c.motion_ids[:2] = torch.tensor([0, 1])
        c.time_steps[:2] = torch.tensor([0, 23])
        self.env.scene.env_origins[1, 0] = 1000
        self.assertEqual(c.joint_pos[:2, 0].tolist(), [0, 123])
        self.assertEqual(c.body_pos_w[:2, 2, 0].tolist(), [20, 1143])
        self.assertEqual(c.anchor_pos_w[:2, 0].tolist(), [10, 1133])
        future = c.lookahead
        self.assertEqual(future[0, :, 0].tolist(), [1, 1, 1, 1])
        self.assertEqual(future[1, :, 0].tolist(), [1124, 1124, 1124, 1124])
        torch.testing.assert_close(future[:2, :, -1], torch.full((2, 4), 0.1))
        torch.testing.assert_close(
            motion_phase(self.env, "motion")[:2, 0], torch.tensor([0.0, 23 / 24])
        )

    def test_subset_reset_isolated(self):
        torch.manual_seed(42)
        c = self.command
        c.motion_ids.fill_(1)
        c.time_steps.fill_(12)
        selected = torch.tensor([1, 3, 5, 7])
        c._resample_command(selected)
        self.assertEqual(c.time_steps[::2].tolist(), [12] * 4)
        self.assertEqual(c.motion_ids[::2].tolist(), [1] * 4)
        self.assertTrue(torch.all(self.env.sim.data.act[selected] == 0))
        self.assertTrue(torch.all(self.env.sim.data.act[::2] == 1))
        self.assertTrue(torch.all(c.robot.ctrl[selected] == 0))
        self.assertTrue(torch.all(c.robot.ctrl[::2] == 1))
        torch.testing.assert_close(
            c.robot.data.joint_pos[selected], c.joint_pos[selected]
        )

    def test_uniform_starts_exclude_terminal_frame(self):
        c = self.command
        seen = set()
        for _ in range(20):
            c._resample_command(torch.arange(8))
            seen.update(c.motion_ids.tolist())
            self.assertTrue(torch.all(c.time_steps < c.motion_lengths - 1))
        self.assertEqual(seen, {0, 1})
        self.assertEqual(int(c.consume_motion_sample_counts().sum()), 160)
        self.assertEqual(int(c.consume_motion_sample_counts().sum()), 0)

    def test_body_groups_keep_their_own_source_clips_after_resets(self):
        c = self.command
        # Different physical-body groups must never sample each other's source.
        c.set_motions(
            self.paths,
            motion_ranges=(
                torch.tensor([0] * 4 + [1] * 4),
                torch.ones(8, dtype=torch.long),
            ),
        )
        for _ in range(20):
            c._resample_command(torch.arange(8))
            self.assertEqual(c.motion_ids.tolist(), [0] * 4 + [1] * 4)
            self.assertTrue(torch.all(c.robot.data.joint_pos[:4, 0] < 100))
            self.assertTrue(torch.all(c.robot.data.joint_pos[4:, 0] >= 100))
        before = c.time_steps.clone()
        c._resample_command(torch.tensor([1, 5]))
        self.assertEqual(c.motion_ids.tolist(), [0] * 4 + [1] * 4)
        torch.testing.assert_close(
            c.time_steps[[0, 2, 3, 4, 6, 7]], before[[0, 2, 3, 4, 6, 7]]
        )

    def test_new_library_clears_old_body_group_ranges(self):
        c = self.command
        c._motion_ranges = (
            torch.ones(8, dtype=torch.long),
            torch.ones(8, dtype=torch.long),
        )
        c.set_motion(self.paths[0])
        self.assertIsNone(c._motion_ranges)
        c._resample_command(torch.arange(8))
        self.assertEqual(c.motion_ids.tolist(), [0] * 8)

    def test_cached_library_clears_or_rebinds_body_ranges(self):
        c = self.command
        library = c.motion_library
        starts, counts = (
            torch.zeros(8, dtype=torch.long),
            torch.ones(8, dtype=torch.long),
        )
        c.set_motions(self.paths, motion_ranges=(starts, counts))
        starts.fill_(1)  # Caller's buffers must not alias the installed ownership.
        c._resample_command(torch.arange(8))
        self.assertEqual(c.motion_ids.tolist(), [0] * 8)
        c.set_motions(self.paths, motion_ranges=(starts, counts))
        c._resample_command(torch.arange(8))
        self.assertEqual(c.motion_ids.tolist(), [1] * 8)
        c.set_motions(self.paths)
        self.assertIsNone(c._motion_ranges)
        self.assertIs(c.motion_library, library)

    def test_invalid_ranges_leave_old_library_and_binding_intact(self):
        c = self.command
        c.set_motions(
            self.paths,
            motion_ranges=(
                torch.zeros(8, dtype=torch.long),
                torch.ones(8, dtype=torch.long),
            ),
        )
        library, ranges = c.motion_library, c._motion_ranges
        for start, count in ((-1, 1), (0, 0), (1, 2), (2, 1)):
            with self.subTest(start=start, count=count), self.assertRaises(ValueError):
                c.set_motions(
                    self.paths,
                    motion_ranges=(torch.full((8,), start), torch.full((8,), count)),
                )
            self.assertIs(c.motion_library, library)
            self.assertIs(c._motion_ranges, ranges)
        with self.assertRaises(ValueError):
            c.set_motions(self.paths, motion_ranges=(torch.zeros(8), torch.ones(8)))
        with self.assertRaises(ValueError):
            c.set_motions(
                self.paths,
                motion_ranges=(
                    torch.zeros(7, dtype=torch.long),
                    torch.ones(7, dtype=torch.long),
                ),
            )

    def test_two_frame_clip_done_boundary(self):
        c = self.command
        c.motion_ids[1] = 1
        c._update_command(torch.tensor([0]))
        self.assertEqual(c.time_steps[:2].tolist(), [1, 0])
        self.assertEqual(
            motion_finished(self.env, "motion")[:2].tolist(), [True, False]
        )
        c._update_command(torch.tensor([0]))
        self.assertEqual(c.time_steps[:2].tolist(), [1, 0])
        self.assertEqual(c.motion_ids[:2].tolist(), [0, 1])

    def test_failed_install_is_atomic(self):
        c = self.command
        old = c.motion_library
        c.time_steps[1] = 1
        write_motion(self.paths[1], fps=2)
        with self.assertRaises(ValueError):
            c.set_motions((self.paths[1],))
        self.assertIs(c.motion_library, old)
        self.assertEqual(int(c.time_steps[1]), 1)

    def test_failure_uses_old_motion_and_bin(self):
        c = self.command
        c.cfg.sampling_mode = "adaptive"
        c._resample_command(torch.arange(8))  # Start valid episodes.
        c.motion_ids[:2] = torch.tensor([0, 1])
        c.time_steps[:2] = torch.tensor([0, 20])
        self.env.common_step_counter += 1
        self.env.termination_manager.terminated[:2] = True
        c._resample_command(torch.tensor([0, 1]))
        self.assertEqual(c._current_bin_failed[0].tolist(), [1.0])
        self.assertEqual(c._current_bin_failed[1].tolist(), [0.0, 0.0, 1.0])

    def test_repeated_failures_accumulate(self):
        c = self.command
        c.cfg.sampling_mode = "adaptive"
        c._resample_command(torch.arange(8))
        c.motion_ids.zero_()
        c.time_steps.zero_()
        self.env.common_step_counter += 1
        self.env.termination_manager.terminated[:2] = True
        c._resample_command(torch.tensor([0]))
        c._resample_command(torch.tensor([1]))
        self.assertEqual(c._current_bin_failed[0].tolist(), [2.0])

    def test_fresh_embodiment_ignores_stale_failure(self):
        c = self.command
        c.cfg.sampling_mode = "adaptive"
        self.env.termination_manager.terminated.fill_(True)
        c.reset_sampling_statistics()
        c._resample_command(torch.tensor([0, 1]))
        c._resample_command(torch.tensor([2, 3]))
        self.assertEqual(sum(float(v.sum()) for v in c._current_bin_failed), 0.0)

    def test_overlapping_resets_count_each_failure_once(self):
        c = self.command
        c.cfg.sampling_mode = "adaptive"
        c._resample_command(torch.arange(8))
        c.motion_ids.zero_()
        c.time_steps.zero_()
        self.env.common_step_counter += 1
        self.env.termination_manager.terminated[0] = True
        c._resample_command(torch.tensor([0]))
        c._resample_command(torch.tensor([0]))
        self.assertEqual(sum(float(v.sum()) for v in c._current_bin_failed), 1.0)
        # A genuinely new failure after another physical step still counts.
        self.env.common_step_counter += 1
        c._resample_command(torch.tensor([0]))
        self.assertEqual(sum(float(v.sum()) for v in c._current_bin_failed), 2.0)

    def test_fresh_body_repeated_reset_does_not_consume_stale_failure(self):
        c = self.command
        c.cfg.sampling_mode = "adaptive"
        self.env.termination_manager.terminated[0] = True
        c.reset_sampling_statistics()
        c._resample_command(torch.tensor([0]))
        c._resample_command(torch.tensor([0]))
        self.assertEqual(sum(float(v.sum()) for v in c._current_bin_failed), 0.0)

    def test_no_smoothing_across_clips(self):
        c = self.command
        c.cfg.sampling_mode = "adaptive"
        c.bin_failed_count[1].fill_(1000)
        c.motion_ids.zero_()
        c._adaptive_sampling(torch.arange(8))
        self.assertEqual(c.time_steps.tolist(), [0] * 8)
        self.assertTrue(torch.all(c.metrics["sampling_top1_prob"] == 1))

    def test_single_clip_evaluation_frame_alignment(self):
        c = self.command
        c.set_motion(self.paths[1])
        c.cfg.sampling_mode = "start"
        c.cfg.stop_at_motion_end = True
        c.reset(torch.arange(8))
        self.assertEqual(c.time_steps.tolist(), [0] * 8)
        c._update_command(torch.arange(8))
        self.assertEqual(c.time_steps.tolist(), [1] * 8)
        self.assertEqual(c.joint_pos[:, 0].tolist(), [101.0] * 8)
        for _ in range(30):
            c._update_command()
        self.assertEqual(c.time_steps.tolist(), [24] * 8)
        self.assertTrue(torch.all(motion_finished(self.env, "motion")))

    def test_selected_env_scrubber_uses_local_length(self):
        c = self.command
        c.motion_ids[:2] = torch.tensor([0, 1])
        # A frame valid in one clip must not index another packed clip.
        c.reset_to_frame(torch.tensor([0, 1]), 10)
        self.assertEqual(c.time_steps[:2].tolist(), [1, 10])
        self.assertEqual(c.robot.data.joint_pos[:2, 0].tolist(), [1, 110])
        self.assertEqual(c.time_steps[2:].tolist(), [0] * 6)
        c.reset_to_frame(torch.tensor([1]), -2)
        self.assertEqual(int(c.time_steps[1]), 0)

    def test_batched_failure_bins_and_decay_keep_clip_boundaries(self):
        c = self.command
        c.cfg.sampling_mode = "adaptive"
        c.motion_ids[:] = torch.tensor([0, 1, 1, 0, 1, 1, 1, 1])
        c.time_steps[:] = torch.tensor([1, 0, 12, 1, 24, 12, 0, 24])
        self.env.common_step_counter = 1
        c._episode_valid.fill_(True)
        self.env.termination_manager.terminated[:] = torch.tensor(
            [True, True, True, False, True, False, False, False]
        )
        expected = [torch.zeros_like(x) for x in c._current_bin_failed]
        for env_id in (0, 1, 2, 4):
            clip = int(c.motion_ids[env_id])
            width = expected[clip].numel()
            bin_id = min(int(c.time_steps[env_id]) * width // c._clip_lengths[clip], width - 1)
            expected[clip][bin_id] += 1
        c._resample_command(torch.arange(8))
        for actual, reference in zip(c._current_bin_failed, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        c._update_command()
        for actual, reference in zip(c.bin_failed_count, expected):
            torch.testing.assert_close(actual, reference * c.cfg.adaptive_alpha, rtol=0, atol=0)
        self.assertEqual(sum(float(v.sum()) for v in c._current_bin_failed), 0)

    def test_adaptive_sampling_matches_the_full_clip_loop(self):
        from copy import deepcopy
        import math

        c = self.command
        c.motion_ids[:] = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
        for counts in c.bin_failed_count:
            counts.copy_(torch.arange(counts.numel(), dtype=torch.float32) + 1)
        env_ids = torch.tensor([0, 2, 6])
        expected_steps = c.time_steps.clone()
        expected_metrics = deepcopy(c.metrics)
        torch.manual_seed(51)
        for clip, (length, counts) in enumerate(zip(c._clip_lengths, c.bin_failed_count)):
            ids = env_ids[c.motion_ids[env_ids] == clip]
            if not ids.numel():
                continue
            probabilities = counts + c.cfg.adaptive_uniform_ratio / counts.numel()
            padded = torch.nn.functional.pad(
                probabilities[None, None], (0, c.kernel.numel() - 1), mode="replicate"
            )
            probabilities = torch.nn.functional.conv1d(padded, c.kernel[None, None]).flatten()
            probabilities /= probabilities.sum()
            bins = torch.multinomial(probabilities, ids.numel(), replacement=True)
            expected_steps[ids] = (
                (bins + torch.rand(ids.numel())) / counts.numel() * (length - 1)
            ).long().clamp(max=length - 2)
            entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
            expected_metrics["sampling_entropy"][ids] = entropy / math.log(counts.numel()) if counts.numel() > 1 else 1.0
            pmax, imax = probabilities.max(0)
            expected_metrics["sampling_top1_prob"][ids] = pmax
            expected_metrics["sampling_top1_bin"][ids] = imax.float() / counts.numel()
        torch.manual_seed(51)
        c._adaptive_sampling(env_ids)
        torch.testing.assert_close(c.time_steps, expected_steps, rtol=0, atol=0)
        for name in expected_metrics:
            torch.testing.assert_close(c.metrics[name], expected_metrics[name], rtol=0, atol=0)
