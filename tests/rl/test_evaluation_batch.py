"""Deterministic task ownership, reset isolation and terminal metrics."""

from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from tests.rl.test_motion_command import make_command
from tests.rl.test_motion_library import write_motion


class QueueTests(TestCase):
    def test_unique_tasks_and_subject_ownership(self):
        from myoacross.rl.runners import evaluate as batch

        queue = batch.EvaluationQueue(
            [2, 3],
            2,
            [batch.EvaluationCondition(),
             batch.EvaluationCondition("morphology", .02, 0, 0),
             batch.EvaluationCondition("morphology", .02, 0, 1),
             batch.EvaluationCondition("reference", 0, 2.5, 0),
             batch.EvaluationCondition("reference", 0, 2.5, 1)],
        )
        seen = [set(), set()]
        ids = list(range(4))
        while ids:
            assigned = queue.assign(ids)
            ids = []
            for env_id, task in assigned.items():
                if task is None:
                    continue
                subject = env_id // 2
                self.assertEqual(task.subject_index, subject)
                key = (task.motion_index, task.condition_index)
                self.assertNotIn(key, seen[subject])
                seen[subject].add(key)
                ids.append(env_id)
        self.assertEqual([len(x) for x in seen], [10, 15])
        self.assertEqual(sum(c == 0 for _, c in seen[0]), 2)

    def test_reset_binding_is_explicit_and_owned(self):
        with TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}.npz" for i in range(2)]
            for i, p in enumerate(paths):
                write_motion(p, 25, i * 100)
            command, env = make_command(paths, n=4, mode="start")
            starts = torch.tensor([0, 0, 1, 1])
            command.set_motions(
                paths, motion_ranges=(starts, torch.ones(4, dtype=torch.long))
            )
            self.assertTrue(hasattr(command, "set_evaluation_motion_ids"))
            command.set_evaluation_motion_ids(torch.arange(4), starts)
            command._resample_command(torch.arange(4))
            self.assertEqual(command.motion_ids.tolist(), [0, 0, 1, 1])
            command.time_steps.fill_(7)
            command._resample_command(torch.tensor([1]))
            self.assertEqual(command.time_steps.tolist(), [7, 0, 7, 7])
            with self.assertRaises(ValueError):
                command.set_evaluation_motion_ids(torch.tensor([0]), torch.tensor([1]))
            command.set_motions(paths)
            self.assertIsNone(command._evaluation_motion_ids)

    def test_accumulator_masks_finished_and_keeps_terminal_step(self):
        from myoacross.rl.runners.evaluate import MetricAccumulator

        metrics = MetricAccumulator(2, "cpu")
        metrics.add(
            torch.tensor([True, True]), torch.ones(2, 5), torch.tensor([2.0, 3.0])
        )
        metrics.add(
            torch.tensor([False, True]),
            torch.full((2, 5), 3.0),
            torch.tensor([7.0, 4.0]),
        )
        self.assertEqual(metrics.steps.tolist(), [1, 2])
        torch.testing.assert_close(
            metrics.means(), torch.tensor([[1.0] * 5, [2.0] * 5], dtype=torch.float64)
        )
        self.assertEqual(metrics.reward.tolist(), [2.0, 7.0])
        metrics.reset(torch.tensor([0]))
        self.assertEqual(metrics.steps.tolist(), [0, 2])

    def test_rotation_metric_excludes_unscored_bodies(self):
        from types import SimpleNamespace
        from myoacross.rl.runners.evaluate import _compute_tracking_errors

        q = torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]])
        actual = q.clone()
        actual[:, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        pos = torch.zeros(1, 2, 3)
        cmd = SimpleNamespace(
            robot_body_pos_w=pos,
            robot_body_quat_w=actual,
            robot_anchor_pos_w=pos[:, 1],
            robot_anchor_quat_w=q[:, 1],
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.zeros(1, 1), joint_vel=torch.zeros(1, 1)
            )
        )
        ref = dict(
            pos=pos,
            quat=q,
            anchor_pos=pos[:, 1],
            anchor_quat=q[:, 1],
            joint=robot.data.joint_pos,
            velocity=robot.data.joint_vel,
        )
        error = _compute_tracking_errors(cmd, robot, ref, [1], torch.tensor([True]))
        self.assertEqual(error[0, 2].item(), 0.0)
