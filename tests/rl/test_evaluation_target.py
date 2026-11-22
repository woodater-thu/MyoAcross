import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from myoacross.rl.runners import evaluate


class EvaluationTargetTests(TestCase):
    def run_evaluation(self, **options):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "data": {
                    "dataset_version": "embodiment_motion_v1",
                    "embodiment": "baseline",
                    "motion_ids": ["walk"],
                },
                "encoder": {"mode": "none"},
            }
            row = dict(
                subject_id="KIT_2",
                motion_id="walk",
                category="walk_medium",
                parameter=None,
                relative_change=0.0,
                coverage=1.0,
                completed=True,
                body_position_error_m=0.01,
                aligned_body_position_error_m=0.01,
                body_rotation_error_deg=1.0,
                joint_angle_error_deg=1.0,
                joint_velocity_error_deg_s=1.0,
                end_reason="completed",
            )
            with (
                patch.object(
                    evaluate.torch,
                    "load",
                    return_value={
                        "config": config,
                        "algorithm": {"encoder_mode": "none"},
                    },
                ),
                patch.object(
                    evaluate.DatasetRecord,
                    "from_split",
                    return_value=SimpleNamespace(subject_ids=("KIT_2",)),
                ) as load,
                patch.object(
                    evaluate, "_run_evaluation_batch", return_value=[row]
                ) as batch,
            ):
                evaluate.evaluate_policy(root / "baseline.pt", root / "out", **options)
                metadata = json.loads((root / "out/config.json").read_text())
                self.assertEqual(batch.call_args.args[2], [evaluate.EvaluationCondition()])
                self.assertTrue((root / "out/summary.json").exists())
                self.assertFalse((root / "out/sensitivity.csv").exists())
                self.assertFalse(list((root / "out").rglob("rollout.npz")))
                self.assertTrue(load.call_args.kwargs["require_quality_pass"])
                return load.call_args.kwargs, metadata

    def test_baseline_policy_uses_segment_test_targets(self):
        args, metadata = self.run_evaluation(
            target_embodiment="segment_scale", split="test"
        )
        self.assertEqual(args["embodiment"], "segment_scale")
        self.assertEqual(metadata["training_embodiment"], "baseline")
        self.assertEqual(metadata["target_embodiment"], "segment_scale")
        self.assertEqual(
            metadata["checkpoint_config"]["data"]["embodiment"], "baseline"
        )

    def test_default_keeps_checkpoint_embodiment(self):
        args, _ = self.run_evaluation()
        self.assertEqual(args["embodiment"], "baseline")

    def test_target_subset_is_forwarded(self):
        args, _ = self.run_evaluation(
            target_embodiment="segment_scale", subject_ids=["KIT_2"]
        )
        self.assertEqual(args["subject_ids"], ["KIT_2"])

    def test_motion_selection_cannot_expand_checkpoint(self):
        with patch.object(
            evaluate.torch,
            "load",
            return_value={
                "config": {
                    "data": {
                        "embodiment": "baseline",
                        "dataset_version": "embodiment_motion_v1",
                        "motion_ids": ["walk"],
                    }
                },
                "algorithm": {"encoder_mode": "none"},
            },
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint motion selection"):
                evaluate.evaluate_policy("unused.pt", "unused", motion_ids=["run"])

    def test_invalid_target_fails_before_loading_checkpoint(self):
        with patch.object(evaluate.torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "target_embodiment"):
                evaluate.evaluate_policy(
                    "missing.pt", "unused", target_embodiment="invalid"
                )
            load.assert_not_called()


class ExplicitCatalogTests(TestCase):
    def test_unregistered_catalog_bypasses_fixed_split(self):
        from myoacross.rl.embodiments.subjects import DatasetRecord, SubjectRecord, MotionRecord
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('model.xml', 'source.npz', 'motion.npz'):
                (root / name).touch()
            catalog = DatasetRecord([SubjectRecord('image_person', 'test', root / 'model.xml',
                (MotionRecord('walk', root / 'source.npz', root / 'motion.npz'),))])
            config = {'data': {'dataset_version': 'embodiment_motion_v1',
                              'embodiment': 'segment_scale', 'motion_ids': ['walk']}}
            row = dict(subject_id='image_person', motion_id='walk', category='walk',
                       completed=True, coverage=1., parameter=None, **{key: 0. for key in evaluate.ERROR_KEYS})
            with (
                patch.object(evaluate.torch, 'load', return_value={'config': config}),
                patch.object(evaluate.DatasetRecord, 'from_split') as fixed,
                patch.object(evaluate, '_run_evaluation_batch', return_value=[row]) as batch,
            ):
                evaluate.evaluate_policy(root / 'checkpoint.pt', root / 'evaluation', catalog=catalog)
                fixed.assert_not_called()
                self.assertIs(batch.call_args.args[0], catalog)
                config['data']['motion_ids'] = ['run']
                with self.assertRaisesRegex(ValueError, 'checkpoint motion selection'):
                    evaluate.evaluate_policy(root / 'checkpoint.pt', root / 'invalid', catalog=catalog)
                self.assertFalse((root / 'invalid').exists())
