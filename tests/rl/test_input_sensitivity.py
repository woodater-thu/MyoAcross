from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest import TestCase
import numpy as np
import torch
import mujoco
from myoacross.rl.runners import evaluate


class InputSensitivityTests(TestCase):
    def test_grid_has_one_baseline_and_three_samples_per_level(self):
        for target in ('morphology', 'reference', 'combined'):
            conditions = evaluate._build_conditions(dict(target=target, num_samples=3))
            self.assertEqual(len(conditions), 10)
            self.assertEqual(conditions[0], evaluate.EvaluationCondition())
            self.assertEqual([c.sample for c in conditions[1:4]], [0, 1, 2])
        with self.assertRaises(ValueError):
            evaluate._build_conditions(dict(relative_changes=[.05]))

    def test_signs_pair_levels_and_do_not_depend_on_world_order(self):
        a = evaluate._perturbation_signs(42, 'subject', 'motion', 0)
        b = evaluate._perturbation_signs(42, 'subject', 'motion', 0)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 18 + len(evaluate.REFERENCE_JOINT_NAMES))
        self.assertEqual(set(a), {-1, 1})
        self.assertFalse(np.array_equal(a, evaluate._perturbation_signs(42, 'subject', 'motion', 1)))

    def test_morphology_is_physical_and_does_not_modify_source(self):
        values = np.arange(1, 19, dtype=np.float32)
        signs = np.tile([-1, 1], 9)
        result = evaluate._prepare_morphology_perturbation(values, .05, signs)
        np.testing.assert_allclose(result, values * (1 + signs * .05), rtol=1e-6)
        np.testing.assert_array_equal(values, np.arange(1, 19))
        np.testing.assert_array_equal(evaluate._prepare_morphology_perturbation(values, 0, signs), values)

    def test_reference_fk_changes_leg_targets_and_keeps_root(self):
        xml = '<mujoco><worldbody><body name="Full Body"><freejoint/><geom size=".1" mass="1"/><body name="pelvis"><geom size=".1" mass="1"/>'
        for name in evaluate.REFERENCE_JOINT_NAMES:
            xml += f'<body name="{name}_body" pos="0 0 -.1"><joint name="{name}" axis="0 1 0"/><geom size=".03" mass=".1"/>'
        xml += '</body>' * (len(evaluate.REFERENCE_JOINT_NAMES) + 2) + '</worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        names = list(evaluate.REFERENCE_JOINT_NAMES)
        bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        clip = dict(joint_pos=np.zeros((2, len(names)), np.float32),
                    body_pos_w=np.repeat(data.xpos[None, 1:], 2, axis=0).astype(np.float32),
                    body_quat_w=np.repeat(data.xquat[None, 1:], 2, axis=0).astype(np.float32))
        original = {k: v.copy() for k, v in clip.items()}
        out = evaluate._prepare_reference_perturbation(model, names, bodies, clip, 5., np.ones(len(names)))
        np.testing.assert_allclose(out['joint_pos'], np.deg2rad(5), rtol=1e-6)
        np.testing.assert_allclose(out['body_pos_w'][:, :2], clip['body_pos_w'][:, :2])
        self.assertFalse(np.allclose(out['body_pos_w'][:, -1], clip['body_pos_w'][:, -1]))
        for k in clip: np.testing.assert_array_equal(clip[k], original[k])
        self.assertIs(evaluate._prepare_reference_perturbation(model, names, bodies, clip, 0, np.ones(len(names))), clip)

    def test_reference_observation_replacement_preserves_clean_buffers(self):
        dynamic = dict(joint=torch.zeros(1, 12), lookahead=torch.zeros(1, 2, 19), reference=torch.zeros(1, 10))
        dynamic['joint'][0, :2] = torch.tensor([1., 2.])
        command = SimpleNamespace(robot_anchor_pos_w=torch.tensor([[100., 200., 0.]]), robot_anchor_quat_w=torch.tensor([[1., 0, 0, 0]]))
        positions = torch.tensor([[[[1., 2, 3], [4., 5, 6]], [[7., 8, 9], [10., 11, 12]]]])
        world_positions = positions + command.robot_anchor_pos_w[:, None, None, :]
        result = evaluate._apply_reference_perturbation(
            dynamic, command, torch.tensor([0]), torch.tensor([[3., 4.]]), world_positions)
        torch.testing.assert_close(result['joint'][0, 4:6], torch.tensor([3., 4.]))
        torch.testing.assert_close(result['joint'][0, 8:10], torch.tensor([2., 2.]))
        torch.testing.assert_close(result['lookahead'][..., :6], positions.flatten(-2))
        self.assertIs(result['reference'], dynamic['reference'])
        self.assertEqual(dynamic['joint'][0, 4:6].sum(), 0)

    def test_summary_pairs_baseline_and_weights_subjects(self):
        rows = []
        for subject, scores in [('a', [1., 1.]), ('b', [0.])]:
            for score in scores:
                rows.append(dict(subject_id=subject, perturbation_target='reference',
                    morphology_magnitude=0., reference_magnitude_deg=5.,
                    **{k: score for k in evaluate.METRICS},
                    **{'delta_' + k: score - 1 for k in evaluate.METRICS}))
        result = evaluate._summarize_sensitivity(rows)[0]
        self.assertEqual(result['completed'], .5)
        self.assertEqual(result['completion_drop'], .5)

    def test_sweep_exports_one_baseline_and_new_protocol_fields(self):
        from unittest.mock import patch
        import json, csv
        config = dict(data=dict(dataset_version='embodiment_motion_v1', embodiment='segment_scale'))
        def batch(catalog, checkpoint, conditions, output, **kwargs):
            return [dict(subject_id='subject', motion_id='motion', category='walk',
                         parameter=None if i == 0 else c.target,
                         perturbation_target=c.target, morphology_magnitude=c.morphology_magnitude,
                         reference_magnitude_deg=c.reference_magnitude_deg, sample=c.sample,
                         **{k: 1. for k in evaluate.METRICS},
                         **{'delta_' + k: 0. for k in evaluate.METRICS})
                    for i, c in enumerate(conditions)]
        with tempfile.TemporaryDirectory() as temp, patch.object(evaluate.torch, 'load', return_value={
            'config':config, 'algorithm':dict(encoder_mode='none')}), patch.object(
                evaluate.DatasetRecord, 'from_split', return_value=SimpleNamespace(subject_ids=['subject'])), patch.object(
                evaluate, '_run_evaluation_batch', side_effect=batch):
            out = evaluate.evaluate_policy(Path(temp)/'checkpoint.pt', Path(temp)/'output',
                                           perturbation=dict(target='combined', num_samples=3))
            rows = list(csv.DictReader((out/'trials.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 10)
            self.assertEqual(sum(r['parameter'] == '' for r in rows), 1)
            self.assertEqual(json.loads((out/'summary.json').read_text())['num_trials'], 1)
            self.assertEqual(len(list(csv.DictReader((out/'sensitivity.csv').read_text().splitlines()))), 4)
            self.assertFalse(list(out.rglob('rollout.npz')))

    def test_configs_match_grid_and_use_separate_output_paths(self):
        from myoacross.utils.config import load_config
        from myoacross.utils.paths import PROJECT_ROOT
        paths = list((PROJECT_ROOT/'myoacross/configs/experiments/input_sensitivity').glob('*.yaml'))
        self.assertEqual(len(paths), 15)
        outputs = set()
        for path in paths:
            cfg = load_config(path)
            self.assertEqual(len(evaluate._build_conditions(cfg['perturbation'])), 10)
            self.assertEqual(cfg['envs_per_subject'], 64)
            self.assertNotIn(cfg['output_dir'], outputs)
            outputs.add(cfg['output_dir'])
