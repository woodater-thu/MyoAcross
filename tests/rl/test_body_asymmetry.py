"""Check physical asymmetry conditions and original reference ownership."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from myoacross.rl.runners import evaluate
from myoacross.rl.runners import generate_embodiment_bundles as generation


class BodyAsymmetryTests(TestCase):
    def test_conditions_include_one_original_and_both_sides(self):
        conditions = generation._build_body_conditions({
            'model_root': 'models', 'relative_shortening': [0, 0.01, 0.025, 0.05, 0.1],
            'sides': ['left', 'right'],
        })
        self.assertEqual(len(conditions), 9)
        self.assertEqual(conditions[0], ('original', None, 0.0))
        self.assertEqual({(side, value) for _, side, value in conditions[1:]},
                         {(side, value) for side in ('left', 'right')
                          for value in (0.01, 0.025, 0.05, 0.1)})

    def test_invalid_shortening_is_rejected(self):
        for value in (-0.1, 1.0, float('nan'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generation._build_body_conditions({'model_root': 'models',
                    'relative_shortening': [value], 'sides': ['left']})

    def test_fixed_generation_changes_only_selected_region_scales(self):
        from myoacross.rl.embodiments.virtual.parameters import REGION_NAMES
        from myoacross.embodiment import export

        with TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / 'model.xml'
            xml.touch()
            scales = {name: 1.0 + index / 100 for index, name in enumerate(REGION_NAMES)}
            (root / 'report.json').write_text(json.dumps({
                'scales': {'regions': scales}, 'mass': {'actual_kg': 70},
                'baseline_path': str(xml), 'subject_id': 'KIT_2',
            }))
            for side, suffix in (('left', 'l'), ('right', 'r')):
                condition = dict(name=side, shortened_side=side, relative_shortening=0.05)
                with patch.object(generation.mujoco.MjSpec, 'from_file'), \
                     patch('myoacross.rl.embodiments.virtual.generator.VirtualEmbodiment',
                           autospec=True) as body, \
                     patch.object(export, 'save'):
                    body.return_value.generate.return_value = (None, None, None, {})
                    models = list(generation._iter_models(root, sampler=None, count=1, seed=42,
                         body_condition=condition, model_root=root/'models'))
                    self.assertIsNone(models[0][0])
                    parameters = body.call_args.args[1]
                    expected = dict(scales)
                    expected['thigh_' + suffix] *= 0.95
                    expected['shank_' + suffix] *= 0.95
                    self.assertEqual(parameters.region_scales,
                                     tuple(expected[name] for name in REGION_NAMES))
                    self.assertEqual(parameters.body_mass_kg, 70)

    def test_loaded_body_keeps_original_motion_records(self):
        from myoacross.rl.embodiments.subjects import DatasetRecord, SubjectRecord, MotionRecord
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/'original.xml'
            original.touch()
            motion = root/'motion.npz'
            motion.touch()
            changed = root/'changed'
            changed.mkdir()
            (changed/'model.xml').touch()
            subject = SubjectRecord('KIT_2', 'test', original, (MotionRecord('walk', motion, motion),))
            bank = root/'bank/KIT_2'
            bank.mkdir(parents=True)
            (bank/'index.json').write_text(json.dumps({
                'source_subject_id': 'KIT_2', 'source_directory': str(changed),
                'samples': [{'features': {'values': [1.0]*len(evaluate.FEATURE_NAMES)}}],
            }))
            catalog, geometry = evaluate._load_body_catalog(DatasetRecord([subject]), root/'bank')
            self.assertIs(catalog.get('KIT_2').motions, subject.motions)
            self.assertEqual(catalog.get('KIT_2').xml_path, changed/'model.xml')
            self.assertEqual(geometry['KIT_2']['leg_length_difference_mm'], 0)

    def test_preloaded_bank_uses_runtime_capacities_but_requires_same_physics(self):
        from dataclasses import asdict
        from mjlab.sim.sim import SimulationCfg
        from mujoco_warp._src import sensor
        from myoacross.rl.embodiments.subjects import DatasetRecord, SubjectRecord, MotionRecord
        with TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root/'model.xml'
            xml.touch()
            motion = root/'motion.npz'
            motion.touch()
            (root/'report.json').write_text(json.dumps({'strategy': 'segment_scale'}))
            catalog = DatasetRecord([SubjectRecord('KIT_2', 'test', xml,
                                      (MotionRecord('walk', motion, motion),))])
            bank = root/'bank'
            (bank/'KIT_2').mkdir(parents=True)
            (bank/'KIT_2/index.json').write_text(json.dumps({
                'source_subject_id': 'KIT_2', 'samples': [{'bundle': 'subject.npz'}],
            }))
            simulation = asdict(SimulationCfg())
            (bank/'bank.json').write_text(json.dumps({
                'embodiment': 'segment_scale', 'morphology_source': 'subject',
                'simulation_config': simulation, 'subject_ids': ['KIT_2'],
                'baseline_path': str(xml),
            }))
            config = {'environment': {}, 'simulation': dict(simulation, nconmax=128, njmax=512)}
            bundle = SimpleNamespace(features=None, subject_height_m=1.7)
            with patch.object(sensor, 'MYOACROSS_BATCHED_SITE_SIZE', True, create=True), \
                 patch.object(evaluate.EmbodimentBundle, 'load', return_value=bundle), \
                 patch.object(evaluate, 'generate_embodiment_bundles', side_effect=AssertionError('export')), \
                 patch.object(evaluate, 'create_tracking_environment', side_effect=RuntimeError('environment reached')):
                with self.assertRaisesRegex(RuntimeError, 'environment reached'):
                    evaluate._initialize_batch(catalog, config, root/'policy.pt', root/'out',
                                               2, 'cuda:0', 42, bundle_root=bank)
                config['simulation']['mujoco']['timestep'] = 0.003
                with self.assertRaisesRegex(ValueError, 'simulation'):
                    evaluate._initialize_batch(catalog, config, root/'policy.pt', root/'out',
                                               2, 'cuda:0', 42, bundle_root=bank)

    def test_generation_exports_test_conditions_and_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'data': {'embodiment': 'segment_scale', 'split': 'test',
                              'bundle_root': str(root/'bank')},
                      'sampling': None, 'count_per_subject': None, 'seed': None,
                      'workers': 2,
                      'body_perturbation': {'model_root': str(root/'models'),
                          'relative_shortening': [0, 0.025], 'sides': ['left','right']}}
            with patch.object(generation, 'select_target_ids', return_value=['KIT_2']), \
                 patch.object(generation, 'generate_embodiment_bundles', return_value=[]):
                generation.run_generation(config)
            manifest = json.loads((root/'bank/conditions.json').read_text())
            self.assertEqual(manifest['split'], 'test')
            self.assertEqual([c['name'] for c in manifest['conditions']],
                             ['original', 'left_2p5pct', 'right_2p5pct'])

    def test_sweep_preserves_references_and_pairs_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = SimpleNamespace(subject_ids=('KIT_2',))
            config = {'data': {'dataset_version': 'embodiment_motion_v1',
                              'embodiment': 'segment_scale', 'motion_ids': ['walk']}}
            geometry = {'KIT_2': {'thigh_asymmetry_pct': 2.0,
                                 'shank_asymmetry_pct': 2.0,
                                 'leg_length_asymmetry_pct': 2.0,
                                 'leg_length_difference_mm': 16.0}}
            bank_root = root/'bank'
            bank_root.mkdir()
            (bank_root/'conditions.json').write_text(json.dumps({
                'split': 'test', 'embodiment': 'segment_scale',
                'conditions': [dict(name=n, shortened_side=s, relative_shortening=d)
                               for n,s,d in [('original',None,0), ('left_2p5pct','left',0.025),
                                             ('right_2p5pct','right',0.025)]],
            }))
            calls = []

            def run(c, checkpoint, conditions, output, **options):
                calls.append((c, conditions, options['bundle_root']))
                return [dict(subject_id='KIT_2', motion_id='walk', category='walk',
                             parameter=None, relative_change=0.0,
                             reference_path='original/motion.npz',
                             **{key: 1.0 if len(calls) == 1 else 0.5 for key in evaluate.METRICS})]

            with patch.object(evaluate.torch, 'load', return_value={
                    'config': config, 'algorithm': {'encoder_mode': 'rl'}}), \
                 patch.object(evaluate.DatasetRecord, 'from_split', return_value=catalog), \
                 patch.object(evaluate, '_load_body_catalog', return_value=(catalog, geometry)), \
                 patch.object(evaluate, '_run_evaluation_batch', side_effect=run):
                evaluate.evaluate_policy(root / 'policy.pt', root / 'out',
                    bundle_root=bank_root)
            self.assertEqual([bank.name for _,_,bank in calls],
                             ['original','left_2p5pct','right_2p5pct'])
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(c is catalog and conditions == [evaluate.EvaluationCondition()]
                                for c, conditions, bank in calls))
            with (root / 'out/trials.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r['reference_path'] == 'original/motion.npz' for r in rows))
            self.assertEqual(float(rows[1]['body_delta_completed']), -0.5)
            summary = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(summary['completed'], 1.0)
            self.assertTrue((root / 'out/body_conditions.csv').exists())
