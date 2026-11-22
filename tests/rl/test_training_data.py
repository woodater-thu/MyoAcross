"""Virtual training samples original bodies and their generated variants."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from myoacross.rl.embodiments.bundle import EmbodimentBundle
from myoacross.rl.embodiments.features import MorphologyFeatures, load_morphology_features
from myoacross.rl.embodiments.subjects import DatasetRecord, MotionRecord, SubjectRecord
from myoacross.rl.embodiments.training_data import TrainingData, RolloutSelection
from myoacross.rl.embodiments.virtual.parameters import HeightScaleParameters
from myoacross.rl.runners.train_rl import discover_training_index_paths


class MixedTrainingDataTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'models/S/model.xml'
        self.model.parent.mkdir(parents=True)
        self.model.touch()
        motion_path = self.root / 'motion.npz'
        motion_path.touch()
        motion = MotionRecord('motion', motion_path, motion_path)
        self.subject = SubjectRecord('S', 'train', self.model, (motion,))
        self.dataset = DatasetRecord([self.subject])
        self.indices = {}
        self.bundles = {}
        for source, variant in (('subject', 'subject'), ('virtual', 'virtual_000000')):
            directory = self.root / 'bundles' / source / 'S'
            directory.mkdir(parents=True)
            features = MorphologyFeatures((1.0 if source == 'subject' else 1.1,) * 18)
            bundle = EmbodimentBundle(
                source_subject_id='S', variant_id=variant,
                embodiment='height_scale', morphology_source=source,
                parameters=None if source == 'subject' else HeightScaleParameters(1.1, 1.1),
                features=features, subject_height_m=features.values[0],
                model_signature='shared-layout', arrays={'body_mass': np.array([1.0])},
            )
            bundle_path = directory / f'{variant}.npz'
            bundle.save(bundle_path)
            common = dict(
                embodiment='height_scale', morphology_source=source,
                baseline_path=str(self.model), simulation_config={}, sampling=None,
            )
            (directory.parent / 'bank.json').write_text(json.dumps({
                **common, 'subject_ids': ['S'], 'count_per_subject': 1,
            }))
            index_path = directory / 'index.json'
            index_path.write_text(json.dumps({
                **common, 'source_subject_id': 'S', 'source_directory': str(self.model.parent),
                'model_signature': 'shared-layout', 'count': 1,
                'samples': [dict(variant_id=variant, bundle=bundle_path.name,
                                 features=features.to_dict(), subject_height_m=features.values[0])],
            }))
            self.indices[source] = index_path
            self.bundles[source] = bundle_path

    def discover(self, source):
        with patch('myoacross.rl.runners.train_rl.select_target_ids', return_value=['S']):
            return discover_training_index_paths(dict(
                embodiment='height_scale', morphology_source=source,
                bundle_root=str(self.root / 'bundles'),
            ))

    def test_training_requires_explicit_bundle_root(self):
        with self.assertRaisesRegex(ValueError, "bundle_root"):
            discover_training_index_paths(
                {"embodiment": "baseline", "morphology_source": "subject"}
            )

    def test_virtual_discovery_includes_original_features(self):
        paths = self.discover('virtual')
        self.assertEqual(set(paths), set(self.indices.values()))
        features, subject_ids = load_morphology_features(paths)
        self.assertEqual(subject_ids, {'S'})
        self.assertEqual({feature.values[0] for feature in features}, {1.0, 1.1})

    def test_virtual_training_samples_and_loads_both_sources(self):
        data = TrainingData(list(self.indices.values()), self.dataset,
                            embodiment='height_scale', morphology_source='virtual', seed=42)
        self.assertEqual(len(data.subjects), 1)
        self.assertEqual(len(data.subjects[0].variants), 2)
        selections = [data.sample() for _ in range(200)]
        self.assertEqual({selection.bundle_path for selection in selections}, set(self.bundles.values()))
        for source, path in self.bundles.items():
            variant = 'subject' if source == 'subject' else 'virtual_000000'
            bundle = data.load_bundle(RolloutSelection(self.subject, variant, path))
            self.assertEqual(bundle.morphology_source, source)
        self.assertTrue(all(selection.subject == self.subject for selection in selections))

    def test_subject_training_uses_only_original(self):
        paths = self.discover('subject')
        self.assertEqual(paths, [self.indices['subject']])
        data = TrainingData(paths, self.dataset, embodiment='height_scale',
                            morphology_source='subject', seed=42)
        self.assertEqual(data.sample().bundle_path, self.bundles['subject'])

    def test_virtual_discovery_requires_original_bank(self):
        (self.indices['subject'].parent.parent / 'bank.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.discover('virtual')
