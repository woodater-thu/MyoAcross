"""Analyze accepted perturbations without mixing original bodies into samples."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from myoacross.rl.embodiments.features import FEATURE_NAMES
from myoacross.rl.embodiments.virtual.parameters import REGION_NAMES
from myoacross.utils.analysis import analyze_asymmetry as analysis
from myoacross.utils.plotting import plot_model_banks as plotting


class PerturbationAnalysisTests(TestCase):
    def test_extreme_sample_uses_geometry_and_excludes_mass(self):
        bank = dict(feature_names=['trunk_length', 'thigh_length_r', 'body_mass'], rows=[
            dict(subject='A', variant='virtual_0', feature_changes=np.array([10., 10., 90.])),
            dict(subject='B', variant='virtual_1', feature_changes=np.array([-30., 40., 0.])),
        ])
        selected = analysis.select_extreme_samples({'asymmetry_enhanced': bank})['asymmetry_enhanced']
        self.assertEqual(selected['subject'], 'B')
        self.assertEqual(selected['variant'], 'virtual_1')
        self.assertAlmostEqual(selected['geometry_rms_percent'], np.sqrt(1250))

    def test_bank_analysis_pairs_each_virtual_body_with_its_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = list(FEATURE_NAMES)
            parameter_names = list(REGION_NAMES) + ['body_mass_kg']
            bank = root / 'bank'
            sampling = dict(parameter_names=parameter_names,
                            covariance=np.eye(len(parameter_names)).tolist(),
                            max_relative_change=[0.5] * len(parameter_names),
                            covariance_parameters={'global_size_std': 0.01})
            (bank / 'virtual').mkdir(parents=True)
            (bank / 'virtual/bank.json').write_text(json.dumps({'sampling': sampling}))
            for sid, size, noise in [('A', 1.0, [-0.1, 0.0, 0.1]),
                                     ('B', 2.0, [-0.2, 0.0, 0.2])]:
                source = bank / 'subject' / sid
                virtual = bank / 'virtual' / sid
                model = root / 'models/segment_scale' / sid
                for path in (source, virtual, model):
                    path.mkdir(parents=True)
                original = [size] * len(names)
                (source / 'index.json').write_text(json.dumps({'samples': [
                    {'features': {'names': names, 'values': original}, 'subject_height_m': size}
                ]}))
                (model / 'report.json').write_text(json.dumps({
                    'scales': {'regions': dict.fromkeys(REGION_NAMES, size)},
                    'mass': {'actual_kg': size},
                }))
                samples = []
                for index, delta in enumerate(noise):
                    filename = f'virtual_{index}.npz'
                    parameters = dict(region_names=list(REGION_NAMES),
                                      region_scales=[size * (1 + delta)] * len(REGION_NAMES),
                                      body_mass_kg=size * (1 + delta))
                    np.savez(virtual / filename, metadata=json.dumps({'parameters': parameters}))
                    samples.append(dict(variant_id=f'virtual_{index}', bundle=filename,
                                        features={'names': names, 'values': [size * (1 + delta)] * len(names)},
                                        subject_height_m=size * (1 + delta)))
                (virtual / 'index.json').write_text(json.dumps({
                    'source_subject_id': sid, 'sampling': sampling, 'samples': samples,
                }))
            with patch.object(analysis, 'EMBODIMENTS_DIR', root / 'models'):
                data = analysis.load_bank(bank)
            self.assertEqual(len(data['rows']), 6)
            self.assertEqual(data['subject_ids'], ['A', 'B'])
            np.testing.assert_allclose(data['parameter_changes'][:, 0], [-10, 0, 10, -20, 0, 20])
            np.testing.assert_allclose(data['feature_changes'][:, 0], [-10, 0, 10, -20, 0, 20])
            summary = analysis.summarize_bank(data)
            self.assertEqual(summary['samples'], 6)
            self.assertAlmostEqual(summary['parameters']['std_percent'][0], np.std([-10, 0, 10, -20, 0, 20], ddof=1))

    def test_default_comparison_keeps_both_schemes_with_equal_global_size_noise(self):
        values = np.array([[-1., -2.], [1., 2.]])
        rows = [dict(subject='A', variant=f'virtual_{i}',
                     parameter_changes=row, feature_changes=row) for i, row in enumerate(values)]
        data = dict(bank='bank', subject_ids=['A'], originals={'A': np.ones(2)},
                    rows=rows, parameter_names=['trunk', 'body_mass_kg'],
                    feature_names=['trunk_length', 'body_mass'],
                    parameter_changes=values, feature_changes=values,
                    sampling={'covariance': np.eye(2).tolist(),
                              'covariance_parameters': {'global_size_std': 0.01}})
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'analysis'
            with patch.object(analysis, 'load_bank', return_value=data), \
                 patch.object(plotting, 'plot_distributions'), \
                 patch.object(plotting, 'plot_extreme_bodies'):
                analysis.main(['--output', str(output)])
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(set(report['schemes']), {'normal', 'asymmetry_enhanced'})
            self.assertEqual(len((output / 'samples.csv').read_text().splitlines()), 5)

    def test_output_directory_is_not_overwritten(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'existing'
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                analysis.main(['--output', str(output)])
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_two_schemes_share_axes_and_export_editable_figures(self):
        banks, summaries = {}, {}
        values = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 0.0]])
        for label, factor in [('normal', 1), ('asymmetry_enhanced', 5)]:
            data = dict(bank=label, subject_ids=['A'], rows=[{}, {}, {}],
                        sampling={'covariance': (np.eye(2) * factor ** 2).tolist()},
                        parameter_names=['trunk', 'body_mass_kg'],
                        feature_names=['trunk_length', 'body_mass'],
                        parameter_changes=values * factor,
                        feature_changes=values * factor / 2)
            banks[label] = data
            summaries[label] = analysis.summarize_bank(data)
        figures = {}
        exports = []

        def capture(figure, path, **kwargs):
            path = Path(path)
            figures[path.stem] = figure
            exports.append(path)

        with patch('matplotlib.figure.Figure.savefig', capture):
            plotting.plot_distributions(banks, summaries, Path('unused'))
        self.assertEqual({p.suffix for p in exports}, {'.svg', '.pdf', '.png', '.tiff'})
        spread = figures['accepted_spread'].axes
        np.testing.assert_allclose(spread[0].lines[1].get_ydata(), [10, 5])
        for axis in figures['accepted_correlations'].axes[:2]:
            self.assertEqual(axis.images[0].get_clim(), (-1, 1))
