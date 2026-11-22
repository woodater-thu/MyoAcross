"""Compare accepted morphology distributions across generation schemes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from myoacross.utils.config import load_config
from myoacross.utils.paths import EMBODIMENTS_DIR, PROJECT_ROOT, resolve_path


DEFAULT_CONFIGS = [
    PROJECT_ROOT / 'myoacross/configs/rl/generate.yaml',
    PROJECT_ROOT / 'myoacross/configs/experiments/asymmetry_generalization/generate_asymmetry_enhanced_train.yaml',
]


def load_bank(bank: Path) -> dict:
    """Read stored virtual parameters and pair measurements with original bodies."""
    sampling = json.loads((bank / 'virtual/bank.json').read_text())['sampling']
    parameter_names = sampling['parameter_names']
    rows, originals = [], {}
    feature_names = None
    for path in sorted((bank / 'virtual').glob('*/index.json')):
        index = json.loads(path.read_text())
        if index['sampling'] != sampling:
            raise ValueError(f'Inconsistent sampling settings: {path}')
        sid = index['source_subject_id']
        source = json.loads((bank / 'subject' / sid / 'index.json').read_text())['samples'][0]
        names = source['features']['names']
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f'Incompatible feature order: {path}')
        original = np.asarray(source['features']['values'])
        originals[sid] = original
        report = json.loads((EMBODIMENTS_DIR / 'segment_scale' / sid / 'report.json').read_text())
        center = np.array([report['scales']['regions'][name] for name in parameter_names[:-1]]
                          + [report['mass']['actual_kg']])
        for sample in index['samples']:
            with np.load(path.parent / sample['bundle'], allow_pickle=False) as archive:
                parameters = json.loads(archive['metadata'].item())['parameters']
            if parameters['region_names'] != parameter_names[:-1]:
                raise ValueError(f'Incompatible parameter order: {path}')
            if sample['features']['names'] != feature_names:
                raise ValueError(f'Incompatible sample feature order: {path}')
            values = np.asarray(parameters['region_scales'] + [parameters['body_mass_kg']])
            rows.append(dict(subject=sid, variant=sample['variant_id'],
                             bundle=str(path.parent / sample['bundle']),
                             parameter_changes=100 * (values / center - 1),
                             feature_changes=100 * (np.asarray(sample['features']['values']) / original - 1)))
    if len(rows) < 2:
        raise ValueError(f'Need at least two accepted samples: {bank}')
    return dict(bank=str(bank.resolve()), sampling=sampling, parameter_names=parameter_names,
                feature_names=feature_names, subject_ids=sorted(originals), originals=originals,
                rows=rows, parameter_changes=np.array([r['parameter_changes'] for r in rows]),
                feature_changes=np.array([r['feature_changes'] for r in rows]))


def summarize_bank(data: dict) -> dict:
    """Describe observed distributions; report prior and accepted spread separately."""
    def statistics(values):
        return dict(mean_percent=values.mean(axis=0).tolist(),
                    std_percent=values.std(axis=0, ddof=1).tolist(),
                    quantiles_percent={str(q): np.quantile(values, q, axis=0).tolist()
                                       for q in (0, 0.025, 0.5, 0.975, 1)})

    covariance = np.asarray(data['sampling']['covariance'])
    std = np.sqrt(np.diag(covariance))
    prior = np.divide(covariance, np.outer(std, std),
                      out=np.full_like(covariance, np.nan), where=np.outer(std, std) > 0)
    empirical_covariance = np.cov(data['parameter_changes'], rowvar=False)
    observed_std = np.sqrt(np.diag(empirical_covariance))
    empirical = np.divide(empirical_covariance, np.outer(observed_std, observed_std),
                          out=np.full_like(empirical_covariance, np.nan),
                          where=np.outer(observed_std, observed_std) > 0)
    # A constant parameter has no defined correlation. Record it as JSON null.
    matrices = {name: [[float(value) if np.isfinite(value) else None for value in row]
                       for row in matrix]
                for name, matrix in (('prior_correlation', prior), ('empirical_correlation', empirical))}
    return dict(bank=data['bank'], subjects=len(data['subject_ids']), samples=len(data['rows']),
                subject_ids=data['subject_ids'], parameter_names=data['parameter_names'],
                feature_names=data['feature_names'], sampling=data['sampling'],
                parameters=statistics(data['parameter_changes']),
                features=statistics(data['feature_changes']), prior_std_percent=(100 * std).tolist(),
                **matrices)






def select_extreme_samples(banks: dict) -> dict:
    """Select the largest RMS geometry change in each accepted bank."""
    selected = {}
    for label, bank in banks.items():
        geometry = [i for i, name in enumerate(bank['feature_names']) if name != 'body_mass']
        changes = np.array([row['feature_changes'] for row in bank['rows']])
        scores = np.sqrt(np.mean(changes[:, geometry] ** 2, axis=1))
        index = int(np.argmax(scores))
        row = bank['rows'][index]
        selected[label] = dict(subject=row['subject'], variant=row['variant'],
                               row_index=index, geometry_rms_percent=float(scores[index]))
    return selected




def main(argv: list[str] | None = None):
    from myoacross.utils.plotting.plot_model_banks import plot_distributions, plot_extreme_bodies
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', help='Generation YAML; repeat for each bank. Default: normal reference and asymmetry enhanced.')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/experiments/asymmetry_generalization/evaluations/sampling')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Use a new analysis output directory: {output}')
    banks, configs = {}, {}
    for path in args.config or DEFAULT_CONFIGS:
        config = load_config(path)
        data = load_bank(resolve_path(config['data']['bundle_root'], PROJECT_ROOT))
        label = 'normal' if path.stem == 'generate' else path.stem.removeprefix('generate_').removesuffix('_train')
        if label in banks:
            raise ValueError(f'Duplicate generation scheme: {label}')
        banks[label], configs[label] = data, str(path.resolve())
    first = next(iter(banks.values()))
    identities = {(r['subject'], r['variant']) for r in first['rows']}
    for label, data in banks.items():
        if (data['parameter_names'] != first['parameter_names'] or data['feature_names'] != first['feature_names']
                or {(r['subject'], r['variant']) for r in data['rows']} != identities):
            raise ValueError(f'Comparison requires matching subjects, variants, and feature order: {label}')
        if any(not np.array_equal(data['originals'][sid], first['originals'][sid]) for sid in first['subject_ids']):
            raise ValueError(f'Comparison requires matching original bodies: {label}')
    summaries = {label: summarize_bank(data) for label, data in banks.items()}
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(dict(
        generation_configs=configs, schemes=summaries,
        distribution='Accepted virtual bodies only; original bodies define paired changes.',
        aggregation='Pooled accepted samples; observations remain grouped by source subject; descriptive statistics.',
    ), indent=2, allow_nan=False) + '\n')
    with (output / 'samples.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scheme', 'subject', 'variant']
                        + ['parameter_delta_pct_' + n for n in first['parameter_names']]
                        + ['feature_delta_pct_' + n for n in first['feature_names']])
        for label, data in banks.items():
            for row in data['rows']:
                writer.writerow([label, row['subject'], row['variant'], *row['parameter_changes'], *row['feature_changes']])
    plot_distributions(banks, summaries, output)
    plot_extreme_bodies(banks, output)
    print(output)


if __name__ == '__main__':
    main()
