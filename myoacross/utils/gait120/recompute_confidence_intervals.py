"""Recompute 95% subject-bootstrap CIs from saved Gait120 subject summaries.

Preserves subject order, seed 42, covariate refitting and all point estimates.
Outputs current final statistics without modifying the input data directory.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(x, y, z, rng, count, slope=False):
    correlations, slopes = [], []
    for _ in range(count):
        ix = rng.integers(len(x), size=len(x))
        design = np.column_stack([np.ones(len(x)), z[ix]])
        xr = x[ix] - design @ np.linalg.lstsq(design, x[ix], rcond=None)[0]
        yr = y[ix] - design @ np.linalg.lstsq(design, y[ix], rcond=None)[0]
        correlations.append(np.corrcoef(xr, yr)[0, 1])
        if slope:
            full = np.column_stack([np.ones(len(x)), x[ix], z[ix]])
            slopes.append(np.linalg.lstsq(full, y[ix], rcond=None)[0][1] * 10)
    return np.quantile(correlations, [.025, .975]), (np.quantile(slopes, [.025, .975]) if slope else None)


def compute(root, body_count, muscle_count):
    body = read(root/'body_mass/subjects.csv')
    body_stats = read(root/'body_mass/correlations.csv')
    rng = np.random.default_rng(42)
    for condition in ['height_scale', 'regional_scale']:
        records = [r for r in body if r['condition'] == condition]
        x = np.array([float(r['mass']) for r in records])
        y = np.array([float(r['activation_sum11']) for r in records])
        z = np.array([[float(r['height']), float(r['speed'])] for r in records])
        assert len(x) == 120
        for adjustment, controls in [('raw', np.empty((len(x), 0))), ('height_speed', z)]:
            ci, _ = interval(x, y, controls, rng, body_count)
            row = next(r for r in body_stats if r['condition']==condition and r['adjustment']==adjustment)
            row.update(ci_low=ci[0], ci_high=ci[1])
    muscle = read(root/'per_muscle/subjects.csv')
    muscle_stats = read(root/'per_muscle/muscle_statistics.csv')
    rng = np.random.default_rng(42)
    for condition, name in sorted({(r['condition'], r['muscle']) for r in muscle}):
        records = [r for r in muscle if r['condition']==condition and r['muscle']==name]
        x = np.array([float(r['mass']) for r in records])
        y = np.array([float(r['activation']) for r in records])
        z = np.array([[float(r['height']), float(r['speed'])] for r in records])
        assert len(x) == 120
        ci, slope_ci = interval(x, y, z, rng, muscle_count, slope=True)
        row = next(r for r in muscle_stats if r['condition']==condition and r['muscle']==name)
        row.update(ci_low=ci[0], ci_high=ci[1], slope_ci_low=slope_ci[0], slope_ci_high=slope_ci[1])
    return body_stats, muscle_stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--verify-previous', action='store_true')
    args=p.parse_args()
    if args.verify_previous:
        previous=compute(args.input_root, 5000, 2000)
        for recalculated, filename in zip(previous, ['body_mass/correlations.csv', 'per_muscle/muscle_statistics.csv']):
            for new, old in zip(recalculated, read(args.input_root/filename)):
                for key in ['ci_low','ci_high','slope_ci_low','slope_ci_high']:
                    if key in old and old[key]:
                        assert np.isclose(float(new[key]), float(old[key]), rtol=1e-10, atol=1e-12), (filename, key, new[key], old[key])
        print('Verified previous 5000/2000-resample intervals against saved results.', flush=True)
    body, muscle=compute(args.input_root,10000,10000)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for rows, name in [(body,'body_mass_correlations.csv'), (muscle,'muscle_statistics.csv')]:
        for r in rows:
            r['bootstrap_resamples']=10000
            r['confidence_level']=.95
            r['bootstrap_seed']=42
        write(args.output_dir/name,rows)
    print('Saved 10000-resample final statistics.', flush=True)

if __name__=='__main__':
    main()
