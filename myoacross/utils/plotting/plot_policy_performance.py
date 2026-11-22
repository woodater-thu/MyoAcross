"""Plot policy comparison results from existing experiment outputs."""
from pathlib import Path
import argparse
from myoacross.utils.paths import PROJECT_ROOT

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import csv, json, html
    import numpy as np
    R = args.output_dir
    DATA = args.data_root
    SCHEMES = [('Template MLP', 'experiments/policy_comparison/evaluations/myofullbody_mlp/test_iter015000'), ('Unconditioned MLP', 'experiments/policy_comparison/evaluations/segment_mlp/test_iter015000'), ('Concat', 'experiments/policy_comparison/evaluations/segment_concat/test_iter015000'), ('MyoAcross', 'myoacross/rl/evaluations/test_iter015000')]
    METRICS = [('completed', 100, 'Completion (%)', 100), ('coverage', 100, 'Reference coverage (%)', 100), ('body_position_error_m', 1000, 'Tracking error (mm)', 180)]
    INK = '#26323B'
    BLUE = '#4474A7'
    GREEN = '#788961'
    S = ['<svg xmlns="http://www.w3.org/2000/svg" width="510pt" height="154pt" viewBox="0 0 510 154">', '<rect width="510" height="154" fill="white"/>']

    def text(x, y, t, size=7, bold=False, anchor='middle', color=INK):
        S.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{(700 if bold else 400)}" text-anchor="{anchor}" fill="{color}">{html.escape(t)}</text>')

    def line(x1, y1, x2, y2, color=INK, width=0.75):
        S.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
    text(6, 12, 'b  Policy performance', 8, True, 'start')
    rows = []
    for (name, path) in SCHEMES:
        p = DATA / path
        summary = json.loads((p / 'summary.json').read_text())
        assert summary['num_subjects'] == 31 and summary['num_trials'] == 1984
        with (p / 'subjects.csv').open() as f:
            records = sorted(csv.DictReader(f), key=lambda r: r['subject_id'])
        assert len(records) == 31 and all((int(r['num_trials']) == 64 for r in records))
        rows.append((name, summary, records))
    assert all(([r['subject_id'] for r in row[2]] == [r['subject_id'] for r in rows[0][2]] for row in rows))
    draws = np.random.default_rng(42).integers(0, 31, (10000, 31))
    jitter = np.random.default_rng(7).uniform(-3.5, 3.5, 31)
    for (j, (name, _, _)) in enumerate(rows):
        text(6, 52 + 21 * j + 2.4, name, bold=True, anchor='start', color='#52663C' if j == 3 else INK)

    def axis(left, right, low, high, ticks):
        for tick in ticks:
            x = left + (right - left) * (tick - low) / (high - low)
            line(x, 42, x, 129, '#E5E8E9', 0.4)
            line(x, 129, x, 132)
            text(x, 143, str(tick), anchor='start' if tick == low else 'end' if tick == high else 'middle')
        line(left, 129, right, 129)
    line(74, 42, 74, 129, '#000000')
    for j in range(4):
        line(72, 52 + 21 * j, 74, 52 + 21 * j, '#000000')

    def estimate(left, right, low, high, y, mean, ci, color, annotate=True):
        x = left + (right - left) * (mean - low) / (high - low)
        (a, b) = left + (right - left) * (ci - low) / (high - low)
        assert low <= ci[0] <= ci[1] <= high
        line(a, y, b, y, color, 1)
        line(a, y - 2, a, y + 2, color)
        line(b, y - 2, b, y + 2, color)
        S.append(f'<polygon points="{x},{y - 2.5} {x + 2.5},{y} {x},{y + 2.5} {x - 2.5},{y}" fill="{color}" stroke="white" stroke-width="0.5"/>')
        if annotate:
            text(x, y - 7, f'{mean:.2f}', anchor='end' if x > right - 12 else 'start' if x < left + 12 else 'middle', color=color)
    for (k, (metric, factor, title, maxval)) in enumerate(METRICS):
        left = [74, 228, 374][k]
        right = [208, 362, 502][k]
        text((left + right) / 2, 23, title)
        if k < 2:
            overview_right = left + 44
            detail_left = left + 55
            low = 95 if k == 0 else 98
            text((left + overview_right) / 2, 36, 'Overview')
            text((detail_left + right) / 2, 36, 'Detail')
            axis(left, overview_right, 0, 100, [0, 100])
            axis(detail_left, right, low, 100, [95, 97.5, 100] if k == 0 else [98, 99, 100])
        else:
            overview_right = right
            text((left + right) / 2, 36, 'All models')
            axis(left, right, 0, maxval, [0, 60, 120, 180])
        for (j, (name, summary, records)) in enumerate(rows):
            color = GREEN if j == 3 else BLUE
            y = 52 + 21 * j
            vals = np.array([float(r[metric]) * factor for r in records])
            mean = vals.mean()
            assert np.isclose(mean, summary[metric] * factor)
            ci = np.quantile(vals[draws].mean(axis=1), [0.025, 0.975])
            for (v, dy) in zip(vals, jitter):
                x = left + (overview_right - left) * v / maxval
                S.append(f'<circle cx="{x}" cy="{y + dy}" r="1.15" fill="{color}" fill-opacity="0.3"/>')
            estimate(left, overview_right, 0, maxval, y, mean, ci, color, annotate=k == 2 or j == 0)
            if k < 2:
                if j == 0:
                    text((detail_left + right) / 2, y + 2.4, '—', color='#A0A5A8')
                else:
                    for (v, dy) in zip(vals, jitter):
                        if low <= v <= 100:
                            x = detail_left + (right - detail_left) * (v - low) / (100 - low)
                            S.append(f'<circle cx="{x}" cy="{y + dy}" r="1.15" fill="{color}" fill-opacity="0.3"/>')
                    estimate(detail_left, right, low, 100, y, mean, ci, color)
    S.append('</svg>')
    (R / 'figure3b.svg').write_text('\n'.join(S))
if __name__ == '__main__':
    main()
