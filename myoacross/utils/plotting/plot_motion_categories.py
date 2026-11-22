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
    import csv
    from collections import defaultdict
    from html import escape
    import numpy as np
    ROOT = args.data_root.parent
    OUT = args.output_dir / 'motion_categories.svg'
    SCHEMES = [('Unconditioned MLP', 'experiments/policy_comparison/evaluations/segment_mlp/test_iter015000', 'square'), ('Concat', 'experiments/policy_comparison/evaluations/segment_concat/test_iter015000', 'triangle'), ('MyoAcross', 'myoacross/rl/evaluations/test_iter015000', 'diamond')]
    CATEGORIES = [('walk_slow', 'Slow walking'), ('walk_medium', 'Medium walking'), ('walk_fast', 'Fast walking'), ('walk_backward', 'Backward walking'), ('turn_left', 'Left turn'), ('turn_right', 'Right turn'), ('circle_cw', 'Clockwise circling'), ('circle_ccw', 'Counterclockwise circling')]
    (BLUE, GREEN, TEXT) = ('#4474A7', '#788961', '#26323B')
    summaries = []
    ids_ref = None
    for (name, directory, shape) in SCHEMES:
        folder = args.data_root / directory
        with (folder / 'trials.csv').open() as f:
            rows = list(csv.DictReader(f))
        groups = defaultdict(list)
        for row in rows:
            groups[row['category'], row['subject_id']].append(float(row['body_position_error_m']) * 1000)
        ids = sorted({r['subject_id'] for r in rows})
        assert len(ids) == 31 and len(rows) == 1984
        if ids_ref is None:
            ids_ref = ids
        assert ids == ids_ref
        with (folder / 'categories.csv').open() as f:
            reported = {r['category']: float(r['body_position_error_m']) * 1000 for r in csv.DictReader(f)}
        rng = np.random.default_rng(42)
        draws = rng.integers(0, 31, (10000, 31))
        values = []
        for (key, label) in CATEGORIES:
            assert all((len(groups[key, subject]) == 8 for subject in ids))
            per_subject = np.array([np.mean(groups[key, subject]) for subject in ids])
            mean = per_subject.mean()
            assert np.isclose(mean, reported[key])
            (lo, hi) = np.percentile(per_subject[draws].mean(axis=1), [2.5, 97.5])
            values.append((mean, lo, hi))
        summaries.append(values)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="210.8633pt" height="158.1367pt" viewBox="0 0 210.8633 158.1367">', '<rect width="210.8633" height="158.1367" fill="white"/>']

    def text(x, y, value, size=7, color=TEXT, anchor='start', bold=False):
        weight = 'bold' if bold else 'normal'
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')

    def line(x1, y1, x2, y2, color=TEXT, width=0.75):
        svg.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"/>')

    def marker(x, y, shape, color, r=1.7):
        if shape == 'circle':
            svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r}" fill="{color}" stroke="white" stroke-width=".35"/>')
        elif shape == 'square':
            svg.append(f'<rect x="{x - r:.3f}" y="{y - r:.3f}" width="{2 * r}" height="{2 * r}" fill="{color}" stroke="white" stroke-width=".35"/>')
        else:
            pts = [(x, y - r - 0.3), (x + r + 0.3, y + r), (x - r - 0.3, y + r)] if shape == 'triangle' else [(x, y - r - 0.3), (x + r + 0.3, y), (x, y + r + 0.3), (x - r - 0.3, y)]
            points = ' '.join((f'{a:.3f},{b:.3f}' for (a, b) in pts))
            svg.append(f'<polygon points="{points}" fill="{color}" stroke="white" stroke-width=".35"/>')
    text(6, 10, 'd  Accuracy across categories', 8, bold=True)
    for (i, (name, _, shape)) in enumerate(SCHEMES):
        x = (61, 143, 183)[i]
        y = 22
        color = GREEN if i == 2 else BLUE
        marker(x, y - 2, shape, color)
        text(x + 5, y, name, color=color)
    axes = [(66, 208, 35, 85, [40, 60, 80])]
    for (left, right, vmin, vmax, ticks) in axes:
        line(left, 140, right, 140)
        for tick in ticks:
            x = left + (tick - vmin) / (vmax - vmin) * (right - left)
            line(x, 30, x, 140, '#E4E6E7', 0.4)
            line(x, 140, x, 143)
            text(x, 150, str(tick), anchor='middle')
    connections = []
    for (i, values) in enumerate(summaries):
        color = GREEN if i == 2 else BLUE
        dash = ('1.2 1.8', '3 2', '')[i]
        pts = [(66 + (v[0] - 35) / 50 * 142, 37 + row * 13 + (-2, 0, 2)[i]) for (row, v) in enumerate(values)]
        for ((x1, y1), (x2, y2)) in zip(pts, pts[1:]):
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            gap = 2.2 / length
            (a, b) = (x1 + (x2 - x1) * gap, y1 + (y2 - y1) * gap)
            (c, d) = (x2 - (x2 - x1) * gap, y2 - (y2 - y1) * gap)
            connections.append(f'<line x1="{a:.4f}" y1="{b:.4f}" x2="{c:.4f}" y2="{d:.4f}" stroke="{color}" stroke-width="0.65" stroke-dasharray="{dash}" fill="none"/>')
    svg.extend(connections)
    overlay = ['<svg xmlns="http://www.w3.org/2000/svg" width="510pt" height="430.195pt" viewBox="0 0 510 430.195">', '<rect x="0" y="0" width="510" height="430.195" fill="none" stroke="none"/>', '<g transform="translate(294.977 266.137)">', *connections, '</g></svg>']
    OUT.with_name('figure3d_connections_overlay.svg').write_text('\n'.join(overlay))
    for (row, (key, label)) in enumerate(CATEGORIES):
        y = 37 + row * 13
        if key == 'circle_ccw':
            text(61, y - 1, 'Counterclockwise', anchor='end')
            text(61, y + 6, 'circling', anchor='end')
        else:
            text(61, y + 2, label, anchor='end')
        for (i, (_, _, shape)) in enumerate(SCHEMES):
            (mean, lo, hi) = summaries[i][row]
            color = GREEN if i == 2 else BLUE
            yy = y + (-2, 0, 2)[i]
            for (axis_idx, (left, right, vmin, vmax, _)) in enumerate(axes):
                if not vmin <= mean <= vmax:
                    continue

                def xp(v):
                    return left + (v - vmin) / (vmax - vmin) * (right - left)
                line(xp(max(lo, vmin)), yy, xp(min(hi, vmax)), yy, color, 0.65)
                for v in (lo, hi):
                    if vmin <= v <= vmax:
                        line(xp(v), yy - 1, xp(v), yy + 1, color, 0.65)
                marker(xp(mean), yy, shape, color, 1.35)
    text(133, 157, 'Tracking error (mm)', anchor='middle')
    svg.append('</svg>')
    OUT.write_text('\n'.join(svg))
if __name__ == '__main__':
    main()
