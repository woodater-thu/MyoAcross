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
    import csv, html
    import numpy as np
    R = args.output_dir
    DATA = args.data_root
    paths = ['experiments/policy_comparison/evaluations/segment_mlp/test_iter015000', 'experiments/policy_comparison/evaluations/segment_concat/test_iter015000', 'myoacross/rl/evaluations/test_iter015000']
    rows = []
    for path in paths:
        with (DATA / path / 'subjects.csv').open() as f:
            rows.append(sorted(csv.DictReader(f), key=lambda r: r['subject_id']))
    assert all((len(a) == 31 and all((int(r['num_trials']) == 64 for r in a)) for a in rows))
    assert all(([r['subject_id'] for r in a] == [r['subject_id'] for r in rows[0]] for a in rows))
    values = [np.array([float(r['body_position_error_m']) * 1000 for r in a]) for a in rows]
    INK = '#26323B'
    BLUE = '#4474A7'
    GREEN = '#788961'
    S = ['<svg xmlns="http://www.w3.org/2000/svg" width="330pt" height="178pt" viewBox="0 0 330 178">', '<rect width="330" height="178" fill="white"/>']

    def text(x, y, t, size=7, bold=False, anchor='middle', color=INK, rotate=None):
        rot = f' transform="rotate({rotate} {x} {y})"' if rotate else ''
        S.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{(700 if bold else 400)}" text-anchor="{anchor}" fill="{color}"{rot}>{html.escape(t)}</text>')

    def line(x1, y1, x2, y2, color=INK, width=0.75, dashed=False):
        dash = ' stroke-dasharray="3 2"' if dashed else ''
        S.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}/>')
    text(6, 12, 'c  Accuracy across individuals', 8, True, 'start')
    for (k, title) in enumerate(['Unconditioned MLP', 'Concat']):
        left = [30, 205][k]
        right = left + 110
        top = 35
        bottom = 145
        x = values[k]
        y = values[2]
        assert np.all((x >= 40) & (x <= 100) & (y >= 40) & (y <= 100))
        px = lambda v: left + 110 * (v - 40) / 60
        py = lambda v: bottom - 110 * (v - 40) / 60
        text((left + right) / 2, 27, title, color=BLUE)
        for tick in [40, 60, 80, 100]:
            line(px(tick), bottom, px(tick), bottom + 3)
            text(px(tick), 157, str(tick))
            line(left - 3, py(tick), left, py(tick))
            text(left - 6, py(tick) + 2.4, str(tick), anchor='end')
        line(left, top, left, bottom)
        line(left, bottom, right, bottom)
        line(left, bottom, right, top, '#A0A5A8', 0.75, True)
        for (vx, vy) in zip(x, y):
            S.append(f'<circle cx="{px(vx)}" cy="{py(vy)}" r="1.6" fill="{GREEN}" fill-opacity="0.7" stroke="{GREEN}" stroke-width="0.4"/>')
        n = int(np.sum(y < x))
        text(left + 5, 47, f'Lower error: {n}/31', anchor='start', color='#52663C')
    text(165, 174, 'Comparator body-position error (mm)')
    text(9, 90, 'MyoAcross body-position error (mm)', rotate=-90)
    S.append('</svg>')
    (R / 'figure3c.svg').write_text('\n'.join(S))
if __name__ == '__main__':
    main()
