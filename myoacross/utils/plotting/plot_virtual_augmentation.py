"""Plot virtual embodiment count results from existing experiment outputs."""
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
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ROOT = args.data_root.parent
    OUT = args.output_dir
    SCHEMES = [('V0', 'experiments/virtual_embodiment_count/evaluations/virtual0_film/test_iter015000', 's'), ('V10', 'myoacross/rl/evaluations/test_iter015000', 'o')]
    METRICS = [('Completion (%)', 'completed', 100, (75, 101), [75, 87.5, 100]), ('Reference coverage (%)', 'coverage', 100, (92, 100.5), [92, 96, 100]), ('Tracking error (mm)', 'body_position_error_m', 1000, (45, 110), [50, 80, 110])]
    GREEN = '#788961'

    def number(v):
        return float(v == 'True') if v in ('True', 'False') else float(v)
    data = []
    ids_ref = None
    for (label, directory, marker) in SCHEMES:
        folder = args.data_root / directory
        with (folder / 'trials.csv').open() as f:
            rows = list(csv.DictReader(f))
        ids = sorted({r['subject_id'] for r in rows})
        assert len(ids) == 31 and len(rows) == 1984
        if ids_ref is None:
            ids_ref = ids
        assert ids == ids_ref
        values = {key: np.array([np.mean([number(r[key]) for r in rows if r['subject_id'] == subject]) * scale for subject in ids]) for (_, key, scale, _, _) in METRICS}
        assert all((sum((r['subject_id'] == s for r in rows)) == 64 for s in ids))
        data.append(values)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': '#26323B', 'axes.edgecolor': '#26323B', 'xtick.color': '#26323B', 'ytick.color': '#26323B', 'axes.linewidth': 0.75})
    fig = plt.figure(figsize=(510 / 72, 105 / 72), facecolor='white')
    fig.text(6 / 510, 93 / 105, 'b  Generalization to unseen individual models', fontsize=8, weight='bold')
    draws = np.random.default_rng(42).integers(0, 31, (10000, 31))
    jitter = np.random.default_rng(42).uniform(-0.13, 0.13, 31)
    for (i, (title, key, scale, limits, ticks)) in enumerate(METRICS):
        left = [35, 218, 375][i]
        width = [164, 143, 127][i]
        ax = fig.add_axes([left / 510, 19 / 105, width / 510, 64 / 105])
        for (row, (label, directory, marker)) in enumerate(SCHEMES):
            vals = data[row][key]
            mean = vals.mean()
            (lo, hi) = np.percentile(vals[draws].mean(1), [2.5, 97.5])
            outside = (vals < limits[0]) | (vals > limits[1])
            ax.scatter(vals[~outside], row + jitter[~outside], s=8, c=GREEN, alpha=0.28, edgecolors='none', zorder=2)
            for (value, y) in zip(vals[outside], row + jitter[outside]):
                low = value < limits[0]
                edge = limits[0] if low else limits[1]
                ax.scatter(edge, y, s=25, marker='<' if low else '>', c=GREEN, edgecolors='white', linewidths=0.4, zorder=5)
                ax.annotate(f'{value:.2f}', (edge, y), xytext=(5 if low else -5, 8), textcoords='offset points', ha='left' if low else 'right', va='bottom', color=GREEN, fontsize=6.5, zorder=6)
            ax.errorbar(mean, row, xerr=[[mean - lo], [hi - mean]], fmt=marker, color=GREEN, markersize=4, markeredgecolor='white', markeredgewidth=0.45, elinewidth=0.9, capsize=2, capthick=0.75, zorder=4)
            ax.annotate(f'{mean:.2f}', (mean, row), xytext=(0, 10), textcoords='offset points', ha='center', va='bottom', color=GREEN, fontsize=7)
        ax.set_xlim(*limits)
        ax.set_ylim(1.55, -0.55)
        ax.set_xticks(ticks)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([s[0] for s in SCHEMES] if i == 0 else [])
        ax.tick_params(axis='x', length=2, width=0.65, pad=2)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(i == 2)
        ax.spines['right'].set_color('#000000')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('#000000')
        ax.tick_params(axis='y', length=2 if i == 0 else 0, width=0.65)
        ax.grid(axis='x', color='#E4E6E7', linewidth=0.4, zorder=0)
        ax.set_xlabel(title, fontsize=7, labelpad=3)
    fig.savefig(OUT / 'figure5b.svg')
    fig.savefig(OUT / 'figure5b.pdf')
    fig.savefig(OUT / 'figure5b_preview.png', dpi=240)
if __name__ == '__main__':
    main()
