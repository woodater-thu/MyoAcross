"""Plot asymmetry generalization results from existing experiment outputs."""
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
    from collections import defaultdict
    ROOT = args.data_root.parent
    OUT = args.output_dir
    BASE = args.data_root / 'experiments/asymmetry_generalization/evaluations'
    SCHEMES = [('Standard', 'segment_film', '#788961', 'o'), ('Asymmetry augmentation', 'asymmetry_enhanced_film', '#788961', '^')]
    LEVELS = [0, 1, 2.5, 5, 10]
    CONDITIONS = ['original', '{side}_1pct', '{side}_2p5pct', '{side}_5pct', '{side}_10pct']
    METRICS = [('Completion (%)', 'completed', 100, (0, 103), [0, 50, 100]), ('Reference coverage (%)', 'coverage', 100, (0, 103), [0, 50, 100]), ('Tracking error (mm)', 'body_position_error_m', 1000, (40, 170), [50, 100, 150])]

    def value(text):
        return float(text == 'True') if text in ('True', 'False') else float(text)
    summaries = []
    for (label, scheme, color, marker) in SCHEMES:
        folder = BASE / scheme / 'physical_test_iter015000'
        with (folder / 'trials.csv').open() as f:
            rows = list(csv.DictReader(f))
        with (folder / 'body_conditions.csv').open() as f:
            reported = {r['body_condition']: r for r in csv.DictReader(f)}
        ids = sorted({r['subject_id'] for r in rows})
        assert len(ids) == 31 and len(rows) == 31 * 64 * 9
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['body_condition'], row['subject_id']].append(row)
        draws = np.random.default_rng(42).integers(0, 31, (10000, 31))
        result = {}
        for condition in reported:
            assert all((len(grouped[condition, subject]) == 64 for subject in ids))
            for (title, key, scale, limits, ticks) in METRICS:
                per_subject = np.array([np.mean([value(r[key]) for r in grouped[condition, s]]) * scale for s in ids])
                mean = per_subject.mean()
                assert np.isclose(mean, float(reported[condition][key]) * scale)
                (lo, hi) = np.percentile(per_subject[draws].mean(1), [2.5, 97.5])
                result[condition, key] = (mean, lo, hi)
        summaries.append(result)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': '#26323B', 'axes.labelcolor': '#26323B', 'axes.edgecolor': '#26323B', 'xtick.color': '#26323B', 'ytick.color': '#26323B', 'axes.linewidth': 0.75})
    fig = plt.figure(figsize=(510 / 72, 155 / 72), facecolor='white')
    fig.text(6 / 510, 146 / 155, 'c  Generalization to asymmetric models', fontsize=8, weight='bold')
    handles = []
    for (metric_index, (title, key, scale, limits, ticks)) in enumerate(METRICS):
        centre = [106, 274, 440][metric_index]
        pair_width = [122, 122, 120][metric_index]
        group_start = centre - pair_width / 2
        axis_width = (pair_width - 12) / 2
        for (side_index, side) in enumerate(['left', 'right']):
            left = group_start + side_index * (axis_width + 12)
            ax = fig.add_axes([left / 510, 48 / 155, axis_width / 510, 70 / 155])
            for (scheme_index, (label, scheme, color, marker)) in enumerate(SCHEMES):
                values = np.array([summaries[scheme_index][c.format(side=side), key] for c in CONDITIONS])
                (means, lo, hi) = values.T
                artist = ax.errorbar(LEVELS, means, yerr=np.array([means - lo, hi - means]), color=color, marker=marker, markersize=3.0, markeredgewidth=0.35, linestyle=(0, (5, 2.5)) if scheme_index == 0 else '-', markeredgecolor='white', linewidth=0.9, elinewidth=0.55, capsize=1.5, capthick=0.55, label=label, zorder=3 + scheme_index)
                if metric_index == 0 and side_index == 0:
                    handles.append(artist)
            ax.set_xlim(-0.45, 10.45)
            ax.set_ylim(*limits)
            ax.set_xticks([0, 2.5, 5, 10], ['0', '2.5', '5', '10'])
            ax.set_yticks(ticks)
            ax.tick_params(length=2, width=0.65, pad=2, labelsize=7)
            if side_index == 1:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(title, fontsize=7, labelpad=4)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.grid(axis='y', color='#E4E6E7', linewidth=0.4, zorder=0)
            ax.set_title('Left' if side == 'left' else 'Right', fontsize=7, pad=3)
        fig.text(centre / 510, 30 / 155, 'Unilateral shortening (%)', ha='center', fontsize=7)
    fig.legend(handles, [x[0] for x in SCHEMES], loc='center', bbox_to_anchor=(0.5, 139 / 155), ncol=2, frameon=False, fontsize=7, handlelength=2, handletextpad=0.6, columnspacing=2)
    fig.savefig(OUT / 'figure5c.svg')
    fig.savefig(OUT / 'figure5c.pdf')
    fig.savefig(OUT / 'figure5c_preview.png', dpi=240)
if __name__ == '__main__':
    main()
