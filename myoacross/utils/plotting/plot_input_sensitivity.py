"""Plot input sensitivity results from existing experiment outputs."""
from pathlib import Path
import argparse
from myoacross.utils.paths import PROJECT_ROOT

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from collections import defaultdict
    import csv
    import json
    import sys
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    ROOT = args.data_root.parent
    OUT = args.output_dir
    DATA = args.data_root / 'experiments/input_sensitivity/evaluations'
    SCHEMES = [('Unconditioned MLP', 'segment_mlp', '#4474A7', 's', (0, (2, 2))), ('Concat', 'segment_concat', '#4474A7', '^', (0, (5, 2))), ('MyoAcross', 'segment_film', '#788961', 'D', '-')]
    TARGETS = [('b', 'morphology', 'Morphology feature perturbation'), ('c', 'reference', 'Reference-motion input perturbation'), ('d', 'combined', 'Combined perturbation')]
    METRICS = [('Completion (%)', 'completed', 100, (0, 103), [0, 50, 100]), ('Reference coverage (%)', 'coverage', 100, (0, 103), [0, 50, 100]), ('Tracking error (mm)', 'body_position_error_m', 1000, (40, 170), [50, 100, 150])]
    (WIDTH, HEIGHT) = (510, 145)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': '#26323B', 'axes.labelcolor': '#26323B', 'axes.edgecolor': '#26323B', 'xtick.color': '#26323B', 'ytick.color': '#26323B', 'axes.linewidth': 0.75, 'legend.frameon': False, 'savefig.facecolor': 'white'})
    results = {}
    source_rows = []
    summary_rows = []
    subject_ids = None
    for (letter, target, title) in TARGETS:
        for (label, scheme, color, marker, style) in SCHEMES:
            folder = DATA / scheme / 'test_iter15000' / target
            grouped = defaultdict(lambda : defaultdict(list))
            with (folder / 'trials.csv').open() as f:
                for row in csv.DictReader(f):
                    key = (float(row['morphology_magnitude']), float(row['reference_magnitude_deg']))
                    grouped[key][row['subject_id']].append(row)
            ids = sorted(next(iter(grouped.values())))
            assert len(ids) == 31
            if subject_ids is None:
                subject_ids = ids
            assert ids == subject_ids
            draws = np.random.default_rng(42).integers(0, 31, (10000, 31))
            with (folder / 'sensitivity.csv').open() as f:
                reported = {(float(r['morphology_magnitude']), float(r['reference_magnitude_deg'])): r for r in csv.DictReader(f)}
            assert len(grouped) == 4 and set(grouped) == set(reported)
            values = []
            for level in sorted(grouped):
                subjects = grouped[level]
                assert sorted(subjects) == ids
                assert all((len(subjects[s]) == (64 if level == (0.0, 0.0) else 192) for s in ids))
                data = np.array([[np.mean([float(r[k] == 'True') if k == 'completed' else float(r[k]) for r in subjects[s]]) * scale for (_, k, scale, _, _) in METRICS] for s in ids])
                means = data.mean(0)
                for (j, (_, key, scale, _, _)) in enumerate(METRICS):
                    assert np.isclose(means[j], float(reported[level][key]) * scale)
                (lo, hi) = np.percentile(data[draws].mean(1), [2.5, 97.5], axis=0)
                assert np.all(lo >= np.array([0, 0, 0])) and np.all(hi[:2] <= 100 + 1e-08)
                assert lo[2] >= 40 and hi[2] <= 170
                values.append((level, means, lo, hi))
                for (i, s) in enumerate(ids):
                    source_rows.append(dict(target=target, policy=label, subject_id=s, morphology_bias_percent=level[0] * 100, reference_offset_deg=level[1], completion_percent=data[i, 0], coverage_percent=data[i, 1], body_position_error_mm=data[i, 2]))
                for (j, (metric, key, scale, _, _)) in enumerate(METRICS):
                    summary_rows.append(dict(target=target, policy=label, morphology_bias_percent=level[0] * 100, reference_offset_deg=level[1], metric=metric, mean=means[j], ci_low=lo[j], ci_high=hi[j], num_subjects=31, motions_per_subject=64, sign_samples=1 if level == (0.0, 0.0) else 3))
            results[target, scheme] = values

    def draw_row(letter, target, title):
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
        grid = fig.add_gridspec(1, 3, left=42 / WIDTH, right=504 / WIDTH, bottom=43 / HEIGHT, top=102 / HEIGHT, wspace=39 / 128)
        axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
        fig.text(6 / WIDTH, 133 / HEIGHT, f'{letter}  {title}', fontsize=8, weight='bold', gid='row_title')
        handles = []
        for (j, (metric, key, scale, limits, ticks)) in enumerate(METRICS):
            ax = axes[j]
            for (label, scheme, color, marker, style) in SCHEMES:
                vals = results[target, scheme]
                levels = [v[0] for v in vals]
                x = [m * 100 for (m, r) in levels] if target == 'morphology' else [r for (m, r) in levels] if target == 'reference' else list(range(4))
                means = np.array([v[1][j] for v in vals])
                lo = np.array([v[2][j] for v in vals])
                hi = np.array([v[3][j] for v in vals])
                artist = ax.errorbar(x, means, yerr=[means - lo, hi - means], color=color, marker=marker, linestyle=style, markersize=3.4, markeredgecolor='white', markeredgewidth=0.35, linewidth=1.15 if scheme == 'segment_film' else 0.85, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4 if scheme == 'segment_film' else 3, label=label)
                if j == 0:
                    handles.append(artist)
            if target == 'morphology':
                (limits, ticks) = [((80, 101), [80, 85, 90, 95, 100]), ((90, 100.5), [90, 95, 100]), ((50, 90), [50, 60, 70, 80, 90])][j]
            else:
                (limits, ticks) = [((10, 101), [10, 25, 50, 75, 100]), ((50, 101), [50, 75, 100]), ((50, 150), [50, 75, 100, 125, 150])][j]
            for (_, scheme, _, _, _) in SCHEMES:
                assert all((v[2][j] >= limits[0] and v[3][j] <= limits[1] for v in results[target, scheme]))
            ax.set_ylim(limits)
            ax.set_yticks(ticks)
            ax.set_ylabel(metric, fontsize=7, labelpad=4)
            ax.tick_params(length=2, width=0.65, pad=2, labelsize=7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#E4E6E7', linewidth=0.4, zorder=0)
            if target == 'combined':
                ax.set_xlim(-0.13, 3.13)
                ax.set_xticks(range(4), ['0%\n0°', '2%\n2.5°', '5%\n5°', '10%\n10°'])
                ax.set_xlabel('Paired input bias', labelpad=5, fontsize=7)
            else:
                ax.set_xlim(-0.4, 10.4)
                ax.set_xticks([0, 2, 5, 10] if target == 'morphology' else [0, 2.5, 5, 10], ['0', '2', '5', '10'] if target == 'morphology' else ['0', '2.5', '5', '10'])
                ax.set_xlabel('Morphology bias (%)' if target == 'morphology' else 'Reference offset (°)', labelpad=5, fontsize=7)
        fig.legend(handles, [s[0] for s in SCHEMES], loc='center', bbox_to_anchor=(0.5, 119 / HEIGHT), ncol=3, fontsize=7, handlelength=2.2, columnspacing=2, handletextpad=0.6).set_gid('row_legend')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            for item in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
                box = item.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0 and (box.x1 <= fig.bbox.width) and (box.y1 <= fig.bbox.height)
        fig.savefig(OUT / f'figure6{letter}.svg')
        fig.savefig(OUT / f'figure6{letter}.pdf')
        fig.savefig(OUT / f'figure6{letter}_preview.png', dpi=300)
        plt.close(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    for (letter, target, title) in TARGETS:
        draw_row(letter, target, title)
    for (name, rows) in [('figure6_source_data.csv', source_rows), ('figure6_summary.csv', summary_rows)]:
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
    (OUT / 'figure6_protocol.json').write_text(json.dumps(dict(claim='Sensitivity of policy reuse to morphology and reference input errors', data_root='data/experiments/input_sensitivity/evaluations', checkpoint_iteration=15000, models=31, motions_per_model=64, sign_samples=3, bootstrap_resamples=10000, bootstrap_seed=42, intervals='95% percentile intervals resampling individual models', scoring='Clean reference; errors over executed frames including incomplete trials', aggregation='Average sign samples and motions within individuals; equally weight individuals', zero_level='Each target-specific evaluation uses its own clean baseline', widths_pt=WIDTH, heights_pt=HEIGHT, font='DejaVu Sans', body_font_pt=7, title_font_pt=8), indent=2) + '\n')
if __name__ == '__main__':
    main()
