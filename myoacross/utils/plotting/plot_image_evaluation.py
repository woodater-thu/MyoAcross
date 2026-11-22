"""Plot image2msksim results from existing experiment outputs."""
from pathlib import Path
import argparse
from myoacross.utils.paths import PROJECT_ROOT

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import csv, json, sys
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.patches import Rectangle, ConnectionPatch
    ROOT = args.data_root.parent
    OUT = args.output_dir
    DATA = args.data_root / 'experiments/image2msksim'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': '#26323B', 'axes.labelcolor': '#26323B', 'axes.edgecolor': '#26323B', 'xtick.color': '#26323B', 'ytick.color': '#26323B', 'axes.linewidth': 0.75, 'legend.frameon': False, 'savefig.facecolor': 'white'})
    a = json.load(open(DATA / 'evaluations/segment_film/test_iter15000/dataset_summary.json'))
    assert a['status'] == 'finished' and a['num_completed_subjects'] == 88 and (a['num_failed_subjects'] == 0)
    rows = []
    for subject in a['subjects']:
        m = subject['metrics']
        sid = subject['subject_id']
        provenance = json.load(open(DATA / 'subjects' / sid / 'reconstructions' / sid / 'source.json'))
        assert provenance['height_calibration'] is True and m['num_trials'] == 64
        rows.append(dict(subject_id=sid, source_split=subject['source_split'], num_reference_motions=m['num_trials'], completion_percent=100 * m['completed'], mean_tracking_error_mm=1000 * m['body_position_error_m']))
    with (OUT / 'figure7c_source_data.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    held = np.array([r['source_split'] == 'test' for r in rows])
    x = np.array([r['mean_tracking_error_mm'] for r in rows])
    y = np.array([r['completion_percent'] for r in rows])
    assert held.sum() == 22
    (W, H) = (150, 115)
    fig = plt.figure(figsize=(W / 72, H / 72))
    fig.text(0, 107 / H, 'c  Image-entry evaluation', fontsize=8, weight='bold')
    ax = fig.add_axes([31 / W, 24 / H, 113 / W, 70 / H])
    colors = {'train': '#4474A7', 'test': '#D3833D', 'unregistered': '#A0A5A8'}
    point_colors = [colors[r['source_split']] for r in rows]

    def points(axis):
        axis.scatter(x, y, s=12, c=point_colors, edgecolors='white', linewidths=0.25, zorder=3)
    points(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='none', markersize=3.5,
                      markerfacecolor=colors[split], markeredgecolor='none', label=label)
               for split, label in [('train', 'Training split (63)'),
                                    ('test', 'Held-out split (22)'),
                                    ('unregistered', 'Outside splits (3)')]]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.05, 0.04),
              fontsize=6, handletextpad=0.5, borderaxespad=0, labelspacing=0.5)
    ax.set(xlim=(48, 125), ylim=(0, 105), xticks=[50, 75, 100, 125], yticks=[0, 50, 100], xlabel='Tracking error (mm)', ylabel='Completion (%)')
    ax.tick_params(length=2.2, width=0.6, pad=2, labelsize=6.5)
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3
    ax.spines[['top', 'right']].set_visible(False)
    zoom = fig.add_axes([91 / W, 67 / H, 50 / W, 25 / H])
    points(zoom)
    zoom.set(xlim=(50, 66), ylim=(90, 101), xticks=[50, 60], yticks=[90, 100])
    zoom.tick_params(length=1.5, width=0.5, pad=1, labelsize=5.5)
    for spine in zoom.spines.values():
        spine.set_color('#969DA0')
        spine.set_linewidth(0.6)
    ax.add_patch(Rectangle((50, 90), 16, 11, fill=False, edgecolor='#969DA0', linewidth=0.6, linestyle=(0, (3, 2)), zorder=2))
    fig.add_artist(ConnectionPatch(xyA=(66, 90), coordsA=ax.transData, xyB=(0, 0.5), coordsB=zoom.transAxes, color='#969DA0', linewidth=0.55, linestyle=(0, (3, 2)), zorder=1))
    t = a['by_source_split']['test']
    fig.canvas.draw()
    fig.savefig(OUT / 'figure7c.pdf')
    fig.savefig(OUT / 'figure7c.svg')
    fig.savefig(OUT / 'figure7c.png', dpi=300)
    plt.close(fig)
if __name__ == '__main__':
    main()
