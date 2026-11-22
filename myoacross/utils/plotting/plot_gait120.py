"""Plot gait120 results from existing experiment outputs."""
from pathlib import Path
import argparse
from myoacross.utils.paths import PROJECT_ROOT

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--main-only', action='store_true')
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import json
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    ROOT = args.output_dir
    EXPERIMENT = args.data_root / 'experiments/gait120'
    OUT = args.output_dir
    for p in (OUT, args.output_dir / 'qa', ROOT / 'source_data'):
        p.mkdir(parents=True, exist_ok=True)
    (BLUE, GREEN, GRAY, INK) = ('#4474A7', '#788961', '#8B9297', '#26323B')
    POLICIES = ['film', 'concat', 'mlp', 'v0', 'asymmetry_augmentation']
    LABELS = ['MyoAcross', 'Concat', 'Unconditioned MLP', 'MyoAcross V0', 'MyoAcross, asymmetry augmentation']
    COLORS = [GREEN, BLUE, BLUE, GRAY, '#AAB79B']
    SHORT = ['MyoAcross', 'Concat', 'MLP', 'V0', 'Asym.']
    CHANNELS = [('vaslat_r', 'Vastus lateralis'), ('recfem_r', 'Rectus femoris'), ('vasmed_r', 'Vastus medialis'), ('bflh_r', 'Biceps femoris, long head'), ('semiten_r', 'Semitendinosus'), ('tibant_r', 'Tibialis anterior'), ('gasmed_r', 'Medial gastrocnemius'), ('gaslat_r', 'Lateral gastrocnemius'), ('soleus_r', 'Soleus / medial EMG'), ('soleus_lateral', 'Soleus / lateral EMG'), ('perlong_r', 'Peroneus longus'), ('perbrev_r', 'Peroneus brevis')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.65, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': 'black', 'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'savefig.bbox': None, 'figure.constrained_layout.use': False, 'figure.autolayout': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'hatch.linewidth': 0.4})
    (frames, channels, paths) = ({}, {}, [])
    for p in POLICIES:
        base = EXPERIMENT / 'evaluations' / p / 'test_iter15000'
        for fn in ('subjects.csv', 'analysis/per_subject.csv', 'analysis/per_channel.csv'):
            paths.append(str(base / fn))
        a = pd.read_csv(base / 'analysis/per_subject.csv').set_index('subject')
        b = pd.read_csv(base / 'subjects.csv').set_index('subject_id')
        assert a.index.is_unique and len(a) == 120 and a.index.equals(b.index)
        a['tracking_mm'] = b['body_position_error_m'] * 1000
        frames[p] = a
        channels[p] = pd.read_csv(base / 'analysis/per_channel.csv')
    combined = pd.concat([d.assign(policy=p) for (p, d) in frames.items()]).reset_index()
    combined.to_csv(ROOT / 'source_data/subject_metrics.csv', index=False)
    pd.concat([d.assign(policy=p) for (p, d) in channels.items()]).to_csv(ROOT / 'source_data/channel_metrics.csv', index=False)
    subjects = list(frames['film'].index)
    (human_emg, model_emg, human_grf, model_grf) = ([], [], [], [])
    for sid in subjects:
        hp = EXPERIMENT / 'references' / f'{sid}.npz'
        mp = EXPERIMENT / 'evaluations/film/test_iter15000/analysis' / f'{sid}_profiles.npz'
        paths.extend([str(hp), str(mp)])
        with np.load(hp) as h, np.load(mp) as m:
            assert h['emg_names'].tolist() == m['emg_names'].tolist()
            names = m['emg_names'].tolist()
            human_emg.append(h['emg'])
            model_emg.append(m['activation'])
            human_grf.append(h['grf_bw'])
            model_grf.append(m['grf_bw'])
    (human_emg, model_emg, human_grf, model_grf) = map(np.stack, (human_emg, model_emg, human_grf, model_grf))

    def normalize(a):
        low = a.min(axis=-1, keepdims=True)
        span = np.ptp(a, axis=-1, keepdims=True)
        return np.divide(a - low, span, out=np.zeros_like(a), where=span > 1e-12)
    human_emg = normalize(human_emg)
    model_emg = normalize(model_emg)
    phase = np.linspace(0, 100, 100)
    wave_rows = []
    for (domain, first, second, keys) in [('emg', human_emg, model_emg, names), ('grf', human_grf, model_grf, ['left', 'right'])]:
        for (i, key) in enumerate(keys):
            valid = np.isfinite(first[:, i]).all(axis=1) & np.isfinite(second[:, i]).all(axis=1)
            for (kind, arr) in [('measured', first), ('simulated', second)]:
                z = arr[valid, i]
                for (ph, mean, sd) in zip(phase, z.mean(axis=0), z.std(axis=0)):
                    wave_rows.append(dict(domain=domain, channel=key, source=kind, phase_percent=ph, mean=mean, sd=sd, n=len(z)))
    pd.DataFrame(wave_rows).to_csv(ROOT / 'source_data/waveform_summary.csv', index=False)

    def canvas(height):
        return (plt.figure(figsize=(180 / 25.4, height / 25.4)), height)

    def axis(fig, h, x, y, w, ht):
        ax = fig.add_axes([x / 180, 1 - (y + ht) / h, w / 180, ht / h])
        ax.tick_params(length=2.2, width=0.65, pad=2)
        ax.set_axisbelow(True)
        return ax

    def txt(fig, h, x, y, text, **kw):
        return fig.text(x / 180, 1 - y / h, text, va='top', **kw)

    def band(ax, a, b):
        for (arr, color, style) in [(a, '#62676B', '--'), (b, GREEN, '-')]:
            (mean, sd) = (arr.mean(axis=0), arr.std(axis=0))
            ax.fill_between(phase, mean - sd, mean + sd, color=color, alpha=0.14, linewidth=0)
            ax.plot(phase, mean, color=color, lw=1.05, ls=style)

    def hdist(ax, metric, xlabel, limits, ticks, labels=True):
        for (i, p) in enumerate(POLICIES[:3]):
            vals = frames[p][metric].dropna().to_numpy()
            color = COLORS[i]
            jitter = np.random.default_rng(170 + i).uniform(-0.19, 0.19, len(vals))
            ax.scatter(vals, i + jitter, s=3, color=color, alpha=0.25, edgecolors='none', zorder=2)
            ax.boxplot(vals, positions=[i], orientation='horizontal', widths=0.44, patch_artist=True, showfliers=False, manage_ticks=False, boxprops=dict(facecolor=color, alpha=0.28, edgecolor=color, lw=0.7), medianprops=dict(color=color, lw=1.1), whiskerprops=dict(color=color, lw=0.7), capprops=dict(color=color, lw=0.7))
        ax.set(ylim=(2.55, -0.55), xlim=limits, xticks=ticks, yticks=[0, 1, 2], xlabel=xlabel)
        ax.set_yticklabels(LABELS[:3] if labels else [''] * 3)
        ax.grid(axis='x', color='#E4E6E7', lw=0.4)

    def finish(fig, name):
        for extension in ('svg', 'pdf', 'png'):
            fig.savefig(OUT / f'{name}.{extension}', dpi=300)
        plt.close(fig)

    def main_figure():
        (fig, h) = canvas(116)
        txt(fig, h, 2, 3, 'b', fontsize=8, fontweight='bold')
        txt(fig, h, 50.5, 3, 'Reference tracking', fontsize=8, fontweight='bold', ha='center')
        ax = axis(fig, h, 29, 21, 43, 27)
        maxerr = max((frames[p].tracking_mm.max() for p in POLICIES[:3]))
        high = np.ceil(maxerr / 20) * 20
        hdist(ax, 'tracking_mm', 'Tracking error (mm)', (35, 90), [40, 60, 80])
        txt(fig, h, 74, 3, 'c', fontsize=8, fontweight='bold')
        txt(fig, h, 130, 3, 'Vertical GRF', fontsize=8, fontweight='bold', ha='center')
        handles = [Line2D([], [], color='#62676B', ls='--', lw=1, label='Measured'), Line2D([], [], color=GREEN, lw=1, label='MyoAcross')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(130 / 180, 0.935), ncol=2, frameon=False, handlelength=2, columnspacing=1, borderaxespad=0)
        valid_counts = []
        for (i, (x, title)) in enumerate([(82, 'Left'), (135, 'Right')]):
            valid = np.isfinite(human_grf[:, i]).all(axis=1) & np.isfinite(model_grf[:, i]).all(axis=1)
            valid_counts.append(int(valid.sum()))
            ax = axis(fig, h, x, 21, 43, 27)
            band(ax, human_grf[valid, i], model_grf[valid, i])
            ax.set(xlim=(0, 100), xticks=[0, 50, 100], ylim=(0, 1.5), yticks=[0, 0.5, 1, 1.5], xlabel='Stance phase (%)')
            ax.set_title(title, fontsize=8, pad=3)
            if i == 0:
                ax.set_ylabel('Vertical GRF (BW)', labelpad=3)
            else:
                ax.set_yticklabels([])
        txt(fig, h, 77, 62, 'GRF agreement', fontsize=8, fontweight='bold', ha='center')
        txt(fig, h, 156.5, 62, 'Muscle activation', fontsize=8, fontweight='bold', ha='center')
        for (letter, x, title, metric, label, lim, ticks, show) in [('d', 29, 'GRF waveform', 'grf_stance', 'Pearson r', (0.5, 1), [0.5, 0.75, 1], True), ('e', 82, 'GRF error', 'grf_rmse_bw', 'RMSE (BW)', (0, 0.35), [0, 0.1, 0.2, 0.3], False), ('f', 135, 'Muscle activation', 'emg12', 'Mean Pearson r', (-0.2, 0.8), [-0.2, 0.3, 0.8], False)]:
            txt(fig, h, 2 if letter == 'd' else x - 6, 69, letter, fontsize=8, fontweight='bold')
            ax = axis(fig, h, x, 77, 43, 27)
            if metric == 'grf_rmse_bw':
                hi = max((frames[p][metric].max() for p in POLICIES[:3]))
                lim = (0, max(0.35, np.ceil(hi * 10) / 10))
            hdist(ax, metric, label, lim, ticks, show)
        finish(fig, 'gait120_performance')

    def supplementary_figure():
        (fig, h) = canvas(246)
        handles = [Line2D([], [], color='#62676B', ls='--', lw=1, label='Measured EMG'), Line2D([], [], color=GREEN, lw=1, label='MyoAcross activation')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.51, 0.997), ncol=2, frameon=False, borderaxespad=0)
        txt(fig, h, 90, 7, 'MM: MyoAcross     MLP: Unconditioned MLP     V0: MyoAcross V0', ha='center')
        txt(fig, h, 90, 11, 'Asym.: MyoAcross with asymmetry augmentation', ha='center')
        for (j, (key, title)) in enumerate(CHANNELS):
            (row, col) = divmod(j, 3)
            x = 12 + col * 60
            y = 22 + row * 56
            txt(fig, h, x - 10, y - 4, chr(97 + j), fontsize=8, fontweight='bold')
            txt(fig, h, x + 23, y - 4, title, fontsize=8, ha='center')
            idx = names.index(key)
            ax = axis(fig, h, x, y + 1, 45, 23)
            band(ax, human_emg[:, idx], model_emg[:, idx])
            ax.set(xlim=(0, 100), xticks=[0, 50, 100], ylim=(-0.1, 1.15), yticks=[0, 0.5, 1])
            ax.set_xlabel('Gait cycle (%)', labelpad=1)
            if col == 0:
                ax.set_ylabel('Normalized amplitude', labelpad=2)
            else:
                ax.set_yticklabels([])
            bx = axis(fig, h, x, y + 36, 45, 12)
            for (k, p) in enumerate(POLICIES):
                frame = channels[p]
                v = frame[(frame.domain == 'emg') & (frame.channel == key)].pearson.dropna().to_numpy()
                bx.boxplot(v, positions=[k], widths=0.52, patch_artist=True, manage_ticks=False, showfliers=True, boxprops=dict(facecolor=COLORS[k], alpha=0.5, edgecolor=COLORS[k], lw=0.6), medianprops=dict(color=INK, lw=0.7), whiskerprops=dict(color=COLORS[k], lw=0.6), capprops=dict(color=COLORS[k], lw=0.6), flierprops=dict(marker='.', markersize=1.8, markeredgewidth=0, markerfacecolor=COLORS[k], alpha=0.5))
            bx.axhline(0, color='#B9BDC0', lw=0.45, ls=':', zorder=0)
            bx.set(xlim=(-0.6, 4.6), ylim=(-1.08, 1.08), yticks=[-1, 0, 1], xticks=range(5), xticklabels=SHORT)
            bx.tick_params(axis='x', length=0, pad=2)
            if col == 0:
                bx.set_ylabel('Pearson r', labelpad=2)
            else:
                bx.set_yticklabels([])
        finish(fig, 'emg_channels')
    main_figure()
    if not args.main_only:
        supplementary_figure()
    (ROOT / 'provenance.json').write_text(json.dumps(dict(experiment=str(EXPERIMENT), sources=paths, policy_order=POLICIES, width_mm=180, font='DejaVu Sans', content_pt=7, title_pt=8, waveform_band='population standard deviation across subjects (ddof=0)', emg_display='min-max normalize each subject/channel mean profile separately; no phase shifts', correlation='unchanged per-subject Pearson values read from analysis CSV', boxplot='median, Q1-Q3, whiskers to 1.5 IQR; main all points; SI outliers', simulation='no new experiments'), indent=2))
if __name__ == '__main__':
    main()
