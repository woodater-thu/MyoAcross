"""Plot evaluation details results from existing experiment outputs."""
from pathlib import Path
import argparse
from myoacross.utils.paths import PROJECT_ROOT

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import sys, json, re
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.colors import LinearSegmentedColormap
    ROOT = args.output_dir
    R = args.data_root
    D = args.data_root
    OUT = args.output_dir
    OUT.mkdir(exist_ok=True)
    (OUT / 'source_data').mkdir(exist_ok=True)
    (OUT / 'qa').mkdir(exist_ok=True)
    (BLUE, GREEN, GRAY, INK, GRID) = ('#4474A7', '#788961', '#8B9297', '#26323B', '#E4E6E7')
    COLORS = [GRAY, BLUE, GREEN]
    WIDTH = 510 / 72
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7, 'axes.linewidth': 0.75, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'savefig.bbox': None, 'figure.constrained_layout.use': False, 'figure.autolayout': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False})
    PROVENANCE = []
    CAPTIONS = []

    def source(path):
        PROVENANCE.append(str(path))
        return path

    def save_data(name, frame, groups=None):
        frame.to_csv(OUT / 'source_data' / f'{name}.csv', index=False)

    def tidy(ax, axis='x'):
        ax.set_axisbelow(True)
        ax.grid(axis=axis, color=GRID, linewidth=0.45)
        ax.tick_params(length=2.5, width=0.65, pad=2)

    def letter(fig, text, x, y):
        fig.text(x, y, text, fontweight='bold', fontsize=10, va='top')

    def finish(fig, name):
        for extension in ('svg', 'pdf', 'png'):
            fig.savefig(OUT / f'{name}.{extension}', dpi=300)
        plt.close(fig)

    def hbox(ax, values, y, color, width=0.55):
        ax.boxplot(values, positions=y, orientation='horizontal', widths=width, patch_artist=True, showfliers=False, manage_ticks=False, boxprops={'facecolor': color, 'alpha': 0.3, 'edgecolor': color, 'linewidth': 0.7}, medianprops={'color': color, 'linewidth': 1.05}, whiskerprops={'color': color, 'linewidth': 0.65}, capprops={'color': color, 'linewidth': 0.65})

    def s1c():
        obj = json.load(open(source(R / 'experiments/skeletal_fitting/evaluations/report/results.json')))
        names = ['hip_width', 'shoulder_width', 'trunk_length', 'upper_arm_length_r', 'upper_arm_length_l', 'forearm_length_r', 'forearm_length_l', 'thigh_length_r', 'thigh_length_l', 'shank_length_r', 'shank_length_l', 'ankle_subtalar_length_r', 'ankle_subtalar_length_l', 'subtalar_mtp_length_r', 'subtalar_mtp_length_l', 'shoulder_ankle_height_r', 'shoulder_ankle_height_l']
        labels = ['Hip width', 'Shoulder width', 'Trunk length', 'Upper arm, R', 'Upper arm, L', 'Forearm, R', 'Forearm, L', 'Thigh, R', 'Thigh, L', 'Shank, R', 'Shank, L', 'Ankle–subtalar, R', 'Ankle–subtalar, L', 'Subtalar–MTP, R', 'Subtalar–MTP, L', 'Shoulder–ankle height, R', 'Shoulder–ankle height, L']
        methods = ['baseline', 'height_scale', 'segment_scale']
        titles = ['Template', 'Global fit', 'Regional fit']
        frame = pd.DataFrame([{'individual': row['subject'], 'split': row['split'], 'method': row['method'], 'measurement': n, 'signed_residual_mm': 1000 * row['signed_errors_m'][n], 'absolute_error_mm': 1000 * abs(row['signed_errors_m'][n])} for row in obj['rows'] for n in names])
        assert len(frame) == 141 * 3 * 17 and (not frame.duplicated(['individual', 'method', 'measurement']).any())
        save_data('S1c_residuals', frame, ['method'])
        (fig, axs) = plt.subplots(1, 4, figsize=(WIDTH, 4.85), gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
        fig.subplots_adjust(left=0.242, right=0.984, bottom=0.12, top=0.88, wspace=0.16)
        positions = np.arange(17)
        rng = np.random.default_rng(42)
        for (k, (method, title, ax)) in enumerate(zip(methods, titles, axs)):
            vals = [frame[(frame.method == method) & (frame.measurement == n)].signed_residual_mm.to_numpy() for n in names]
            assert all((len(x) == 141 for x in vals))
            assert np.isclose(np.abs(vals).mean(), obj['summary']['all'][method]['mae_m'] * 1000)
            for (y, v) in zip(positions, vals):
                ax.scatter(v, y + rng.uniform(-0.19, 0.19, len(v)), s=2.1, color=COLORS[k], alpha=0.27, linewidths=0, rasterized=True)
            hbox(ax, vals, positions, COLORS[k], 0.5)
            ax.axvline(0, color=INK, linewidth=0.6, linestyle='--', zorder=0)
            ax.set_title(title, fontweight='bold', pad=10)
            ax.set_xlabel('Signed residual (mm)', labelpad=5)
            ax.set_ylim(16.6, -0.6)
            ax.set_yticks(positions, labels if k == 0 else [])
            tidy(ax)
            ax.tick_params(axis='y', length=0)
        axs[0].set_xlim(-150, 250)
        axs[0].set_xticks([-100, 0, 100, 200])
        axs[1].set_xlim(-85, 85)
        axs[1].set_xticks([-75, 0, 75])
        axs[2].set_xlim(-15, 15)
        axs[2].set_xticks([-15, 0, 15])
        ax = axs[3]
        for (k, method) in enumerate(methods):
            vals = [frame[(frame.method == method) & (frame.measurement == n)].absolute_error_mm.mean() for n in names]
            ax.scatter(vals, positions + (k - 1) * 0.19, s=12, marker=['s', '^', 'D'][k], color=COLORS[k], edgecolor='white', linewidth=0.3, zorder=3)
        ax.set_title('Absolute error', fontweight='bold', pad=10)
        ax.set_xlabel('Mean absolute error (mm)', labelpad=5)
        ax.set_yticks([])
        ax.set_ylim(16.6, -0.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        tidy(ax)
        for ax in axs:
            for y in [2.5, 6.5, 10.5, 14.5]:
                ax.axhline(y, color=GRID, linewidth=0.4, zorder=0)
        handles = [Line2D([], [], color=c, marker=m, linestyle='', markersize=4, label=t) for (c, m, t) in zip(COLORS, ['s', '^', 'D'], titles)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.6, 0.986), ncol=3, frameon=False)
        letter(fig, 'c', 0.018, 0.985)
        finish(fig, 'S1c')

    def sidkey(s):
        return (0 if 'BMLmovi' in s else 1, int(re.search('(?:Subject_|KIT_)(\\d+)', s).group(1)))

    def sidlabel(s):
        return ('MoVi ' if 'BMLmovi' in s else 'KIT ') + str(sidkey(s)[1])

    def trials(path):
        df = pd.read_csv(source(path))
        df['completion_pct'] = df.completed.astype(float) * 100
        df['coverage_pct'] = df.coverage * 100
        df['tracking_error_mm'] = df.body_position_error_m * 1000
        return df

    def s2():
        paths = [D / 'experiments/policy_comparison/evaluations/segment_mlp/test_iter015000/trials.csv', D / 'experiments/policy_comparison/evaluations/segment_concat/test_iter015000/trials.csv', D / 'myoacross/rl/evaluations/test_iter015000/trials.csv']
        dfs = [trials(p) for p in paths]
        titles = ['Unconditioned MLP', 'Concat', 'MyoAcross']
        subjects = sorted(dfs[0].subject_id.unique(), key=sidkey)
        cats = ['walk_slow', 'walk_medium', 'walk_fast', 'walk_backward', 'turn_left', 'turn_right', 'circle_cw', 'circle_ccw']
        metrics = ['completion_pct', 'coverage_pct', 'tracking_error_mm']
        aggs = []
        cells = []
        for (title, df) in zip(titles, dfs):
            assert len(df) == 1984 and (not df.duplicated(['subject_id', 'motion_id']).any())
            assert sorted(df.subject_id.unique(), key=sidkey) == subjects
            assert (df.groupby(['subject_id', 'category']).size() == 8).all()
            a = df.groupby('subject_id')[metrics].mean().reindex(subjects)
            aggs.append(a)
            cells.append(df.groupby(['subject_id', 'category']).completion_pct.mean().unstack().reindex(index=subjects, columns=cats))
        save_data('S2_individual_metrics', pd.concat([x.assign(policy=t).reset_index() for (x, t) in zip(aggs, titles)]), ['policy'])
        save_data('S2_category_completion', pd.concat([x.rename_axis(columns='category').stack().rename('completion_pct').reset_index().assign(policy=t) for (x, t) in zip(cells, titles)]), ['policy'])
        delta = aggs[2] - aggs[0]
        assert (delta.tracking_error_mm < 0).all()
        save_data('S2_paired_changes', delta.reset_index())
        (fig, axs) = plt.subplots(1, 5, figsize=(WIDTH, 5.7), gridspec_kw={'width_ratios': [1, 1, 1, 0.7, 0.7]})
        fig.subplots_adjust(left=0.102, right=0.98, top=0.905, bottom=0.225, wspace=0.16)
        cmap = LinearSegmentedColormap.from_list('completion', ['#FAFAF8', '#DBE0D5', GREEN])
        row44 = next((i for (i, s) in enumerate(subjects) if 'Subject_44_' in s))
        for (k, ax) in enumerate(axs[:3]):
            im = ax.imshow(cells[k].values, vmin=0, vmax=100, cmap=cmap, aspect='auto', interpolation='nearest')
            ax.set_yticks(range(31), [sidlabel(s) for s in subjects] if k == 0 else [])
            ax.set_xticks(range(8), ['Slow', 'Medium', 'Fast', 'Backward', 'Left', 'Right', 'CW', 'CCW'], rotation=90, ha='center')
            ax.tick_params(length=0, pad=3)
            ax.set_title(titles[k], fontsize=7.5, fontweight='bold', pad=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_patch(Rectangle((-0.5, row44 - 0.5), 8, 1, fill=False, edgecolor=INK, linewidth=0.8))
        axs[0].get_yticklabels()[row44].set_fontweight('bold')
        for (ax, metric, title, xlabel) in zip(axs[3:], ['tracking_error_mm', 'completion_pct'], ['Tracking error', 'Completion'], ['Change (mm)', 'Change (pp)']):
            vals = delta[metric].values
            ax.axvline(0, color=INK, linewidth=0.65, linestyle='--')
            ax.hlines(np.arange(31), 0, vals, color=GREEN, linewidth=0.7, alpha=0.6)
            ax.scatter(vals, np.arange(31), s=10, color=GREEN, marker='D', linewidth=0.25, edgecolor='white', zorder=3)
            ax.scatter([vals[row44]], [row44], s=22, facecolor='white', edgecolor=INK, marker='D', linewidth=0.8, zorder=4)
            ax.set_ylim(30.5, -0.5)
            ax.set_yticks([])
            ax.set_title(title, fontsize=7.5, fontweight='bold', pad=7)
            ax.set_xlabel(xlabel, labelpad=5)
            tidy(ax)
        axs[3].set_xlim(-26, 2)
        axs[3].set_xticks([-20, -10, 0])
        axs[4].set_xlim(-3, 9)
        axs[4].set_xticks([0, 5])
        cax = fig.add_axes([0.15, 0.081, 0.43, 0.016])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 25, 50, 75, 100])
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, pad=2, labelsize=6.5)
        cb.set_label('Completion (%)', labelpad=3, fontsize=7)
        letter(fig, 'a', 0.018, 0.977)
        letter(fig, 'b', 0.718, 0.977)
        finish(fig, 'S2')

    def meta(p):
        with np.load(p, allow_pickle=False) as z:
            return json.loads(z['metadata'].item())

    def s3():
        banks = [R / 'myoacross/rl/embodiments_bundles/subject', R / 'myoacross/rl/embodiments_bundles/virtual', R / 'experiments/asymmetry_generalization/embodiments_bundles/asymmetry_enhanced/virtual']
        originals = {}
        arows = []
        prows = []
        titles = ['Original', 'Standard V10', 'Asymmetry V10']
        regions = ['pelvis_sacrum', 'trunk', 'shoulder_girdle', 'upper_arm_r', 'upper_arm_l', 'forearm_r', 'forearm_l', 'thigh_r', 'thigh_l', 'shank_r', 'shank_l', 'foot_r', 'foot_l']
        for (k, bank) in enumerate(banks):
            source(bank)
            paths = sorted(bank.glob('*/*.npz'))
            assert len(paths) == [110, 1100, 1100][k]
            for p in paths:
                m = meta(p)
                s = m['source_subject_id']
                f = dict(zip(m['features']['names'], m['features']['values']))
                if k == 0:
                    report = json.load(open(R / 'myoacross/embodiments/segment_scale' / s / 'report.json'))
                    originals[s] = (report['scales']['regions'], f['body_mass'])
                else:
                    pr = m['parameters']
                    (orig, omass) = originals[s]
                    for (n, v) in zip(pr['region_names'], pr['region_scales']):
                        prows.append({'source_individual': s, 'variant': p.stem, 'bank': titles[k], 'parameter': n, 'relative_change_pct': 100 * (v / orig[n] - 1)})
                    prows.append({'source_individual': s, 'variant': p.stem, 'bank': titles[k], 'parameter': 'body_mass', 'relative_change_pct': 100 * (pr['body_mass_kg'] / omass - 1)})
                for seg in ['Thigh', 'Shank', 'Lower limb']:
                    rs = [f[n + '_length_r'] for n in (['thigh', 'shank'] if seg == 'Lower limb' else [seg.lower()])]
                    ls = [f[n + '_length_l'] for n in (['thigh', 'shank'] if seg == 'Lower limb' else [seg.lower()])]
                    arows.append({'source_individual': s, 'variant': p.stem, 'bank': titles[k], 'measurement': seg, 'asymmetry_pct': 200 * abs(sum(rs) - sum(ls)) / (sum(rs) + sum(ls))})
        pp = pd.DataFrame(prows)
        aa = pd.DataFrame(arows)
        assert pp.groupby(['bank', 'parameter']).size().eq(1100).all()
        assert np.isclose(aa[(aa.bank == 'Original') & (aa.measurement == 'Thigh')].asymmetry_pct.mean(), 0.2630977363)
        save_data('S3_accepted_parameters', pp, ['bank'])
        save_data('S3_measured_asymmetry', aa, ['bank'])
        fig = plt.figure(figsize=(WIDTH, 6.0))
        gs = fig.add_gridspec(2, 3, left=0.155, right=0.976, bottom=0.095, top=0.915, height_ratios=[1.7, 1], hspace=0.28, wspace=0.32)
        ax = fig.add_subplot(gs[0, :2])
        mass = fig.add_subplot(gs[0, 2])
        rng = np.random.default_rng(42)
        labels = ['Pelvis–sacrum', 'Trunk', 'Shoulder girdle', 'Upper arm, R', 'Upper arm, L', 'Forearm, R', 'Forearm, L', 'Thigh, R', 'Thigh, L', 'Shank, R', 'Shank, L', 'Foot, R', 'Foot, L']
        for (k, bank) in enumerate(titles[1:]):
            color = [BLUE, GREEN][k]
            y = np.arange(13) + (k - 0.5) * 0.32
            vals = [pp[(pp.bank == bank) & (pp.parameter == n)].relative_change_pct.values for n in regions]
            for (pos, values) in zip(y, vals):
                ax.scatter(values, pos + rng.uniform(-0.1, 0.1, len(values)), s=0.7, alpha=0.11, color=color, linewidths=0, rasterized=True)
            hbox(ax, vals, y, color, 0.25)
            values = pp[(pp.bank == bank) & (pp.parameter == 'body_mass')].relative_change_pct.values
            mass.scatter(k + rng.uniform(-0.15, 0.15, len(values)), values, s=1, alpha=0.14, color=color, linewidths=0, rasterized=True)
            mass.boxplot([values], positions=[k], widths=0.35, showfliers=False, patch_artist=True, boxprops={'facecolor': color, 'alpha': 0.3, 'edgecolor': color, 'linewidth': 0.7}, medianprops={'color': color}, whiskerprops={'color': color, 'linewidth': 0.65}, capprops={'color': color, 'linewidth': 0.65})
        ax.set_yticks(range(13), labels)
        ax.set_ylim(12.6, -0.6)
        ax.set_xlim(-13, 13)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_xlabel('Regional scale change (%)')
        tidy(ax)
        ax.axvline(0, color=INK, linestyle='--', linewidth=0.6, zorder=0)
        ax.tick_params(axis='y', length=0)
        mass.set_xticks([0, 1], ['Standard\nV10', 'Asymmetry\nV10'])
        mass.set_ylabel('Mass change (%)')
        mass.set_ylim(-16, 16)
        mass.set_yticks([-15, 0, 15])
        tidy(mass, axis='y')
        mass.axhline(0, color=INK, linewidth=0.6, linestyle='--')
        ax.set_title('Regional scales', fontweight='bold', pad=8)
        mass.set_title('Body mass', fontweight='bold', pad=8)
        for (j, seg) in enumerate(['Thigh', 'Shank', 'Lower limb']):
            a = fig.add_subplot(gs[1, j])
            tidy(a, axis='y')
            for (k, bank) in enumerate(titles):
                v = aa[(aa.bank == bank) & (aa.measurement == seg)].asymmetry_pct.values
                a.scatter(k + rng.uniform(-0.17, 0.17, len(v)), v, s=1.5, alpha=0.2, color=COLORS[k], linewidths=0, rasterized=True)
                a.boxplot([v], positions=[k], widths=0.42, showfliers=False, patch_artist=True, boxprops={'facecolor': COLORS[k], 'alpha': 0.35, 'edgecolor': COLORS[k], 'linewidth': 0.7}, medianprops={'color': COLORS[k], 'linewidth': 1}, whiskerprops={'color': COLORS[k], 'linewidth': 0.65}, capprops={'color': COLORS[k], 'linewidth': 0.65})
            a.set_xticks([0, 1, 2], ['Original', 'Standard\nV10', 'Asymmetry\nV10'])
            a.set_ylim(-0.4, 22)
            a.set_yticks([0, 5, 10, 15, 20])
            a.set_title('Combined lower limb' if seg == 'Lower limb' else seg, fontweight='bold', pad=7)
            if j == 0:
                a.set_ylabel('Measured asymmetry (%)')
        fig.legend(handles=[Patch(facecolor=BLUE, alpha=0.6, label='Standard V10'), Patch(facecolor=GREEN, alpha=0.6, label='Asymmetry V10')], loc='upper center', bbox_to_anchor=(0.59, 0.994), ncol=2, frameon=False)
        letter(fig, 'a', 0.018, 0.978)
        letter(fig, 'b', 0.018, 0.39)
        finish(fig, 'S3')

    def s4():
        targets = ['morphology', 'reference', 'combined']
        policies = ['virtual0_film', 'segment_film']
        metrics = ['completion_pct', 'coverage_pct', 'tracking_error_mm']
        rows = []
        summaries = []
        curves = {}
        boot = np.random.default_rng(42).integers(0, 31, size=(10000, 31))
        expected_subjects = None
        for policy in policies:
            for target in targets:
                df = trials(D / f'experiments/input_sensitivity/evaluations/{policy}/test_iter15000/{target}/trials.csv')
                assert len(df) == 19840
                assert not df.duplicated(['subject_id', 'motion_id', 'morphology_magnitude', 'reference_magnitude_deg', 'sample']).any()
                assert df.groupby(['subject_id', 'morphology_magnitude', 'reference_magnitude_deg', 'sample']).motion_id.nunique().eq(64).all()
                motion_sets = df.groupby(['subject_id', 'morphology_magnitude', 'reference_magnitude_deg', 'sample']).motion_id.agg(frozenset)
                assert motion_sets.nunique() == 1
                keys = ['subject_id', 'morphology_magnitude', 'reference_magnitude_deg']
                agg = df.groupby(keys)[metrics].mean()
                counts = df.groupby(keys).size()
                assert sorted(counts.unique()) == [64, 192]
                subjects = sorted(df.subject_id.unique(), key=sidkey)
                assert len(subjects) == 31
                if expected_subjects is None:
                    expected_subjects = subjects
                assert subjects == expected_subjects
                levels = sorted(set(zip(df.morphology_magnitude, df.reference_magnitude_deg)))
                assert len(levels) == 4 and levels[0] == (0, 0)
                arr = np.array([[agg.loc[(s, *lev)].values for s in subjects] for lev in levels])
                assert np.isfinite(arr).all()
                for diff in [False, True]:
                    v = arr - arr[0] if diff else arr
                    mean = v.mean(axis=1)
                    ci = np.quantile(v[:, boot, :].mean(axis=2), [0.025, 0.975], axis=1)
                    curves[policy, target, diff] = (mean, ci, levels)
                    for (j, lev) in enumerate(levels):
                        for (k, metric) in enumerate(metrics):
                            summaries.append({'policy': policy, 'target': target, 'change_from_own_baseline': diff, 'morphology_magnitude': lev[0], 'reference_magnitude_deg': lev[1], 'metric': metric, 'mean': mean[j, k], 'ci_low': ci[0, j, k], 'ci_high': ci[1, j, k]})
                for (j, lev) in enumerate(levels):
                    for (i, s) in enumerate(subjects):
                        row = {'individual': s, 'policy': policy, 'target': target, 'morphology_magnitude': lev[0], 'reference_magnitude_deg': lev[1]}
                        row.update({metric: arr[j, i, k] for (k, metric) in enumerate(metrics)})
                        row.update({'delta_' + metric: arr[j, i, k] - arr[0, i, k] for (k, metric) in enumerate(metrics)})
                        rows.append(row)
        save_data('S4_individual_metrics', pd.DataFrame(rows), ['policy', 'target'])
        summary = pd.DataFrame(summaries)
        summary.to_csv(OUT / 'source_data/S4_bootstrap_summary.csv', index=False)
        (fig, axs) = plt.subplots(4, 3, figsize=(WIDTH, 7.15))
        fig.subplots_adjust(left=0.106, right=0.982, bottom=0.095, top=0.91, hspace=0.22, wspace=0.24)
        names = ['Morphology features', 'Reference-motion input', 'Combined inputs']
        for (r, (diff, metricidx)) in enumerate([(False, 0), (False, 2), (True, 0), (True, 2)]):
            for (c, target) in enumerate(targets):
                ax = axs[r, c]
                tidy(ax, axis='y')
                for (p, policy) in enumerate(policies):
                    (mean, ci, levels) = curves[policy, target, diff]
                    x = np.array([lev[1] if target == 'reference' else lev[0] * 100 for lev in levels])
                    color = [BLUE, GREEN][p]
                    marker = ['s', 'D'][p]
                    ls = ['--', '-'][p]
                    ax.plot(x, mean[:, metricidx], color=color, linestyle=ls, marker=marker, markersize=3.4, markeredgecolor='white', markeredgewidth=0.35, linewidth=1, label=['V0', 'V10'][p], zorder=3)
                    ax.fill_between(x, ci[0, :, metricidx], ci[1, :, metricidx], color=color, alpha=0.13, linewidth=0)
                ax.set_xticks(x)
                ax.set_xlim(-0.4, 10.4)
                if r == 0:
                    ax.set_title(names[c], fontweight='bold', pad=9)
                if metricidx == 0:
                    ax.set_ylim((-102, 7) if diff else (-3, 103))
                    ax.set_yticks([-100, -50, 0] if diff else [0, 50, 100])
                if diff:
                    ax.axhline(0, color=INK, linewidth=0.6, linestyle=':', zorder=1)
                if r == 3:
                    if target == 'morphology':
                        ax.set_xlabel('Feature perturbation (%)')
                        ax.set_xticklabels(['0', '2', '5', '10'])
                    elif target == 'reference':
                        ax.set_xlabel('Joint-angle offset (°)')
                        ax.set_xticklabels(['0', '2.5', '5', '10'])
                    else:
                        ax.set_xticklabels(['0 / 0', '2 / 2.5', '5 / 5', '10 / 10'])
                        ax.set_xlabel('Feature (%) / joint angle (°)')
                else:
                    ax.set_xticklabels([])
            axs[r, 0].set_ylabel(['Completion (%)', 'Tracking error (mm)', 'Completion change (pp)', 'Tracking-error change (mm)'][r])
        for r in [1, 3]:
            upper = max((curves[p, t, r == 3][1][1, :, 2].max() for p in policies for t in targets))
            upper = np.ceil(upper / 50) * 50
            for ax in axs[r]:
                ax.set_ylim(-5 if r == 3 else 0, upper)
        handles = [Line2D([], [], color=c, marker=m, linestyle=l, linewidth=1, markersize=4, label=t) for (c, m, l, t) in zip([BLUE, GREEN], ['s', 'D'], ['--', '-'], ['V0', 'V10'])]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.56, 0.986), ncol=2, frameon=False)
        letter(fig, 'a', 0.018, 0.969)
        letter(fig, 'b', 0.018, 0.503)
        finish(fig, 'S4')
    s1c()
    s2()
    s3()
    s4()
    (OUT / 'source_paths.json').write_text(json.dumps(PROVENANCE, indent=2) + '\n')
if __name__ == '__main__':
    main()
