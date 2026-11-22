"""Render existing analysis results."""
COLORS = ('#4477AA', '#EE6677')

from pathlib import Path
import json
import numpy as np
from myoacross.utils.paths import EMBODIMENTS_DIR


def _save_figure(figure, output: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    for extension in ('svg', 'pdf', 'tiff', 'png'):
        options = {'pil_kwargs': {'compression': 'tiff_lzw'}} if extension == 'tiff' else {}
        figure.savefig(output / f'{stem}.{extension}', dpi=300 if extension == 'png' else 600,
                       bbox_inches='tight', facecolor='white', **options)
    plt.close(figure)


def plot_distributions(banks: dict, summaries: dict, output: Path) -> None:
    """Compare relative distributions, dispersion, and accepted correlations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    import scienceplots  # noqa: F401 -- register Matplotlib styles

    labels = list(banks)
    first = next(iter(banks.values()))
    colors = dict(zip(labels, COLORS))
    with plt.style.context(['science', 'nature', 'no-latex', 'bright']), plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    }):
        for field, name_field, stem in (
            ('parameter_changes', 'parameter_names', 'parameter_distributions'),
            ('feature_changes', 'feature_names', 'feature_distributions'),
        ):
            names = first[name_field]
            figure, axis = plt.subplots(figsize=(7.2, 0.35 * len(names) + 1.2), layout='constrained')
            positions = np.arange(len(names))
            offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2) * 0.18
            for label, offset in zip(labels, offsets):
                boxes = axis.boxplot(banks[label][field], positions=positions + offset,
                                     orientation='horizontal', widths=0.14, patch_artist=True,
                                     whis=(2.5, 97.5), showfliers=False, manage_ticks=False)
                for box, median in zip(boxes['boxes'], boxes['medians']):
                    box.set(facecolor=colors[label], edgecolor=colors[label], alpha=0.65)
                    median.set(color='black', linewidth=0.8)
            axis.axvline(0, color='0.5', linestyle='--', linewidth=0.6)
            axis.set_yticks(positions, [n.replace('_length', '').replace('_', ' ') for n in names])
            axis.yaxis.set_minor_locator(NullLocator())
            axis.invert_yaxis()
            axis.set_xlabel('Change from original subject (%)')
            axis.set_title('Boxes: median and interquartile range; whiskers: 2.5th–97.5th percentiles', fontsize=8)
            axis.legend(handles=[Line2D([], [], color=colors[label], linewidth=4, label=label)
                                 for label in labels], loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=len(labels))
            _save_figure(figure, output, stem)

        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout='constrained')
        for axis, field, name_field in zip(axes, ('parameters', 'features'), ('parameter_names', 'feature_names')):
            names = first[name_field]
            x = np.arange(len(names))
            for label in labels:
                axis.plot(x, summaries[label][field]['std_percent'], '.-', color=colors[label], label=label)
            axis.set_xticks(x, [n.replace('_length', '').replace('_', ' ') for n in names], rotation=90, fontsize=7)
            axis.set_ylabel('Accepted-sample SD (percentage points)')
            axis.set_title('Generation parameters' if field == 'parameters' else 'Measured features')
            axis.set_ylim(bottom=0)
        axes[1].legend()
        _save_figure(figure, output, 'accepted_spread')

        figure, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels) + 0.5, 3.5),
                                    layout='constrained', squeeze=False)
        for axis, label in zip(axes[0], labels):
            matrix = np.array(summaries[label]['empirical_correlation'], dtype=float)
            image = axis.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            names = first['parameter_names']
            axis.set_xticks(range(len(names)), [n.replace('_', ' ') for n in names], rotation=90, fontsize=6)
            axis.set_yticks(range(len(names)), [n.replace('_', ' ') for n in names], fontsize=6)
            axis.xaxis.set_minor_locator(NullLocator())
            axis.yaxis.set_minor_locator(NullLocator())
            axis.set_title(label)
        figure.colorbar(image, ax=axes[0].tolist(), label='Accepted-sample correlation', shrink=0.8)
        _save_figure(figure, output, 'accepted_correlations')


def plot_extreme_bodies(banks: dict, output: Path) -> None:
    """Render stored extreme bodies and originals with a shared camera scale."""
    import mujoco
    import matplotlib.pyplot as plt
    from myoacross.rl.embodiments.virtual.generator import VirtualEmbodiment
    from myoacross.rl.embodiments.virtual.parameters import SegmentScaleParameters

    from myoacross.utils.analysis.analyze_asymmetry import select_extreme_samples

    selected = select_extreme_samples(banks)
    bodies = []
    for label, record in selected.items():
        bank = banks[label]
        row = bank['rows'][record['row_index']]
        directory = EMBODIMENTS_DIR / 'segment_scale' / record['subject']
        report = json.loads((directory / 'report.json').read_text())
        baseline = Path(report['baseline_path']).expanduser()
        if not baseline.is_absolute():
            baseline = directory / baseline
        with np.load(row['bundle'], allow_pickle=False) as archive:
            parameters = json.loads(archive['metadata'].item())['parameters']
        embodiment = VirtualEmbodiment(
            mujoco.MjSpec.from_file(str(baseline)),
            SegmentScaleParameters(tuple(parameters['region_scales']), parameters['body_mass_kg']),
            record['subject'],
        )
        # Reconstruct accepted parameters without resampling or repeating quality checks.
        virtual = embodiment.build(embodiment.solve(), parameters['body_mass_kg']).compile()
        original = mujoco.MjModel.from_xml_path(str(directory / 'model.xml'))
        pair = []
        for model in (original, virtual):
            # Hide floor and decorations before rendering so they cannot occlude limbs.
            model.geom_rgba[model.geom_bodyid == 0, 3] = 0
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            # Ignore the world origin when framing the body.
            positions = data.xpos[1:]
            lower, upper = positions.min(axis=0), positions.max(axis=0)
            pair.append((model, data, lower, upper))
        bodies.append(pair)
        record['bundle'] = row['bundle']
        record['feature_changes_percent'] = dict(zip(bank['feature_names'], row['feature_changes'].tolist()))
        record['parameter_changes_percent'] = dict(zip(bank['parameter_names'], row['parameter_changes'].tolist()))

    span = max(float(np.max(upper - lower)) for pair in bodies for _, _, lower, upper in pair) + 0.4
    figure, axes = plt.subplots(len(bodies), 4, figsize=(10, 3.0 * len(bodies)), squeeze=False,
                                layout='constrained')
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    for row_index, ((label, record), pair) in enumerate(zip(selected.items(), bodies)):
        for body_index, (model, data, lower, upper) in enumerate(pair):
            model.vis.global_.offwidth = 600
            model.vis.global_.offheight = 600
            model.vis.global_.fovy = 45
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2,
                                lower[2] - 0.2 + span / 2]
            camera.distance = 1.3 * span
            camera.elevation = 0
            with mujoco.Renderer(model, height=600, width=600) as renderer:
                for view, azimuth in enumerate((90, 0)):
                    camera.azimuth = azimuth
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    rgb = renderer.render().copy()
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render()
                    renderer.disable_segmentation_rendering()
                    ids, types = segmentation[..., 0], segmentation[..., 1]
                    mask = types == mujoco.mjtObj.mjOBJ_TENDON
                    geoms = types == mujoco.mjtObj.mjOBJ_GEOM
                    mask[geoms] = model.geom_bodyid[ids[geoms]] != 0
                    rgb[~mask] = 255
                    axis = axes[row_index, body_index * 2 + view]
                    axis.imshow(rgb)
                    axis.set_axis_off()
                    axis.set_title(f'{"Original" if body_index == 0 else label} / {"Front" if view == 0 else "Side"}', fontsize=9)
        axes[row_index, 0].set_title(
            f'Original / Front\n{label} | Geometry RMS: {record["geometry_rms_percent"]:.2f}%', fontsize=9)
    _save_figure(figure, output, 'extreme_bodies')
    (output / 'extreme_samples.json').write_text(json.dumps(dict(
        selection='Maximum RMS relative change across geometry features; body_mass excluded.',
        pose='Model default pose; shared camera distance and ground-relative framing.',
        schemes=selected,
    ), indent=2, allow_nan=False) + '\n')
