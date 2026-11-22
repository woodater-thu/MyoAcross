"""Draw input perturbations schematic."""
import argparse
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    OUT = args.output_dir
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': '#26323B'})
    (W, H) = (510, 155)
    (BLUE, GREEN, TEXT) = ('#4474A7', '#788961', '#26323B')
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')

    def text(x, y, label, bold=False, color=TEXT, ha='center'):
        return ax.text(x, y, label, fontsize=7, weight='bold' if bold else 'normal', color=color, ha=ha, va='center', linespacing=1.12)

    def box(x, y, w, h, label, biased=False, policy=False):
        edge = GREEN if policy else BLUE if biased else '#59666B'
        fill = '#E7ECD9' if policy else '#E7EFF6' if biased else '#F6F6F5'
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2', edgecolor=edge, facecolor=fill, linewidth=0.75))
        text(x + w / 2, y + h / 2, label, color=edge if biased else TEXT)

    def arrow(x1, y1, x2, y2, color=TEXT):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, linewidth=0.75, color=color, shrinkA=0, shrinkB=0))
    ax.text(6, 11, 'a  Input perturbation design', fontsize=8, weight='bold', va='center')
    text(174, 27, 'Policy inputs', bold=True)
    text(323, 27, 'Policies', bold=True)
    text(451, 27, 'Evaluation', bold=True)
    rows = [(49, 'Morphology feature\nperturbation', '2%, 5%, 10%', False, True), (83, 'Reference-motion\ninput perturbation', '2.5°, 5°, 10°', True, False), (117, 'Combined\nperturbation', '2% / 2.5°; 5% / 5°\n10% / 10°', True, True)]
    for (y, title, levels, reference_bias, morphology_bias) in rows:
        text(6, y - 5, title, bold=True, ha='left')
        text(6, y + 12, levels, ha='left')
        observations = 'Observations: states + biased reference' if reference_bias else 'Observations: states + reference'
        features = 'Biased morphology features' if morphology_bias else 'Morphology features'
        box(97, y - 14, 154, 12, observations, biased=reference_bias)
        box(97, y + 2, 154, 12, features, biased=morphology_bias)
        box(275, y - 14, 96, 28, '')
        text(323, y - 8, 'Unconditioned MLP', color=BLUE)
        text(323, y, 'Concat', color=BLUE)
        text(323, y + 8, 'MyoAcross', color=GREEN)
        arrow(251, y - 8, 275, y - 8, BLUE if reference_bias else TEXT)
        arrow(251, y + 8, 275, y + 8, BLUE if morphology_bias else TEXT)
        arrow(371, y, 398, y)
    box(398, 35, 106, 96, '')
    text(451, 68, '31 held-out individual models')
    text(451, 83, '×')
    text(451, 98, '64 reference motions')
    text(6, 146, 'Feature input: Concat and MyoAcross', ha='left')
    text(504, 146, '31 models × 64 motions; 3 sign samples per nonzero level', ha='right')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in ax.texts:
        b = item.get_window_extent(renderer)
        assert b.x0 >= 0 and b.y0 >= 0 and (b.x1 <= fig.bbox.width) and (b.y1 <= fig.bbox.height)
    fig.savefig(OUT / 'figure6a.svg')
    fig.savefig(OUT / 'figure6a.pdf')
    fig.savefig(args.output_dir / 'figure6a_preview.png', dpi=300)
    print(OUT / 'figure6a.svg')

if __name__ == "__main__":
    main()
