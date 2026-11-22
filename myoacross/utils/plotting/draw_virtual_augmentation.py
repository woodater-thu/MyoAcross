"""Draw virtual augmentation schematic."""
import argparse
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    OUT = args.output_dir
    INK = '#26323B'
    GREEN = '#788961'
    RED = '#AC3028'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'text.color': INK})
    fig = plt.figure(figsize=(510 / 72, 180 / 72), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 510), ylim=(0, 180))
    ax.axis('off')

    def text(x, y, label, size=7, bold=False, ha='center', color=INK):
        ax.text(x, y, label, fontsize=size, fontweight='bold' if bold else 'normal', ha=ha, va='center', color=color)

    def box(x, y, w, h, fill='#F7F8F5', edge=GREEN):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=3', facecolor=fill, edgecolor=edge, linewidth=0.75))

    def arrow(x1, y1, x2, y2, color=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, linewidth=0.75, color=color, shrinkA=0, shrinkB=0))

    def line(x1, y1, x2, y2, color=INK):
        ax.plot([x1, x2], [y1, y2], color=color, lw=0.75, solid_capstyle='butt')
    text(6, 171, 'a  Augmentation experiments', 8, True, 'left')
    text(134, 153, 'Training models', 7, True)
    text(299, 153, 'Policy inputs and conditioning', 7, True)
    text(455.5, 153, 'Shared evaluation', 7, True)
    text(6, 137, 'Virtual model augmentation', 7, True, 'left')
    text(6, 70, 'Asymmetry-enhanced training', 7, True, 'left')
    rows = [118, 90, 50, 22]
    labels = ['V0', 'Standard', 'Standard', 'Enhanced']
    training = ['110 original models', '110 original + 1100 virtual models', '110 original + 1100 virtual models', '110 original + 1100 virtual models']
    notes = ['', 'Standard sampling', 'Standard sampling', 'Enhanced asymmetry sampling']
    for (row, (y, label, model, note)) in enumerate(zip(rows, labels, training, notes)):
        text(6, y, label, 7, True, 'left', GREEN)
        box(70, y - 11, 128, 22, '#FCF7F7', RED)
        if note:
            text(134, y + 4, model, 7)
            text(134, y - 5, note, 7)
        else:
            text(134, y, model, 7)
        arrow(198, y, 221, y)
        box(221, y - 11, 156, 22, '#F2F5EC', GREEN)
        text(226, y + 4, 'musculoskeletal model states', 7, ha='left')
        text(226, y - 5, '+ morphology feature', 7, ha='left')
        arrow(330, y + 4, 341, y + 4, GREEN)
        arrow(330, y - 5, 341, y - 5, GREEN)
        box(341, y - 7, 31, 14, '#D7E2C4', GREEN)
        text(356.5, y, 'FiLM', 7)
        line(377, y, 393, y)
    for (lo, hi, center) in [(90, 118, 104), (22, 50, 36)]:
        line(393, lo, 393, hi)
        arrow(393, center, 407, center)
    box(407, 76, 97, 56, '#F5F5F3', '#59666D')
    for (y, label) in [(120, '31 held-out models'), (107, '64 reference motions/model'), (94, '1984 trials'), (81, 'Matching reference motions')]:
        text(455.5, y, label, 7)
    box(407, 8, 97, 56, '#F5F5F3', '#59666D')
    for (y, label) in [(55, '31 held-out models'), (43, '9 shortening conditions'), (31, '64 reference motions/model'), (19, 'Original-model references')]:
        text(455.5, y, label, 7)
    fig.savefig(OUT / 'figure5a.svg')
    fig.savefig(OUT / 'figure5a.pdf')
    fig.savefig(args.output_dir / 'figure5a_preview.png', dpi=240)
    print(OUT / 'figure5a.svg')

if __name__ == "__main__":
    main()
