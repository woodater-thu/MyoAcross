"""Draw policy comparison schematic."""
import argparse
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import html
    R = args.output_dir
    INK = '#29353E'
    BLUE = '#416F97'
    GREEN = '#73865B'
    RED = '#AC3028'
    S = ['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="510pt" height="144pt" viewBox="0 0 510 144">', '<rect width="510" height="144" fill="white"/>']

    def text(x, y, t, bold=False, anchor='middle', color=INK, size=8):
        S.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{(700 if bold else 400)}" text-anchor="{anchor}" fill="{color}">{html.escape(t)}</text>')

    def rect(x, y, w, h, fill, edge, rx=3):
        S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{edge}" stroke-width="0.75"/>')

    def line(points, c=INK, arrow=False):
        S.append(f'''<polyline points="{' '.join((f'{x},{y}' for (x, y) in points))}" fill="none" stroke="{c}" stroke-width="0.75" stroke-linejoin="round"/>''')
        if arrow:
            (x, y) = points[-1]
            (px, py) = points[-2]
            import math
            a = math.atan2(y - py, x - px)
            l = 4
            w = 1.8
            pts = [(x, y), (x - l * math.cos(a) + w * math.sin(a), y - l * math.sin(a) - w * math.cos(a)), (x - l * math.cos(a) - w * math.sin(a), y - l * math.sin(a) + w * math.cos(a))]
            S.append(f'''<polygon points="{' '.join((f'{a},{b}' for (a, b) in pts))}" fill="{c}"/>''')
    text(6, 12, 'a  Policy comparisons', True, 'start', size=9)
    text(140, 30, 'Training models', True, size=9)
    text(295, 30, 'Policy inputs and conditioning', True, size=9)
    text(453, 30, 'Shared evaluation', True, size=9)
    rows = [49, 76, 103, 130]
    for (y, label) in zip(rows, ['Template MLP', 'Unconditioned MLP', 'Concat', 'MyoAcross']):
        text(6, y + 3, label, True, 'start', GREEN if label == 'MyoAcross' else INK)
    rect(87, 39, 106, 20, '#FBF7F7', RED)
    text(140, 52, 'Original MyoFullBody')
    rect(87, 66, 106, 74, '#FBF7F7', RED)
    text(140, 106, 'Individual + virtual models')
    for y in rows:
        line([(193, y), (204, y)], INK, True)
    for (j, y) in enumerate(rows):
        color = BLUE if j < 3 else GREEN
        rect(204, y - 10, 182, 20, '#F0F5F9' if j < 3 else '#F2F6EE', color)
        if j < 2:
            text(211, y + 3, 'Observations', anchor='start')
            line([(264, y), (349, y)], BLUE, True)
        else:
            text(211, y + 3, 'Observations + morphology features', anchor='start')
            line([(339, y), (349, y)], color, True)
        rect(350, y - 7, 31, 14, '#DCEAF4' if j < 3 else '#D7E2C4', color, 2)
        text(365.5, y + 3, 'MLP' if j < 3 else 'FiLM')
    for y in rows:
        line([(386, y), (392, y)], INK)
    line([(392, 49), (392, 130)], INK)
    line([(392, 89), (399, 89)], INK, True)
    rect(399, 39, 109, 101, '#F5F5F3', '#59666D')
    for (y, t) in [(57, '31 unseen individual models'), (80, '64 reference motions/model'), (106, 'Fixed policy'), (129, 'No individual fine-tuning')]:
        text(453.5, y, t)
    S.append('</svg>')
    (R / 'figure3a.svg').write_text('\n'.join(S))
    print(R / 'figure3a.svg')

if __name__ == "__main__":
    main()
