"""Draw image workflow schematic."""
import argparse
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import base64, html, math
    R = args.assets_dir
    INK = '#29353E'
    BLUE = '#416F97'
    GREEN = '#73865B'
    GRAY = '#697078'
    S = ['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="510pt" height="252pt" viewBox="0 0 510 252">']

    def text(x, y, t, size=7, bold=False, color=INK, anchor='middle'):
        S.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{(700 if bold else 400)}" text-anchor="{anchor}" fill="{color}">{html.escape(t)}</text>')

    def box(x, y, w, h, fill='#F4F4F2', edge=GRAY):
        S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{edge}" stroke-width="0.75"/>')

    def arrow(points, c=INK):
        (x, y) = points[-1]
        (p, q) = points[-2]
        a = math.atan2(y - q, x - p)
        L = 4
        W = 1.7
        points = points[:-1] + [(x - L * math.cos(a), y - L * math.sin(a))]
        S.append(f'''<polyline points="{' '.join((f'{p},{q}' for (p, q) in points))}" fill="none" stroke="{c}" stroke-width="0.75" stroke-linejoin="round"/>''')
        v = [(x, y), (x - L * math.cos(a) + W * math.sin(a), y - L * math.sin(a) - W * math.cos(a)), (x - L * math.cos(a) - W * math.sin(a), y - L * math.sin(a) + W * math.cos(a))]
        S.append(f'''<polygon points="{' '.join((f'{p},{q}' for (p, q) in v))}" fill="{c}"/>''')

    def img(name, x, y, w, h):
        b = base64.b64encode((R / 'assets' / name).read_bytes()).decode()
        S.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" xlink:href="data:image/png;base64,{b}"/>')
    text(6, 12, 'a  Image-to-simulation workflow', 8, True, anchor='start')
    for (x, label) in [(40, 'Single-image input'), (132, 'Estimated SKEL'), (268, 'Individual model'), (463, 'Simulation')]:
        text(x, 34, label, bold=True)
    img('demo_input_anonymized.png', 9, 46, 62, 110)
    img('demo_skel_overlay.png', 101, 46, 62, 110)
    box(238, 43, 59, 115, '#FFFFFF', '#B9B9B5')
    box(433, 43, 60, 115, '#FFFFFF', '#B9B9B5')
    img('demo_model.png', 244, 49, 48, 107)
    img('demo_policy_simulation.png', 438, 49, 50, 107)
    arrow([(72, 93), (99, 93)])
    text(85, 83, 'HSMR')
    arrow([(164, 93), (238, 93)])
    text(202, 73, 'Skeletal measurements')
    text(202, 83, 'and regional scaling')
    box(228, 164, 80, 19, '#F2E9DA', '#9A8971')
    text(268, 176.5, 'Recorded mass: 67 kg')
    arrow([(268, 164), (268, 157)])
    box(8, 211, 76, 29)
    text(46, 222, '64 source motions')
    text(46, 232, 'across 8 categories')
    box(111, 211, 79, 29)
    text(150.5, 228, 'Motion retargeting')
    box(217, 211, 79, 29)
    text(256.5, 222, 'Model-specific')
    text(256.5, 232, 'reference motions')
    arrow([(84, 225), (111, 225)])
    arrow([(190, 225), (217, 225)])
    arrow([(238, 147), (220, 147), (220, 194), (150, 194), (150, 211)])
    box(319, 164, 79, 21, '#D7E2C4', GREEN)
    text(358.5, 177, 'Morphology features', color=GREEN)
    arrow([(297, 132), (358, 132), (358, 164)], GREEN)
    box(319, 211, 79, 29, '#DCEAF4', BLUE)
    text(358.5, 222, 'Observations')
    text(358.5, 232, 'States + reference', color=BLUE)
    arrow([(296, 225), (319, 225)], BLUE)
    arrow([(433, 145), (418, 145), (418, 198), (358, 198), (358, 211)], BLUE)
    text(458, 195, 'Musculoskeletal states', color=BLUE)
    box(423, 211, 79, 29, '#F2F6EE', GREEN)
    text(462.5, 222, 'Pretrained')
    text(462.5, 232, 'MyoAcross policy', bold=True, color=GREEN)
    arrow([(398, 225), (423, 225)], BLUE)
    arrow([(398, 174.5), (462, 174.5), (462, 211)], GREEN)
    arrow([(502, 225), (507, 225), (507, 111), (493, 111)])
    S.append(f'<text x="502" y="163" transform="rotate(-90 502 163)" font-family="Arial" font-size="7" text-anchor="middle" fill="{INK}">Muscle excitations</text>')
    S.append('</svg>')
    (args.output_dir / 'figure7a.svg').write_text('\n'.join(S))
    print(args.output_dir / 'figure7a.svg')

if __name__ == "__main__":
    main()
