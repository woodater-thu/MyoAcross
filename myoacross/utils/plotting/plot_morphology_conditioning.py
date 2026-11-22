"""Render the four policy comparison schemes; requires matplotlib and numpy."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Polygon,
    FancyBboxPatch,
    FancyArrowPatch,
    Circle,
    Rectangle,
)
from matplotlib.colors import to_rgb

from myoacross.utils.config import load_config
from myoacross.utils.paths import PROJECT_ROOT

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "myoacross" / "figures"
ink = "#29353E"
edge = "#59666D"
blue = "#A7C8E1"
green = "#B4D6AF"
peach = "#F1C4A5"
purple = "#C5B6DC"
teal = "#318C83"


def draw_conditioning(ax, config, title, letter):
    """Draw one policy scheme using its resolved training configuration."""
    network = config["network"]
    mode = config["encoder"]["mode"]

    def text(x, y, t, size=10, weight="normal", color=ink, ha="center"):
        ax.text(
            x,
            y,
            t,
            ha=ha,
            va="center",
            fontsize=size,
            fontweight=weight,
            color=color,
            linespacing=1.35,
            zorder=12,
        )

    def tint(c, f):
        return tuple(min(1, v * f) for v in to_rgb(c))

    def panel(y, h, c, title, letter):
        ax.add_patch(
            FancyBboxPatch(
                (0.16, y),
                17.68,
                h,
                boxstyle="round,pad=.02,rounding_size=.16",
                fc=c,
                ec="#CAD1D2",
                lw=1.1,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                (0.53, y + h - 0.30),
                0.18,
                color={
                    "a": "#C3D8E7", "b": "#C3D8E7",
                    "c": "#C5DEBE", "d": "#D1C3E4",
                }[letter],
                zorder=1,
            )
        )
        text(0.53, y + h - 0.30, letter, 13, "bold")
        text(0.84, y + h - 0.30, title, 14, "bold", ha="left")

    def arrow(a, b, c=ink, lw=1.35):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=12, color=c, lw=lw, zorder=9
            )
        )

    def route(pts, c=ink, ls="-"):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1], ys[:-1], color=c, lw=1.25, ls=ls, zorder=8)
        arrow(pts[-2], pts[-1], c)

    def slab(x, y, w=0.24, h=0.92, c=blue, dx=0.18, dy=0.18):
        ax.add_patch(
            Polygon(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                fc=c,
                ec=edge,
                lw=0.85,
                zorder=5,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (x, y + h),
                    (x + dx, y + h + dy),
                    (x + w + dx, y + h + dy),
                    (x + w, y + h),
                ],
                fc=tint(c, 1.10),
                ec=edge,
                lw=0.85,
                zorder=5,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (x + w, y),
                    (x + w + dx, y + dy),
                    (x + w + dx, y + h + dy),
                    (x + w, y + h),
                ],
                fc=tint(c, 0.84),
                ec=edge,
                lw=0.85,
                zorder=5,
            )
        )

    def stack(x, y, label, h=0.90, colors=(blue, blue, blue), width=0.2, gap=0.26):
        for k, c in enumerate(colors):
            slab(x + k * gap, y + k * 0.025, width, h, c)
        text(x + 0.48, y - 0.27, label, 10, "bold")

    def prism(x, y, label, c=green):
        ax.add_patch(
            Polygon(
                [
                    (x, y - 0.46),
                    (x, y + 0.46),
                    (x + 0.76, y + 0.28),
                    (x + 0.76, y - 0.28),
                ],
                fc=c,
                ec=edge,
                lw=1.05,
                zorder=5,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (x, y + 0.46),
                    (x + 0.17, y + 0.65),
                    (x + 0.94, y + 0.47),
                    (x + 0.76, y + 0.28),
                ],
                fc=tint(c, 1.10),
                ec=edge,
                lw=1.05,
                zorder=5,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (x + 0.76, y - 0.28),
                    (x + 0.94, y - 0.09),
                    (x + 0.94, y + 0.47),
                    (x + 0.76, y + 0.28),
                ],
                fc=tint(c, 0.82),
                ec=edge,
                lw=1.05,
                zorder=5,
            )
        )
        text(x + 0.47, y - 0.78, label, 10, "bold")

    def vector(x, y, n=8, c=peach, w=0.95, label=None):
        for j in range(n):
            ax.add_patch(
                Rectangle(
                    (x + j * w / n, y - 0.19),
                    w / n,
                    0.38,
                    fc=tint(c, 0.9 + 0.13 * (j % 3)),
                    ec="white",
                    lw=0.6,
                    zorder=6,
                )
            )
        ax.add_patch(
            Rectangle((x, y - 0.19), w, 0.38, fill=False, ec=edge, lw=0.8, zorder=7)
        )
        if label:
            text(x + w / 2, y - 0.44, label, 10)

    def obs(x, y):
        for i in range(3):
            xx = x + i * 0.22
            ax.add_patch(
                FancyBboxPatch(
                    (xx, y - 0.49 + i * 0.035),
                    0.72,
                    0.87,
                    boxstyle="round,pad=.02,rounding_size=.045",
                    fc="white",
                    ec=edge,
                    lw=0.8,
                    zorder=3 + i,
                )
            )
            t = np.linspace(0, 0.56, 50)
            ax.plot(
                xx + 0.08 + t,
                y - 0.11 + i * 0.035 + 0.15 * np.sin(t * 15 + i),
                color=["#6898C4", "#8AA77A", "#BE9B72"][i],
                lw=1.5,
                zorder=6 + i,
            )
        text(x + 0.62, y - 0.77, "State + reference", 10, "bold")
        text(x + 0.62, y - 0.99, r"$o_t$", 10)

    def body(x, y, scale=1, c=teal, leg_scale=1, width_scale=1):
        # A human figure, not a robot architecture or an image placeholder.
        pts = {
            "neck": (0, 0.33),
            "hip": (0, -0.07),
            "ls": (-0.15, 0.29),
            "rs": (0.15, 0.29),
            "le": (-0.23, 0.08),
            "re": (0.23, 0.08),
            "lw": (-0.27, -0.12),
            "rw": (0.27, -0.12),
            "lh": (-0.08, -0.08),
            "rh": (0.08, -0.08),
            "lk": (-0.11, -0.34),
            "rk": (0.11, -0.34),
            "la": (-0.12, -0.60),
            "ra": (0.12, -0.60),
        }
        pts = {
            name: (
                px * width_scale,
                -0.08 + (py + 0.08) * leg_scale if py < -0.08 else py,
            )
            for name, (px, py) in pts.items()
        }
        for a, b in [
            ("neck", "hip"),
            ("ls", "rs"),
            ("ls", "le"),
            ("le", "lw"),
            ("rs", "re"),
            ("re", "rw"),
            ("lh", "rh"),
            ("lh", "lk"),
            ("lk", "la"),
            ("rh", "rk"),
            ("rk", "ra"),
        ]:
            p, q = pts[a], pts[b]
            ax.plot(
                [x + p[0] * scale, x + q[0] * scale],
                [y + p[1] * scale, y + q[1] * scale],
                color=c,
                lw=2.1,
                solid_capstyle="round",
                zorder=5,
            )
        ax.add_patch(
            Circle(
                (x, y + 0.47 * scale),
                0.10 * scale,
                fc=tint(c, 1.35),
                ec=c,
                lw=0.8,
                zorder=5,
            )
        )

    def morphology(x, y):
        body(x, y, 0.77, "#8A9E8D")
        body(x + 0.49, y, 0.93, teal)
        body(x + 0.99, y, 0.82, "#88A398")
        text(x + 0.49, y - 0.83, "Body morphology", 10, "bold")

    def norm(x, y):
        ax.add_patch(
            FancyBboxPatch(
                (x, y - 0.32),
                1.4,
                0.64,
                boxstyle="round,pad=.025,rounding_size=.065",
                fc="#F8FAED",
                ec="#99AC8F",
                lw=0.9,
                zorder=4,
            )
        )
        text(x + 0.7, y, "Log + normalize", 9)

    def output(x, y):
        # The path represents separate actor and critic backbones.
        route([(x, y), (x + 0.26, y), (x + 0.26, y + 0.31), (x + 0.54, y + 0.31)])
        route([(x + 0.26, y), (x + 0.26, y - 0.31), (x + 0.54, y - 0.31)])
        for yy, cc, lab in [
            (y + 0.31, peach, r"$\pi(a_t\mid o_t,m)$"),
            (y - 0.31, blue, r"$V(o_t,m)$"),
        ]:
            for j in range(3):
                ax.add_patch(
                    Circle(
                        (x + 0.66 + j * 0.22, yy),
                        0.078,
                        fc=cc,
                        ec=edge,
                        lw=0.65,
                        zorder=10,
                    )
                )
            text(x + 1.6, yy, lab, 11, ha="left")
        text(x + 1.35, y - 0.75, "Actor / critic outputs", 10, "bold")

    def main_path(y, film=False, concat=False, unconditioned=False):
        obs(0.65, y)
        arrow((2.02, y), (2.5, y))
        prism(2.65, y, "Observation encoders", blue)
        arrow((3.7, y), (4.30, y))
        vector(4.43, y, label=r"$h_t$", c=blue)
        if concat:
            arrow((5.50, y), (6.01, y))
            ax.add_patch(
                Circle((6.25, y), 0.22, fc="#F8FCF1", ec=teal, lw=1.3, zorder=10)
            )
            text(6.25, y, r"$\Vert$", 17, color=teal)
            text(6.25, y - 0.48, "Concat", 10, "bold", color=teal)
            arrow((6.50, y), (7.11, y))
            vector(7.24, y, n=5, c=blue, w=0.7)
            vector(7.94, y, n=3, c=green, w=0.42)
            text(7.80, y - 0.44, r"$[h_t;e_m]$", 10)
            arrow((8.50, y), (9.08, y))
        else:
            arrow((5.5, y), (9.08, y))
            text(7.22, y + 0.21, "Dynamic feature stream", 9, color="#78858B")
        stack(
            9.24, y - 0.44, f"Input projection\n{network['hidden_dim']}",
            h=0.86, colors=(blue, blue),
        )
        arrow((10.00, y), (10.61, y))
        stack(
            10.78,
            y - 0.44,
            "FiLM residual\nblocks" if film else "Residual MLP\nblocks",
            h=0.86,
            colors=(purple, purple, purple) if film else (blue, blue, blue),
        )
        text(11.25, y + 0.84, f"× {network['num_blocks']}", 11, "bold")
        arrow((11.72, y), (12.30, y))
        stack(
            12.44, y - 0.37,
            "MLP heads\n" + " → ".join(map(str, network["head_hidden_dims"])),
            h=0.72, colors=(peach, peach), gap=0.25,
        )
        arrow((13.22, y), (13.67, y))
        output(13.67, y)
        if unconditioned:
            # Replace conditional output notation with the actual unconditional inputs.
            for artist in list(ax.texts):
                if (
                    artist.get_position()[0] > 15
                    and abs(artist.get_position()[1] - y) < 0.5
                ):
                    artist.set_text(artist.get_text().replace(",m", ""))

    color = {"none": "#F1F6FA", "concat": "#F2F7EC", "rl": "#F7F2FA"}[mode]
    panel(0.10, 4.25, color, title, letter)
    baseline = config["data"]["embodiment"] == "baseline"
    training_bodies = (
        "Original MyoFullBody"
        if baseline
        else "Segment-scale subjects + virtual bodies"
    )
    text(0.84, 3.72, f"Training bodies: {training_bodies}", 10, ha="left")
    y = 1.40
    main_path(
        y, film=mode == "rl", concat=mode == "concat", unconditioned=mode == "none",
    )

    if mode == "none":
        if baseline:
            body(1.25, 2.95, 0.80, "#6898C4")
            text(1.25, 2.26, "One standard model", 10, "bold")
            arrow((1.75, 2.95), (7.55, 2.95))
        else:
            for x, scale, legs, width in (
                (0.85, 0.72, 0.90, 1.08),
                (1.55, 0.80, 1.00, 1.00),
                (2.25, 0.87, 1.08, 0.92),
            ):
                body(x, 2.95, scale, teal, legs, width)
            text(1.55, 2.26, "Subject models", 10, "bold")
            arrow((2.65, 2.95), (3.55, 2.95), teal)
            text(3.10, 3.28, "Perturb", 9, color=teal)
            for x, scale, legs, width in (
                (3.95, 0.75, 1.06, 0.92),
                (4.65, 0.82, 0.92, 1.10),
                (5.35, 0.85, 1.02, 1.02),
            ):
                body(x, 2.95, scale, "#88A398", legs, width)
            text(4.65, 2.26, "Virtual variants", 10, "bold")
            arrow((5.85, 2.95), (7.55, 2.95), teal)
            route([(2.65, 2.95), (2.85, 2.95), (2.85, 3.50),
                   (6.65, 3.50), (6.65, 2.95), (7.55, 2.95)], teal)
        ax.add_patch(FancyBboxPatch(
            (7.65, 2.61), 3.0, 0.68,
            boxstyle="round,pad=.025,rounding_size=.065",
            fc="white", ec=edge, lw=0.9,
        ))
        text(9.15, 2.95, "Training environments", 11, "bold")
        route([(9.15, 2.61), (9.15, 2.03), (1.25, 2.03),
               (1.25, y + 0.50)], "#6898C4")
        text(13.70, 2.95, "No morphology input\nModels supply simulated observations", 11,
             color="#55788F")
    else:
        morphology(0.97, 2.90)
        arrow((2.27, 2.97), (2.60, 2.97), teal)
        norm(2.74, 2.97)
        arrow((4.28, 2.97), (4.66, 2.97), teal)
        vector(4.78, 2.97, c=green, w=0.87, label=r"$m\in\mathbb{R}^{18}$")
        arrow((5.78, 2.97), (6.25, 2.97), teal)
        if mode == "concat":
            widths = network["group_hidden_dims"]["morphology"]
            encoder_label = "Observation encoder\n" + " → ".join(map(str, widths))
            encoded_label = rf"$e_m\in\mathbb{{R}}^{{{widths[-1]}}}$"
        else:
            encoder = config["morphology_encoder"]
            widths = [*encoder["hidden_dims"], encoder["latent_dim"]]
            encoder_label = "Morphology encoder\n" + " → ".join(map(str, widths))
            encoded_label = rf"$z_m\in\mathbb{{R}}^{{{widths[-1]}}}$"
        prism(6.38, 2.97, encoder_label, green)
        arrow((7.45, 2.97), (7.90, 2.97), teal)
        vector(8.03, 2.97, c=green, w=0.9, label=encoded_label)
        if mode == "concat":
            route([
                (9.07, 2.97), (9.22, 2.97), (9.22, 2.03),
                (6.25, 2.03), (6.25, y + 0.25),
            ], teal)
            text(
                13.9, 2.97, "Separate actor / critic\nmorphology group encoders",
                11, color="#49735A",
            )
        else:
            arrow((9.07, 2.97), (9.55, 2.97), teal)
            stack(
                9.72, 2.67, "FiLM generators\n" + str(network["generator_hidden_dim"]),
                h=0.58, colors=(purple, purple), gap=0.24,
            )
            route([(10.46, 2.97), (11.25, 2.97), (11.25, y + 0.59)], "#8D6CAD")
            text(11.68, 3.24, r"$\Delta\gamma_\ell,\,\beta_\ell$", 11, color="#765597")
            text(
                14.81, 2.97,
                r"$\widetilde{h}_\ell=(1+\Delta\gamma_\ell)\odot\mathrm{LN}(h_\ell)+\beta_\ell$",
                11,
            )
            text(14.81, 2.56, "Shared morphology encoder", 10, color="#765597")
    text(
        9, 0.25,
        "Separate actor / critic backbones · PPO · Evaluate on held-out segment-scale subjects",
        9, color="#617079",
    )


def render_conditioning(output_dir: Path) -> None:
    """Save the architecture figure in PNG, PDF, and SVG formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )
    config_root = PROJECT_ROOT / "myoacross/configs"
    schemes = (
        ("experiments/policy_comparison/train_myofullbody_mlp.yaml", "MyoFullBody MLP"),
        ("experiments/policy_comparison/train_segment_mlp.yaml", "Segment MLP"),
        ("experiments/policy_comparison/train_segment_concat.yaml", "Segment Concat"),
        ("rl/train.yaml", "Segment FiLM"),
    )
    fig, axes = plt.subplots(4, 1, figsize=(18, 18))
    fig.patch.set_facecolor("white")
    for ax, (config_path, title), letter in zip(axes, schemes, "abcd"):
        ax.set(xlim=(0, 18), ylim=(0, 4.5))
        ax.axis("off")
        draw_conditioning(ax, load_config(config_root / config_path), title, letter)
    fig.subplots_adjust(left=0.006, right=0.994, top=0.994, bottom=0.009, hspace=0.02)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            output_dir / f"morphology-conditioning.{ext}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )

    # Keep the generated SVG clean in Git diffs.
    svg = output_dir / "morphology-conditioning.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    render_conditioning(args.output_dir)


if __name__ == "__main__":
    main()
