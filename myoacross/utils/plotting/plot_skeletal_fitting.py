"""Render existing analysis results."""
from pathlib import Path
import numpy as np
METHOD_STYLES = (
    ("baseline", "MyoFullBody", "#BBBBBB"),
    ("height_scale", "Global scale", "#4477AA"),
    ("segment_scale", "Segment scale", "#228833"),
)


def _save_figure(figure, output: Path, stem: str) -> None:
    """Export an editable figure and its print and preview files."""
    import matplotlib.pyplot as plt

    for extension in ("svg", "pdf", "tiff", "png"):
        figure.savefig(
            output / f"{stem}.{extension}",
            dpi=300 if extension == "png" else 600,
            bbox_inches="tight",
            facecolor="white",
            **({"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}),
        )
    plt.close(figure)


def plot_fitting_results(names, rows, summary, output: Path) -> None:
    """Show subject distributions, measurement errors, and paired improvements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scienceplots  # noqa: F401 -- register Matplotlib styles

    with plt.style.context(["science", "nature", "no-latex", "bright"]), plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    }):
        values = [
            np.array([row["mae_m"] for row in rows if row["method"] == method]) * 1000
            for method, _, _ in METHOD_STYLES
        ]
        figure, axis = plt.subplots(figsize=(4.2, 3.3), layout="constrained")
        boxes = axis.boxplot(values, positions=range(3), widths=0.45,
                             patch_artist=True, showfliers=False)
        jitter = np.random.default_rng(42)
        for index, ((_, _, color), samples) in enumerate(zip(METHOD_STYLES, values)):
            boxes["boxes"][index].set(facecolor=color, alpha=0.25)
            boxes["medians"][index].set(color="black", linewidth=1.2)
            axis.scatter(index + jitter.uniform(-0.12, 0.12, len(samples)), samples,
                         s=10, color=color, alpha=0.55, edgecolors="none", zorder=3)
        axis.set_xticks(range(3), [label for _, label, _ in METHOD_STYLES])
        axis.set_ylabel("Subject MAE (mm)")
        axis.set_ylim(bottom=0)
        axis.set_title(f"17 measurements per subject; n = {len(values[0])}", fontsize=8)
        _save_figure(figure, output, "subject_mae")

        figure, axis = plt.subplots(figsize=(5.6, 4.7), layout="constrained")
        positions = np.arange(len(names))
        for offset, (method, label, color) in zip((-0.18, 0.0, 0.18), METHOD_STYLES):
            errors = summary["all"][method]["per_measurement_mae_m"]
            axis.scatter([errors[name] * 1000 for name in names], positions + offset,
                         s=27, color=color, label=label)
        labels = []
        for name in names:
            side = ""
            if name.endswith(("_r", "_l")):
                side = f" ({name[-1].upper()})"
                name = name[:-2]
            labels.append(name.replace("_", " ").capitalize() + side)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(left=0)
        axis.set_xlabel("Mean absolute error (mm)")
        axis.set_title(f"Equally weighted subjects; n = {len(values[0])}", fontsize=8)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
        _save_figure(figure, output, "measurement_mae")

        global_errors = {row["subject"]: row["mae_m"] for row in rows
                         if row["method"] == "height_scale"}
        regional_errors = {row["subject"]: row["mae_m"] for row in rows
                           if row["method"] == "segment_scale"}
        subjects = sorted(global_errors)
        x = np.array([global_errors[subject] for subject in subjects]) * 1000
        y = np.array([regional_errors[subject] for subject in subjects]) * 1000
        figure, axis = plt.subplots(figsize=(3.8, 3.8), layout="constrained")
        axis.scatter(x, y, s=18, color=METHOD_STYLES[2][2], alpha=0.65, edgecolors="none")
        limit = max(x.max(), y.max()) * 1.05
        axis.plot([0, limit], [0, limit], linestyle="--", color="#BBBBBB", linewidth=0.8)
        axis.set(xlim=(0, limit), ylim=(0, limit),
                 xlabel="Global scale subject MAE (mm)",
                 ylabel="Segment scale subject MAE (mm)")
        axis.set_aspect("equal")
        improved = int(np.count_nonzero(y < x))
        axis.set_title(f"Lower segment MAE: {improved}/{len(subjects)} subjects "
                       f"({100 * improved / len(subjects):.1f}%)", fontsize=8)
        _save_figure(figure, output, "paired_subject_mae")
