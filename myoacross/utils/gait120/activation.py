"""Summarize complete-cycle activation and its body-mass associations."""

import pandas as pd
from .comparison_paths import parse_analysis_args
from . import muscle_statistics, body_mass


def run(root, gait_root, output):
    output.mkdir(parents=True, exist_ok=False)
    muscle_statistics.run(root, gait_root, output / "per_muscle")
    totals = output / "summed_activation"
    totals.mkdir()
    for name, keys in [
        ("subjects", ["condition", "subject"]),
        ("motions", ["condition", "subject", "motion"]),
    ]:
        data = pd.read_csv(output / "per_muscle" / f"{name}.csv")
        data.groupby(keys).agg(
            mass=("mass", "first"),
            height=("height", "first"),
            speed=("speed", "mean"),
            activation_sum11=("activation", "sum"),
        ).reset_index().to_csv(totals / f"{name}.csv", index=False)
    body_mass.run(totals, gait_root, output / "body_mass")


if __name__ == "__main__":
    args = parse_analysis_args(__doc__)
    run(args.root, args.gait_root, args.output)
