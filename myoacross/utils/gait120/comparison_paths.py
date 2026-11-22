"""Paths for Gait120 model comparisons."""

import argparse
from pathlib import Path
from myoacross.utils.paths import PROJECT_ROOT


def parse_analysis_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data/experiments/gait120/model_comparison",
    )
    parser.add_argument(
        "--gait-root", type=Path, default=PROJECT_ROOT / "data/experiments/gait120"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.gait_root = args.gait_root.resolve()
    args.output = args.output.resolve()
    return args
