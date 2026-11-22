"""
Unified entry point for:
- virtual embodiments generation,
- RL
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from myoacross.utils.config import load_config

from myoacross.utils.paths import (
    PROJECT_ROOT,
    resolve_path,
)


def run_generate(config: dict) -> list[Path]:
    """Dispatch the generation configuration to its runner."""
    from .runners.generate_embodiment_bundles import run_generation

    return run_generation(config)


def run_train(config: dict, config_dir: Path, *, resume: Path | None = None) -> Path:
    """Dispatch the training configuration to its runner."""
    from .runners.train_rl import run_training

    return run_training(config, config_dir, resume=resume)


def run_evaluate(config: dict) -> Path:
    """Resolve evaluation paths and dispatch to the policy evaluator."""
    from myoacross.rl.runners.evaluate import evaluate_policy

    options = dict(config)
    for name in ("checkpoint_path", "output_dir"):
        path = Path(options[name]).expanduser()
        options[name] = path if path.is_absolute() else PROJECT_ROOT / path
    if options.get("retargets_root") is not None:
        path = Path(options["retargets_root"]).expanduser()
        options["retargets_root"] = (
            path if path.is_absolute() else PROJECT_ROOT / path
        )
    return evaluate_policy(**options)


def _prepare_task_backend(
    task: str, config: dict, config_path: Path, *, resume: Path | None = None
) -> None:
    """Load an isolated adapter before importing multi-body task implementations."""
    needs_backend = task == "evaluate" or (
        task == "train" and config.get("training", {}).get("embodiments_per_rank", 1) > 1
    )
    if not needs_backend:
        return
    import importlib.util
    import sys
    from myoacross.rl.runners.multi_embodiment.backend import MARKER, run_with_backend

    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or spec.origin is None:
        raise RuntimeError("mujoco_warp is not installed")
    if MARKER in (Path(spec.origin).parent / "_src/sensor.py").read_text():
        return
    command = [sys.executable, "-m", "myoacross", "rl", task, "--config", str(config_path)]
    if resume is not None:
        command.extend(["--resume", str(resume.expanduser().resolve())])
    raise SystemExit(run_with_backend(command, PROJECT_ROOT / "artifacts/multi_embodiment_backend"))


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and dispatch one task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=("generate", "train", "evaluate"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the task's YAML configuration",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Full RL checkpoint to continue; use a new experiment name and a total iteration target",
    )
    args = parser.parse_args(argv)
    if args.resume is not None and args.task != "train":
        parser.error("--resume is supported only for train")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and args.task != "train":
        parser.error("torchrun is supported only for the train task")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    # All public configuration data and output paths are relative to the repository.
    for key in ("bundle_root", "retargets_root"):
        if key in config.get("data", {}):
            config["data"][key] = str(resolve_path(config["data"][key], PROJECT_ROOT))
    _prepare_task_backend(args.task, config, config_path, resume=args.resume)

    if args.task == "generate":
        index_paths = run_generate(config)
        print("Generated bundle index paths:")
        for path in index_paths:
            print(f"  - {path}")

    elif args.task == "evaluate":
        output_dir = run_evaluate(config)
        print(f"Saved evaluation results: {output_dir}")

    elif args.task == "train":
        checkpoint_path = run_train(config, config_dir, resume=args.resume)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Saved RL checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
