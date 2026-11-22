"""Validate a PPO restart before constructing environments or output files."""

from collections.abc import Mapping
from pathlib import Path

from myoacross.utils.paths import PROJECT_ROOT


def _resume_path(value):
    """Resolve saved and current data paths relative to the repository root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _algorithm_settings(settings):
    """Ignore a removed loss only when it did not affect training."""
    result = dict(settings)
    if result.get("reconstruction_coef") == 0:
        result.pop("reconstruction_coef")
    return result


def validate_resume_checkpoint(
    checkpoint: Mapping, config: Mapping, world_size: int
) -> int:
    """Allow a new output name, logging, save cadence and total iteration target.

    Batch size and learning settings must match so iteration/global-step counters
    retain their meaning. Simulation/RNG state is not present in older snapshots;
    resumed training starts fresh episodes with restored algorithm state.
    """
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("Resume checkpoint must have a positive completed iteration")
    if config["training"]["num_iterations"] <= iteration:
        raise ValueError(
            "num_iterations must exceed the resume iteration (it is the total target)"
        )
    saved = checkpoint["config"]
    if "simulation" not in saved:
        raise ValueError("Resume checkpoint has no saved simulation configuration")
    if saved.get("distributed", {}).get("world_size", 1) != world_size:
        raise ValueError(
            "Resume requires the same world_size to preserve sample counts"
        )
    for field in ("network", "environment", "algorithm"):
        saved_value, current_value = saved[field], config[field]
        if field == "algorithm":
            saved_value = _algorithm_settings(saved_value)
            current_value = _algorithm_settings(current_value)
        if saved_value != current_value:
            raise ValueError(f"Resume configuration differs in {field}")
    for field in ("num_steps_per_env", "updates_per_embodiment"):
        if saved["training"][field] != config["training"][field]:
            raise ValueError(f"Resume configuration differs in training.{field}")
    if saved["training"].get("embodiments_per_rank", 1) != config["training"].get(
        "embodiments_per_rank", 1
    ):
        raise ValueError(
            "Resume configuration differs in training.embodiments_per_rank"
        )
    for field in ("embodiment", "morphology_source"):
        if saved["data"][field] != config["data"][field]:
            raise ValueError(f"Resume configuration differs in data.{field}")
    for field in (
        "dataset_version",
        "motion_set",
        "subject_ids",
        "motion_ids",
        "retargets_root",
        "bundle_root",
        "require_quality_pass",
        "virtual_reference_mode",
    ):
        saved_value, current_value = saved["data"].get(field), config["data"].get(field)
        if field in ("retargets_root", "bundle_root"):
            saved_value, current_value = (
                _resume_path(saved_value),
                _resume_path(current_value),
            )
        if saved_value != current_value:
            raise ValueError(f"Resume configuration differs in data.{field}")
    if saved["runtime"]["seed"] != config["runtime"]["seed"]:
        raise ValueError("Resume base seed must match the checkpoint")
    if saved["encoder"]["mode"] != config["encoder"]["mode"]:
        raise ValueError("Resume encoder mode differs from checkpoint")
    algorithm = checkpoint["algorithm"]
    for field in ("policy_state_dict", "optimizer_state_dict", "ppo_config"):
        if field not in algorithm:
            raise ValueError(f"Resume checkpoint is missing {field}")
    if algorithm["encoder_mode"] != config["encoder"]["mode"]:
        raise ValueError("Resume algorithm encoder mode differs from checkpoint")
    return iteration
