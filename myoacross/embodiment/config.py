"""Load and validate an embodiment configuration."""

from pathlib import Path

from myoacross.utils.config import load_config as compose_config


def load_config(config_path: str | Path) -> dict:
    """Load an embodiment configuration."""

    config = compose_config(config_path)

    # Validate the field of config
    unknown = set(config) - {"name", "optimization"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {list(unknown)}")

    strategy = config.get("name")
    if not isinstance(strategy, str) or strategy not in {
        "baseline",
        "height_scale",
        "segment_scale",
    }:
        raise ValueError(f"Invalid embodiment strategy: {strategy!r}")

    # Validate the optimization fields for each fitting strategy
    if strategy in {"segment_scale", "height_scale"}:
        optimization = config.get("optimization", {})
        if not isinstance(optimization, dict):
            raise ValueError("optimization must be a mapping")

        allowed = {
            "scale_bounds",
            "prior_weight",
            "sigma_symmetry",
            "sigma_adjacency",
            "max_normalized_measurement_error",
            "max_nfev",
            "measurement_tolerances",
        }
        if strategy == "height_scale":
            allowed -= {
                "sigma_symmetry",
                "sigma_adjacency",
                "max_normalized_measurement_error",
            }
        unknown = set(optimization) - allowed
        if unknown:
            raise ValueError(f"Unknown optimization options: {list(unknown)}")

        config["optimization"] = optimization

    elif "optimization" in config:
        raise ValueError(f"{strategy} does not support optimization settings")

    return config
