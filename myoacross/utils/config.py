"""Compose YAML configurations and return resolved application settings."""

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from myoacross.utils.paths import PROJECT_ROOT


def load_config(path: str | Path) -> dict:
    """Load a configuration with local and project defaults without changing cwd."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config_root = PROJECT_ROOT / "myoacross/configs"
    search_path = json.dumps(f"file://{config_root}")
    with initialize_config_dir(config_dir=str(path.parent), version_base="1.3"):
        config = compose(
            config_name=path.name,
            overrides=[f"hydra.searchpath=[{search_path}]"],
        )
        values = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    if not isinstance(values, dict):
        raise ValueError("Configuration must be a mapping")
    return values
