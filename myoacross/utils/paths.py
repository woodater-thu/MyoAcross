from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path of myoacross assets
ASSETS_DIR = PROJECT_ROOT / "data/myoacross/assets"

# Path of SMPL models
SKEL_MODEL_DIR  = PROJECT_ROOT / "data/models/skel"
SMPLH_MODEL_DIR = PROJECT_ROOT / "data/models/smplh"

# Path of dataset
DATASET_ROOT            = PROJECT_ROOT / "data/dataset"
KIT_SUBJECTS_PATH       = PROJECT_ROOT / "data/dataset/KIT/subjects.json"
BMLMOVI_SUBJECTS_PATH   = PROJECT_ROOT / "data/dataset/BMLmovi/subjects.json"

# Path of intermediate assets and outputs
MEASUREMENTS_DIR        = PROJECT_ROOT / "data/myoacross/embodiments/measurements"
EMBODIMENTS_DIR         = PROJECT_ROOT / "data/myoacross/embodiments"
RETARGETS_DIR           = PROJECT_ROOT / "data/myoacross/retargets"

def resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve relative paths against the supplied base directory."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured paths must be non-empty strings")

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()
