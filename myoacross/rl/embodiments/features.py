"""Morphology features extracted offline from compiled MyoFullBody models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path

if TYPE_CHECKING:
    import mujoco


# Explicit measurements.
MEASUREMENT_NAMES: tuple[str, ...] = (
    "thigh_length_r",
    "shank_length_r",
    "thigh_length_l",
    "shank_length_l",
    "ankle_subtalar_length_r",
    "subtalar_mtp_length_r",
    "ankle_subtalar_length_l",
    "subtalar_mtp_length_l",
    "upper_arm_length_r",
    "forearm_length_r",
    "upper_arm_length_l",
    "forearm_length_l",
    "hip_width",
    "shoulder_width",
    "trunk_length",
    "shoulder_ankle_height_r",
    "shoulder_ankle_height_l",
)

# Explicit features
FEATURE_NAMES: tuple[str, ...] = (
    *MEASUREMENT_NAMES,
    "body_mass",
)
FEATURE_UNITS: tuple[str, ...] = ("m",) * 17 + ("kg",)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class MorphologyFeatures:
    """
    Morphology measurements of one MyoFullBody model.

    The first 17 values are lengths in meters.
    The last value is total body mass in kilograms.

    Store physical values, not log-transformed or standardized values.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)

        if array.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected shape ({FEATURE_DIM},), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Morphology features must be finite")
        if np.any(array <= 0.0):
            raise ValueError("Morphology features must be strictly positive")

        # Store immutable Python values, independent of the input array.
        object.__setattr__(
            self,
            "values",
            tuple(float(value) for value in array),
        )


    @classmethod
    def from_measurements(
        cls,
        measurements: Mapping[str, float],
        body_mass_kg: float,
    ) -> MorphologyFeatures:
        """Construct features from named measurements and actual body mass."""

        missing = set(MEASUREMENT_NAMES) - set(measurements)
        if missing:
            raise KeyError(f"Missing measurements: {sorted(missing)}")

        return cls(
            values=(
                *(measurements[name] for name in MEASUREMENT_NAMES),
                body_mass_kg,
            )
        )


    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
    ) -> MorphologyFeatures:
        """
        Extract features offline from a compiled standalone subject.

        The model must contain one subject and no additional massive objects.
        Measurements are evaluated in the model's reference configuration.
        """
        from myoacross.embodiment.measurements import (
            compute_myofullbody_landmarks,
            compute_myofullbody_measurements,
        )

        landmarks, names = compute_myofullbody_landmarks(model)
        measurements = compute_myofullbody_measurements(landmarks, names)

        # Include auxiliary subject bodies; exclude the world body.
        mass = float(np.sum(model.body_mass[1:], dtype=np.float64))

        return cls.from_measurements(measurements, mass)


    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> MorphologyFeatures:
        """Restore physical features from dict."""

        if tuple(record.get("names", ())) != FEATURE_NAMES:
            raise ValueError("Morphology feature names or order do not match")

        if tuple(record.get("units", ())) != FEATURE_UNITS:
            raise ValueError("Morphology feature units do not match")

        return cls(values=tuple(record["values"]))

    
    @property
    def body_mass_kg(self) -> float:
        return self.values[-1]


    def as_array(self) -> np.ndarray:
        """Return a new array of physical feature values."""
        return np.asarray(self.values, dtype=np.float64)


    def as_log_array(self) -> np.ndarray:
        """Return log features using the fixed meter/kilogram convention."""
        return np.log(self.as_array())


    def to_dict(self) -> dict:
        """Return a JSON-compatible, self-describing feature record."""
        return {
            "names": list(FEATURE_NAMES),
            "units": list(FEATURE_UNITS),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class MorphologyFeatureNormalizer:
    """Fixed per-feature normalization statistics in log space."""

    log_mean: tuple[float, ...]
    log_scale: tuple[float, ...]

    def __post_init__(self) -> None:
        mean = np.asarray(self.log_mean, dtype=np.float64)
        scale = np.asarray(self.log_scale, dtype=np.float64)

        if mean.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected mean shape ({FEATURE_DIM},), got {mean.shape}")
        if scale.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected scale shape ({FEATURE_DIM},), got {scale.shape}")
        if not np.isfinite(mean).all():
            raise ValueError("Normalization means must be finite")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("Normalization scales must be finite and positive")

        object.__setattr__(self, "log_mean", tuple(float(value) for value in mean)
        )
        object.__setattr__(self, "log_scale", tuple(float(value) for value in scale))

    @classmethod
    def fit(
        cls,
        features: Iterable[MorphologyFeatures],
        *,
        min_std: float = 1e-6,
    ) -> MorphologyFeatureNormalizer:
        """
        Fit log-space statistics from one or more training bodies.

        Each body contributes one equally weighted sample.
        Constant dimensions are centered without rescaling.
        """
        if not np.isfinite(min_std) or min_std <= 0.0:
            raise ValueError("min_std must be finite and positive")

        samples = [feature.as_log_array() for feature in features]
        if not samples:
            raise ValueError("At least one training sample is required")

        log_values = np.stack(samples, axis=0)
        mean = log_values.mean(axis=0)
        std = log_values.std(axis=0, ddof=0)

        scale = np.where(std < min_std, 1.0, std)

        return cls(
            log_mean=tuple(mean),
            log_scale=tuple(scale),
        )
    
    def transform(self, physical_values: np.ndarray) -> np.ndarray:
        """Convert positive physical values to standardized log features."""

        values = self._validate_array(physical_values)
        if np.any(values <= 0.0):
            raise ValueError("Physical features must be strictly positive")

        mean = np.asarray(self.log_mean)
        scale = np.asarray(self.log_scale)

        return (np.log(values) - mean) / scale
    
    def inverse_transform(self, normalized_values: np.ndarray) -> np.ndarray:
        """Recover physical values from standardized log features."""

        values = self._validate_array(normalized_values)

        mean = np.asarray(self.log_mean)
        scale = np.asarray(self.log_scale)

        with np.errstate(over="raise", invalid="raise"):
            physical = np.exp(values * scale + mean)

        if not np.isfinite(physical).all() or np.any(physical <= 0.0):
            raise ValueError("Inverse transform produced invalid physical values")

        return physical

    def to_dict(self) -> dict:
        """Return JSON-compatible normalization statistics."""

        return {
            "names": list(FEATURE_NAMES),
            "units": list(FEATURE_UNITS),
            "transform": "log_standardize",
            "log_mean": list(self.log_mean),
            "log_scale": list(self.log_scale),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> MorphologyFeatureNormalizer:
        """Restore statistics after checking the feature definition."""
        
        if tuple(record.get("names", ())) != FEATURE_NAMES:
            raise ValueError("Normalization feature names or order do not match")
        if tuple(record.get("units", ())) != FEATURE_UNITS:
            raise ValueError("Normalization feature units do not match")
        if record.get("transform") != "log_standardize":
            raise ValueError("Unsupported feature transformation")

        return cls(
            log_mean=tuple(record["log_mean"]),
            log_scale=tuple(record["log_scale"]),
        )

    @staticmethod
    def _validate_array(values: np.ndarray) -> np.ndarray:
        """Validate one vector or a batch with shape (..., FEATURE_DIM)."""
        array = np.asarray(values, dtype=np.float64)

        if array.ndim < 1 or array.shape[-1] != FEATURE_DIM:
            raise ValueError(f"Expected shape (..., {FEATURE_DIM}), got {array.shape}")
        if array.size == 0:
            raise ValueError("Feature array must not be empty")
        if not np.isfinite(array).all():
            raise ValueError("Feature array must be finite")

        return array

    def save(self, path: str | Path) -> None:
        """Save fixed training-set statistics without overwriting a file."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("x", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> MorphologyFeatureNormalizer:
        """Restore fixed statistics without fitting them again."""
        path = Path(path).expanduser()

        with path.open(encoding="utf-8") as file:
            record = json.load(file)

        if not isinstance(record, dict):
            raise ValueError("Normalization statistics must be a mapping")

        return cls.from_dict(record)


def load_morphology_features(
    index_paths: Sequence[str | Path],
) -> tuple[list[MorphologyFeatures], set[str]]:
    """
    Load features and source-subject IDs from generation indices.

    No simulation arrays are loaded. 
    A sample is identified by (source_subject_id, variant_id).
    Shared baseline has one sample and no source-subject IDs.
    """

    if not index_paths:
        raise ValueError("At least one index path is required")

    features = []
    subject_ids = set()
    sample_ids = set()
    visited_paths = set()

    for index_path in index_paths:
        path = Path(index_path).expanduser().resolve()

        if path in visited_paths:
            raise ValueError(f"Repeated index path: {path}")
        visited_paths.add(path)

        with path.open(encoding="utf-8") as file:
            record = json.load(file)

        if not isinstance(record, dict):
            raise ValueError(f"Index must contain a mapping: {path}")

        if "source_subject_id" not in record:
            raise ValueError(f"Missing source_subject_id: {path}")

        subject_id = record["source_subject_id"]
        embodiment = record.get("embodiment")
        morphology_source = record.get("morphology_source")

        if embodiment not in ("baseline", "height_scale", "segment_scale"):
            raise ValueError(f"Invalid embodiment: {path}")
        if morphology_source not in ("subject", "virtual"):
            raise ValueError(f"Invalid morphology_source: {path}")

        if embodiment == "baseline":
            if subject_id is not None:
                raise ValueError(
                    f"Shared baseline requires source_subject_id=None: {path}"
                )
            if morphology_source != "subject":
                raise ValueError(
                    f"Baseline requires subject morphology: {path}"
                )
            if len(index_paths) != 1:
                raise ValueError(
                    "Shared baseline requires exactly one index"
                )
        elif not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError(f"Invalid source_subject_id: {path}")

        samples = record.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Index must contain non-empty samples: {path}")

        count = record.get("count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != len(samples)
        ):
            raise ValueError(f"Index count does not match samples: {path}")

        if embodiment == "baseline" and count != 1:
            raise ValueError(
                f"Shared baseline requires exactly one sample: {path}"
            )

        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample record: {path}")

            variant_id = sample.get("variant_id")
            if not isinstance(variant_id, str) or not variant_id.strip():
                raise ValueError(f"Invalid variant_id: {path}")

            sample_id = (subject_id, variant_id)
            if sample_id in sample_ids:
                raise ValueError(f"Repeated virtual subject: {sample_id}")

            sample_ids.add(sample_id)
            features.append(MorphologyFeatures.from_dict(sample["features"]))

        if subject_id is not None:
            subject_ids.add(subject_id)

    return features, subject_ids


def fit_morphology_feature_normalizer(
    train_index_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    excluded_subject_ids: Iterable[str] = (),
) -> Path:
    """Fit and save fixed morphology statistics from training indices."""
    features, subject_ids = load_morphology_features(train_index_paths)

    overlap = subject_ids.intersection(excluded_subject_ids)
    if overlap:
        raise ValueError(
            f"Excluded subjects found in training indices: {sorted(overlap)}"
        )

    normalizer = MorphologyFeatureNormalizer.fit(features)

    output_path = Path(output_path).expanduser().resolve()
    normalizer.save(output_path)

    return output_path
