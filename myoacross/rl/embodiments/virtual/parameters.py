"""Physical parameters for offline morphology generation"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myoacross.embodiment.segment_scale.model import OPTIMIZATION_REGIONS


REGION_NAMES: tuple[str, ...] = tuple(OPTIMIZATION_REGIONS)

SEGMENT_PARAMETER_NAMES: tuple[str, ...] = (
    *REGION_NAMES,
    "body_mass_kg",
)

HEIGHT_PARAMETER_NAMES: tuple[str, ...] = (
    "uniform_scale",
    "body_mass_kg",
)


def _positive_vector(
    values: np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    """Validate one finite, strictly positive parameter vector."""
    array = np.asarray(values, dtype=np.float64)

    if array.shape != (size,):
        raise ValueError(f"{name}: expected shape ({size},), got {array.shape}")

    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"{name}: all values must be finite and strictly positive")

    return array


@dataclass(frozen=True, slots=True)
class SegmentScaleParameters:
    """Absolute regional scales and target total body mass."""

    region_scales: tuple[float, ...]
    body_mass_kg: float

    def __post_init__(self) -> None:
        values = _positive_vector(
            np.asarray(
                (*self.region_scales, self.body_mass_kg),
                dtype=np.float64,
            ),
            size=len(SEGMENT_PARAMETER_NAMES),
            name="SegmentScaleParameters",
        )

        object.__setattr__(
            self,
            "region_scales",
            tuple(float(value) for value in values[:-1]),
        )
        object.__setattr__(
            self,
            "body_mass_kg",
            float(values[-1]),
        )

    def as_array(self) -> np.ndarray:
        """Return parameters in SEGMENT_PARAMETER_NAMES order."""
        return np.asarray(
            (*self.region_scales, self.body_mass_kg),
            dtype=np.float64,
        )

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
    ) -> SegmentScaleParameters:
        """Construct parameters from the sampler's output vector."""
        values = _positive_vector(
            values,
            size=len(SEGMENT_PARAMETER_NAMES),
            name="SegmentScaleParameters",
        )

        return cls(
            region_scales=tuple(float(value) for value in values[:-1]),
            body_mass_kg=float(values[-1]),
        )


@dataclass(frozen=True, slots=True)
class HeightScaleParameters:
    """One absolute uniform scale and target total body mass."""

    uniform_scale: float
    body_mass_kg: float

    def __post_init__(self) -> None:
        values = _positive_vector(
            np.asarray(
                (self.uniform_scale, self.body_mass_kg),
                dtype=np.float64,
            ),
            size=len(HEIGHT_PARAMETER_NAMES),
            name="HeightScaleParameters",
        )

        object.__setattr__(
            self,
            "uniform_scale",
            float(values[0]),
        )
        object.__setattr__(
            self,
            "body_mass_kg",
            float(values[1]),
        )

    def as_array(self) -> np.ndarray:
        """Return parameters in HEIGHT_PARAMETER_NAMES order."""
        return np.asarray(
            (self.uniform_scale, self.body_mass_kg),
            dtype=np.float64,
        )

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
    ) -> HeightScaleParameters:
        """Construct parameters from the sampler's output vector."""
        values = _positive_vector(
            values,
            size=len(HEIGHT_PARAMETER_NAMES),
            name="HeightScaleParameters",
        )

        return cls(
            uniform_scale=float(values[0]),
            body_mass_kg=float(values[1]),
        )


GenerationParameters = SegmentScaleParameters | HeightScaleParameters
