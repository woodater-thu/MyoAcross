"""
Covariance construction and bounded morphology sampling.

Covariance coefficients are engineering priors, not fitted population statistics. 
Covariances describe relative Gaussian perturbations before rejection sampling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np

from myoacross.rl.embodiments.virtual.parameters import (
    HEIGHT_PARAMETER_NAMES,
    REGION_NAMES,
    SEGMENT_PARAMETER_NAMES,
    HeightScaleParameters,
    SegmentScaleParameters,
)


ParametersT = TypeVar(
    "ParametersT",
    SegmentScaleParameters,
    HeightScaleParameters,
)

PAIRED_REGIONS = (
    "upper_arm",
    "forearm",
    "thigh",
    "shank",
    "foot",
)

AXIAL_REGIONS = (
    "pelvis_sacrum",
    "trunk",
    "shoulder_girdle",
)


def _validate_coefficients(values: Mapping[str, float]) -> None:
    """Require finite, nonnegative factor coefficients."""
    for name, value in values.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                f"{name} must be finite and nonnegative"
            )


def _covariance_from_factors(
    parameter_names: Sequence[str],
    factors: Sequence[Mapping[str, float]],
) -> np.ndarray:
    """Return A @ A.T, with one independent Gaussian per column."""
    indices = {
        name: index
        for index, name in enumerate(parameter_names)
    }

    matrix = np.zeros(
        (len(parameter_names), len(factors)),
        dtype=np.float64,
    )

    for column, weights in enumerate(factors):
        for name, weight in weights.items():
            matrix[indices[name], column] = weight

    with np.errstate(over="ignore", invalid="ignore"):
        covariance = matrix @ matrix.T

    if not np.isfinite(covariance).all():
        raise ValueError(
            "Covariance overflow; reduce factor coefficients"
        )

    return covariance


def build_segment_covariance(
    *,
    global_size_std: float,
    axial_std: float,
    upper_limb_std: float,
    lower_limb_std: float,
    region_std: Mapping[str, float],
    asymmetry_std: float,
    mass_size_coefficient: float,
    mass_residual_std: float,
) -> np.ndarray:
    """
    Build covariance in SEGMENT_PARAMETER_NAMES order.

    Relative perturbations contain:
    - a shared global size factor;
    - axial, upper-limb and lower-limb factors;
    - independent regional factors;
    - opposite left/right asymmetry factors;
    - a mass residual independent of global size.

    Mass also receives the global size factor multiplied by mass_size_coefficient.
    """
    expected = set(AXIAL_REGIONS + PAIRED_REGIONS)

    if set(region_std) != expected:
        raise ValueError(
            f"region_std must contain exactly {sorted(expected)}"
        )

    _validate_coefficients({
        "global_size_std": global_size_std,
        "axial_std": axial_std,
        "upper_limb_std": upper_limb_std,
        "lower_limb_std": lower_limb_std,
        "asymmetry_std": asymmetry_std,
        "mass_size_coefficient": mass_size_coefficient,
        "mass_residual_std": mass_residual_std,
        **{
            f"region_std.{name}": value
            for name, value in region_std.items()
        },
    })

    factors: list[dict[str, float]] = [
        {
            **{
                name: global_size_std
                for name in REGION_NAMES
            },
            "body_mass_kg": (
                mass_size_coefficient * global_size_std
            ),
        },
        {
            name: axial_std
            for name in AXIAL_REGIONS
        },
        {
            f"{region}_{side}": upper_limb_std
            for region in ("upper_arm", "forearm")
            for side in ("r", "l")
        },
        {
            f"{region}_{side}": lower_limb_std
            for region in ("thigh", "shank")
            for side in ("r", "l")
        },
    ]

    for name in AXIAL_REGIONS:
        factors.append({
            name: region_std[name],
        })

    for name in PAIRED_REGIONS:
        factors.append({
            f"{name}_r": region_std[name],
            f"{name}_l": region_std[name],
        })
        factors.append({
            f"{name}_r": asymmetry_std,
            f"{name}_l": -asymmetry_std,
        })

    factors.append({
        "body_mass_kg": mass_residual_std,
    })

    return _covariance_from_factors(
        SEGMENT_PARAMETER_NAMES,
        factors,
    )


def build_height_covariance(
    *,
    global_size_std: float,
    mass_size_coefficient: float,
    mass_residual_std: float,
) -> np.ndarray:
    """
    Build covariance in HEIGHT_PARAMETER_NAMES order.

    With independent standard normal variables z_size and z_mass:

        scale_noise = global_size_std * z_size
        mass_noise = (
            mass_size_coefficient * scale_noise
            + mass_residual_std * z_mass
        )

    This is a relative size/mass prior, not an exact constant-density constraint.
    """
    _validate_coefficients({
        "global_size_std": global_size_std,
        "mass_size_coefficient": mass_size_coefficient,
        "mass_residual_std": mass_residual_std,
    })

    return _covariance_from_factors(
        HEIGHT_PARAMETER_NAMES,
        [
            {
                "uniform_scale": global_size_std,
                "body_mass_kg": (
                    mass_size_coefficient * global_size_std
                ),
            },
            {
                "body_mass_kg": mass_residual_std,
            },
        ],
    )


class MorphologySampler:
    """
    Sample bounded relative perturbations around a fixed subject.

    sampled = center * (1 + noise)

    Covariance and bounds follow the corresponding parameter order.
    Reject the entire noise vector when any bound is exceeded.

    The caller owns the random generator. Each sample is drawn around
    the supplied center, rather than around the previous sample.
    """

    def __init__(
        self,
        covariance: np.ndarray,
        max_relative_change: np.ndarray,
        *,
        max_attempts: int = 1000,
    ) -> None:
        covariance = np.array(
            covariance,
            dtype=np.float64,
            copy=True,
        )
        bounds = np.array(
            max_relative_change,
            dtype=np.float64,
            copy=True,
        )

        if bounds.ndim != 1 or bounds.size == 0:
            raise ValueError(
                "max_relative_change must be a nonempty vector"
            )

        dimension = bounds.size

        if covariance.shape != (dimension, dimension):
            raise ValueError(
                f"Expected covariance shape "
                f"({dimension}, {dimension}), "
                f"got {covariance.shape}"
            )

        if not np.isfinite(covariance).all():
            raise ValueError("Covariance must be finite")

        if not np.allclose(
            covariance,
            covariance.T,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Covariance must be symmetric")

        if not np.isfinite(bounds).all():
            raise ValueError("Relative bounds must be finite")

        if np.any(bounds <= 0.0) or np.any(bounds >= 1.0):
            raise ValueError(
                "Each relative bound must satisfy 0 < bound < 1"
            )

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts <= 0
        ):
            raise ValueError(
                "max_attempts must be a positive integer"
            )

        covariance = 0.5 * covariance + 0.5 * covariance.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        if eigenvalues.min() < -1e-12:
            raise ValueError(
                "Covariance must be positive semidefinite"
            )

        self._factor = (
            eigenvectors
            * np.sqrt(np.maximum(eigenvalues, 0.0))
        )
        self._bounds = bounds
        self._dimension = dimension
        self._max_attempts = max_attempts

    def sample(
        self,
        center: ParametersT,
        rng: np.random.Generator,
    ) -> ParametersT:
        """Return a sample of the same parameter type as center."""
        center_values = center.as_array()

        if center_values.shape != (self._dimension,):
            raise ValueError(
                f"Sampler dimension {self._dimension} does not "
                f"match {type(center).__name__} dimension "
                f"{center_values.size}"
            )

        for _ in range(self._max_attempts):
            noise = (
                self._factor
                @ rng.standard_normal(self._dimension)
            )

            if np.any(np.abs(noise) > self._bounds):
                continue

            sampled_values = center_values * (1.0 + noise)
            return type(center).from_array(sampled_values)

        raise RuntimeError(
            "Failed to sample within bounds; "
            "check covariance and relative limits"
        )

    