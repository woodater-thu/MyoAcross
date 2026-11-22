"""Normalization for dynamic observations and morphology features."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from myoacross.rl.runners import distributed

from myoacross.rl.embodiments.features import (
    FEATURE_DIM,
    MorphologyFeatureNormalizer,
)


class MorphologyNormalizer(nn.Module):
    """
    Transform physical morphology into standardized log features.

    Statistics are fitted offline using training subjects only.
    They are saved in the policy state_dict and never updated online.
    """

    def __init__(self, normalizer: MorphologyFeatureNormalizer) -> None:
        super().__init__()

        self.register_buffer(
            "log_mean",
            torch.tensor(normalizer.log_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "log_scale",
            torch.tensor(normalizer.log_scale, dtype=torch.float32),
        )


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Accept positive physical values with shape (..., FEATURE_DIM).

        Positivity and finiteness are validated when morphology is loaded.
        """
        if features.ndim < 1 or features.shape[-1] != FEATURE_DIM:
            raise ValueError(
                f"Expected morphology shape (..., {FEATURE_DIM}), got {tuple(features.shape)}"
            )

        if not features.is_floating_point():
            raise TypeError("Morphology features must be floating point")

        return (torch.log(features) - self.log_mean) / self.log_scale


class ObservationNormalizer(nn.Module):
    """
    Normalize the final feature dimension.

    Leading dimensions are treated as samples when updating statistics.
    forward() never changes statistics.
    """

    def __init__(
        self,
        feature_dim: int,
        min_std: float = 1e-3,
    ) -> None:
        super().__init__()

        if (
            isinstance(feature_dim, bool)
            or not isinstance(feature_dim, int)
            or feature_dim <= 0
        ):
            raise ValueError("feature_dim must be a positive integer")

        if not math.isfinite(min_std) or min_std <= 0:
            raise ValueError("min_std must be finite and positive")

        self.feature_dim = feature_dim
        self.min_std = min_std

        self.register_buffer(
            "mean", torch.zeros(feature_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "variance", torch.ones(feature_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "count", torch.zeros((), dtype=torch.float64)
        )


    def _validate(self, values: torch.Tensor) -> None:
        if values.ndim < 1 or values.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected (..., {self.feature_dim}), got {tuple(values.shape)}")

        if not values.is_floating_point():
            raise TypeError("Observations must be floating point")


    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Merge a batch into the running population statistics."""

        self._validate(values)

        samples = values.detach().reshape(-1, self.feature_dim).to(
            device=self.mean.device,
            dtype=self.mean.dtype,
        )

        distributed.require_finite(samples, "observation statistics")
        batch_count, batch_mean, batch_variance = distributed.moments(samples)
        if batch_count.item() == 0:
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * (batch_count / total_count)

        combined_m2 = (
            self.variance * self.count
            + batch_variance * batch_count
            + delta.square() * self.count * batch_count / total_count
        )

        self.mean.copy_(new_mean)
        self.variance.copy_((combined_m2 / total_count).clamp_min(0.0))
        self.count.copy_(total_count)


    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize without modifying statistics."""
        self._validate(values)

        mean = self.mean.to(dtype=values.dtype)
        std = self.variance.sqrt().clamp_min(self.min_std)
        std = std.to(dtype=values.dtype)

        return (values - mean) / std


class GroupedObservationNormalizer(nn.Module):
    """Maintain separate feature statistics for each observation category."""

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        min_std: float = 1e-3,
    ) -> None:
        super().__init__()

        if not feature_dims:
            raise ValueError("feature_dims must not be empty")

        self.normalizers = nn.ModuleDict({
            name: ObservationNormalizer(
                feature_dim=dimension,
                min_std=min_std,
            )
            for name, dimension in feature_dims.items()
        })


    @torch.no_grad()
    def update(
        self,
        observations: Mapping[str, torch.Tensor],
    ) -> None:
        for name, normalizer in self.normalizers.items():
            normalizer.update(observations[name])


    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: normalizer(observations[name])
            for name, normalizer in self.normalizers.items()
        }
    