"""Encode standardized morphology features for policy conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math

import torch
from torch import nn

from myoacross.rl.embodiments.features import FEATURE_DIM


@dataclass(frozen=True, slots=True)
class MorphologyEncoderConfig:
    """Saved architecture fields for supported policy checkpoints."""

    input_dim: int = FEATURE_DIM
    hidden_dims: tuple[int, ...] = (64, 128)
    latent_dim: int = 256
    layernorm_eps: float = 1e-5

    def __post_init__(self) -> None:
        hidden_dims = tuple(self.hidden_dims)

        for value in (self.input_dim, *hidden_dims, self.latent_dim):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Network dimensions must be positive integers")

        # The final feature is mass; preceding features are measurements.
        if self.input_dim < 2:
            raise ValueError("input_dim must be at least 2")

        object.__setattr__(self, "hidden_dims", hidden_dims)

        eps = float(self.layernorm_eps)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("layernorm_eps must be finite and positive")
        object.__setattr__(self, "layernorm_eps", eps)

    @classmethod
    def from_checkpoint(cls, config: Mapping) -> MorphologyEncoderConfig:
        """Read current settings or supported legacy RL encoder settings."""
        if "morphology_encoder" in config:
            return cls(**config["morphology_encoder"])
        settings = dict(config["autoencoder"])
        settings.pop("measurement_weight", None)
        settings.pop("mass_weight", None)
        return cls(**settings)


class MorphologyEncoder(nn.Module):
    """Map standardized log morphology features to a learned embedding."""

    def __init__(self, config: MorphologyEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = self._build_mlp(
            dimensions=(
                config.input_dim,
                *config.hidden_dims,
                config.latent_dim,
            ),
        )
        self.encoder.append(
            nn.LayerNorm(
                config.latent_dim,
                eps=config.layernorm_eps,
            )
        )

    @staticmethod
    def _build_mlp(dimensions: tuple[int, ...]) -> nn.Sequential:
        """Apply a linear layer and SiLU at each encoder stage."""
        layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        return nn.Sequential(*layers)

    @staticmethod
    def _validate_input(
        values: torch.Tensor,
        expected_dim: int,
        name: str,
    ) -> None:
        """Validate layout without copying tensor values to CPU."""
        if values.ndim < 1 or values.shape[-1] != expected_dim:
            raise ValueError(
                f"{name} must have shape (..., {expected_dim}), got {tuple(values.shape)}"
            )

        if values.numel() == 0:
            raise ValueError(f"{name} must not be empty")

        if not values.is_floating_point():
            raise TypeError(f"{name} must be floating point")

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Map standardized log features to the morphology embedding."""
        self._validate_input(features, self.config.input_dim, "features")
        return self.encoder(features)
