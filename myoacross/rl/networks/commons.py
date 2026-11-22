"""Shared network components for morphology-conditioned control."""

from __future__ import annotations

import torch
from torch import nn
from collections.abc import Mapping




class FiLMLayer(nn.Module):
    """
    Modulate hidden features from a morphology embedding.

    output = (1 + delta_gamma) * hidden + beta

    Leading batch dimensions of hidden and embedding must match.
    """

    def __init__(
        self,
        embedding_dim: int,
        generator_hidden_dim,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.modulation = nn.Sequential(
            nn.Linear(embedding_dim, generator_hidden_dim),
            nn.SiLU(),
            nn.Linear(generator_hidden_dim, 2 * hidden_dim),
        )

        # Initialize the final projection close to zero.
        output_layer = self.modulation[-1]
        nn.init.normal_(output_layer.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape[:-1] != embedding.shape[:-1]:
            raise ValueError("Hidden features and embedding must have matching batch dimensions")

        delta_gamma, beta = self.modulation(embedding).chunk(2, dim=-1)

        return (1.0 + delta_gamma) * hidden + beta


class FiLMResBlock(nn.Module):
    """Pre-normalized residual block with morphology-conditioned FiLM."""

    def __init__(
        self,
        hidden_dim: int = 2048,
        embedding_dim: int = 256,
        generator_hidden_dim: int = 512,
        residual_scale: float = 0.1,
        layernorm_eps: float = 1e-5,
        use_film: bool = True,
    ) -> None:
        super().__init__()

        self.film = FiLMLayer(
            embedding_dim=embedding_dim,
            generator_hidden_dim=generator_hidden_dim,
            hidden_dim=hidden_dim,
        ) if use_film else None
        self.norm = nn.LayerNorm(
            hidden_dim,
            eps=layernorm_eps,
            elementwise_affine=False,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.residual_scale = residual_scale

    def forward(
        self,
        hidden: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.norm(hidden)

        if embedding is not None and self.film is not None:
            residual = self.film(residual, embedding)

        residual = self.linear1(residual)
        residual = self.activation(residual)
        residual = self.linear2(residual)

        return hidden + self.residual_scale * residual


class ObservationEncoder(nn.Module):
    """Encode one observation category using Linear + SiLU layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()

        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        for dim in (input_dim, *hidden_dims):
            if (
                isinstance(dim, bool)
                or not isinstance(dim, int)
                or dim <= 0
            ):
                raise ValueError("Encoder dimensions must be positive integers")

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.SiLU(),
            ])
            previous_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim < 1 or values.shape[-1] != self.input_dim:
            raise ValueError(f"Expected shape (..., {self.input_dim}), got {tuple(values.shape)}")
        return self.network(values)


class LookaheadEncoder(nn.Module):
    """Encode each future frame, then fuse the ordered frame features."""

    def __init__(
        self,
        frame_dim: int,
        num_frames: int,
        frame_hidden_dims: tuple[int, ...] = (128, 64),
        fusion_hidden_dims: tuple[int, ...] = (128,),
    ) -> None:
        super().__init__()

        if (
            isinstance(num_frames, bool)
            or not isinstance(num_frames, int)
            or num_frames <= 0
        ):
            raise ValueError("num_frames must be a positive integer")

        self.frame_dim = frame_dim
        self.num_frames = num_frames

        self.frame_encoder = ObservationEncoder(
            input_dim=frame_dim,
            hidden_dims=frame_hidden_dims,
        )
        self.fusion = ObservationEncoder(
            input_dim=num_frames * self.frame_encoder.output_dim,
            hidden_dims=fusion_hidden_dims,
        )

        self.output_dim = self.fusion.output_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected = (self.num_frames, self.frame_dim)

        if values.ndim < 2 or tuple(values.shape[-2:]) != expected:
            raise ValueError(f"Expected lookahead shape (..., {expected[0]}, {expected[1]}), got {tuple(values.shape)}")

        # (..., K, frame_dim) -> (..., K, frame_embedding_dim)
        encoded = self.frame_encoder(values)

        # Preserve frame order; do not average across time.
        return self.fusion(encoded.flatten(start_dim=-2))


class GroupedObservationEncoder(nn.Module):
    """Encode named observation groups and concatenate their embeddings."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        hidden_dims: Mapping[str, tuple[int, ...]],
        *,
        lookahead: LookaheadEncoder | None = None,
    ) -> None:
        super().__init__()

        if not input_dims:
            raise ValueError("At least one observation group is required")

        if set(input_dims) != set(hidden_dims):
            raise ValueError("input_dims and hidden_dims must contain the same groups")

        if "lookahead" in input_dims:
            raise ValueError("Configure lookahead through the separate lookahead argument")

        # Preserve the explicitly configured concatenation order.
        self.group_names = tuple(input_dims)

        self.encoders = nn.ModuleDict({
            name: ObservationEncoder(
                input_dim=input_dims[name],
                hidden_dims=tuple(hidden_dims[name]),
            )
            for name in self.group_names
        })

        self.lookahead = lookahead
        self.output_dim = sum(
            encoder.output_dim
            for encoder in self.encoders.values()
        )

        if self.lookahead is not None:
            self.output_dim += self.lookahead.output_dim

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        encoded = [
            self.encoders[name](observations[name])
            for name in self.group_names
        ]

        if self.lookahead is not None:
            encoded.append(self.lookahead(observations["lookahead"]))

        batch_shape = encoded[0].shape[:-1]
        if any(value.shape[:-1] != batch_shape for value in encoded):
            raise ValueError("Observation groups have different batch shapes")

        return torch.cat(encoded, dim=-1)


class GroupedFiLMBackbone(nn.Module):
    """
    Encode observation groups with FiLM or a morphology observation group.

    Dynamic observations must already be normalized.
    The conditioning input is either an encoder embedding or normalized features.
    """

    def __init__(
        self,
        observation_encoder: GroupedObservationEncoder,
        embedding_dim: int,
        *,
        hidden_dim: int = 2048,
        num_blocks: int = 3,
        generator_hidden_dim: int = 256,
        head_hidden_dims: tuple[int, ...] = (1024, 512),
        residual_scale: float = 0.1,
        layernorm_eps: float = 1e-5,
        concatenate_morphology: bool = False,
    ) -> None:
        super().__init__()

        if (
            isinstance(num_blocks, bool)
            or not isinstance(num_blocks, int)
            or num_blocks <= 0
        ):
            raise ValueError("num_blocks must be a positive integer")

        self.observation_encoder = observation_encoder
        self.concatenate_morphology = concatenate_morphology
        self.embedding_dim = embedding_dim

        input_dim = observation_encoder.output_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            FiLMResBlock(
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                generator_hidden_dim=generator_hidden_dim,
                residual_scale=residual_scale,
                layernorm_eps=layernorm_eps,
                use_film=not concatenate_morphology,
            )
            for _ in range(num_blocks)
        ])

        self.head = ObservationEncoder(
            input_dim=hidden_dim,
            hidden_dims=head_hidden_dims,
        )
        self.output_dim = self.head.output_dim

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.concatenate_morphology:
            if embedding is None or embedding.shape[-1] != self.embedding_dim:
                raise ValueError("Concatenation requires morphology features with matching width")
            observations = {**observations, "morphology": embedding}
            embedding = None
        hidden = self.observation_encoder(observations)
        hidden = self.input_projection(hidden)

        for block in self.blocks:
            hidden = block(hidden, embedding)

        return self.head(hidden)
