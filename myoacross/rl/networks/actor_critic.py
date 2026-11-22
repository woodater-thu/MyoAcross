"""Define and build actor–critic networks with a shared morphology encoder."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


from myoacross.rl.networks.morphology_encoder import (
    MorphologyEncoder,
    MorphologyEncoderConfig,
)
from myoacross.rl.embodiments.features import (
    FEATURE_DIM,
    MorphologyFeatureNormalizer,
)
from myoacross.rl.networks.normalization import (
    GroupedObservationNormalizer,
    MorphologyNormalizer,
)
from myoacross.rl.networks.commons import (
    GroupedFiLMBackbone,
    GroupedObservationEncoder,
    LookaheadEncoder,
)

EncoderMode = Literal["none", "rl", "concat"]


class MuscleMorphologyActor(nn.Module):
    """Generate bounded excitation from a pre-sigmoid Gaussian."""

    def __init__(
        self,
        backbone: GroupedFiLMBackbone,
        action_dim: int,
        *,
        init_std: float = 0.5,
        initial_excitation: float = 0.1,
    ) -> None:
        super().__init__()

        if (
            isinstance(action_dim, bool)
            or not isinstance(action_dim, int)
            or action_dim <= 0
        ):
            raise ValueError("action_dim must be a positive integer")

        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")

        if not math.isfinite(initial_excitation) or not 0 < initial_excitation < 1:
            raise ValueError("initial_excitation must be in (0, 1)")

        self.backbone = backbone
        self.mean_head = nn.Linear(backbone.output_dim, action_dim)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(
            self.mean_head.bias,
            math.log(initial_excitation / (1 - initial_excitation)),
        )

        # Standard deviation in pre-sigmoid space.
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std)))

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        embedding: torch.Tensor | None = None,
    ) -> Normal:
        """Return the pre-sigmoid distribution, not the action distribution."""

        hidden = self.backbone(observations, embedding)
        mean = self.mean_head(hidden)
        std = self.log_std.exp().expand_as(mean)

        return Normal(mean, std)

    @staticmethod
    def _log_jacobian(pre_sigmoid: torch.Tensor) -> torch.Tensor:
        """Stable elementwise log derivative of sigmoid."""

        return F.logsigmoid(pre_sigmoid) + F.logsigmoid(-pre_sigmoid)

    @classmethod
    def action_log_prob(
        cls,
        distribution: Normal,
        pre_sigmoid: torch.Tensor,
    ) -> torch.Tensor:
        """Excitation log density, summed over muscle dimensions."""

        return (
            distribution.log_prob(pre_sigmoid) - cls._log_jacobian(pre_sigmoid)
        ).sum(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        observations: Mapping[str, torch.Tensor],
        embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return excitation, pre-sigmoid sample, and action log probability."""

        distribution = self(observations, embedding)
        pre_sigmoid = distribution.sample()

        return (
            torch.sigmoid(pre_sigmoid),
            pre_sigmoid,
            self.action_log_prob(distribution, pre_sigmoid),
        )

    @torch.no_grad()
    def deterministic(
        self,
        observations: Mapping[str, torch.Tensor],
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sigmoid(mu), the componentwise excitation median."""
        return torch.sigmoid(self(observations, embedding).mean)

    @classmethod
    def entropy_estimate(cls, distribution: Normal) -> torch.Tensor:
        """Differentiable Monte Carlo estimate of excitation entropy."""
        pre_sigmoid = distribution.rsample()

        return (distribution.entropy() + cls._log_jacobian(pre_sigmoid)).sum(dim=-1)


class MorphologyCritic(nn.Module):
    """Predict one state value per observation."""

    def __init__(self, backbone: GroupedFiLMBackbone) -> None:
        super().__init__()

        self.backbone = backbone
        self.value_head = nn.Linear(backbone.output_dim, 1)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        observations: Mapping[str, torch.Tensor],
        embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return values with shape (..., 1)."""
        hidden = self.backbone(observations, embedding)
        return self.value_head(hidden)


class MorphologyActorCritic(nn.Module):
    """Share morphology conditioning while keeping actor/critic independent."""

    def __init__(
        self,
        actor: MuscleMorphologyActor,
        critic: MorphologyCritic,
        actor_observation_normalizer: GroupedObservationNormalizer,
        critic_observation_normalizer: GroupedObservationNormalizer,
        *,
        encoder_mode: EncoderMode,
        morphology_encoder: MorphologyEncoder | None = None,
        morphology_normalizer: MorphologyNormalizer | None = None,
    ) -> None:
        super().__init__()

        valid_modes = {"none", "rl", "concat"}
        if encoder_mode not in valid_modes:
            raise ValueError(f"Unknown encoder mode: {encoder_mode}")

        if encoder_mode == "none":
            if morphology_encoder is not None or morphology_normalizer is not None:
                raise ValueError(
                    "Unconditional mode does not use an morphology_encoder"
                )
        elif encoder_mode == "concat":
            if morphology_encoder is not None or morphology_normalizer is None:
                raise ValueError(
                    "Concat requires a morphology normalizer and no morphology_encoder"
                )
        elif morphology_encoder is None or morphology_normalizer is None:
            raise ValueError(
                "Conditional modes require an morphology_encoder and a morphology normalizer"
            )

        self.actor = actor
        self.critic = critic
        self.actor_observation_normalizer = actor_observation_normalizer
        self.critic_observation_normalizer = critic_observation_normalizer
        self.encoder_mode = encoder_mode
        self.morphology_encoder = morphology_encoder
        self.morphology_normalizer = morphology_normalizer

        if self.morphology_encoder is not None:
            self.morphology_encoder.encoder.requires_grad_(encoder_mode == "rl")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Restore supported legacy encoder weights without retaining a decoder."""
        if self.encoder_mode == "rl":
            converted = {}
            for name, value in state_dict.items():
                if name.startswith("autoencoder.decoder."):
                    continue
                if name.startswith("autoencoder.encoder."):
                    name = "morphology_encoder." + name.removeprefix("autoencoder.")
                if name in converted:
                    raise ValueError(
                        f"Duplicate policy parameter after conversion: {name}"
                    )
                converted[name] = value
            state_dict = converted
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _encode_morphology(
        self,
        morphology: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return standardized features and their shared embedding."""

        if self.encoder_mode == "none":
            return None, None

        if morphology is None:
            raise ValueError("Conditional policy requires morphology features")

        assert self.morphology_normalizer is not None

        features = self.morphology_normalizer(morphology)
        if self.encoder_mode == "concat":
            return features, features
        assert self.morphology_encoder is not None
        embedding = self.morphology_encoder.encode(features)

        return features, embedding

    @torch.no_grad()
    def update_observation_statistics(
        self,
        actor_observations: Mapping[str, torch.Tensor],
        critic_observations: Mapping[str, torch.Tensor],
    ) -> None:
        """Update statistics explicitly, outside rollout and PPO epochs."""

        self.actor_observation_normalizer.update(actor_observations)
        self.critic_observation_normalizer.update(critic_observations)

    @torch.no_grad()
    def act(
        self,
        actor_observations: Mapping[str, torch.Tensor],
        critic_observations: Mapping[str, torch.Tensor],
        morphology: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample muscle excitation and evaluate its current state."""

        actor_inputs = self.actor_observation_normalizer(actor_observations)
        critic_inputs = self.critic_observation_normalizer(critic_observations)
        _, embedding = self._encode_morphology(morphology)

        distribution = self.actor(actor_inputs, embedding)
        pre_sigmoid = distribution.sample()
        actions = torch.sigmoid(pre_sigmoid)

        return {
            "actions": actions,
            "pre_sigmoid": pre_sigmoid,
            "log_prob": self.actor.action_log_prob(distribution, pre_sigmoid),
            "values": self.critic(critic_inputs, embedding).squeeze(-1),
            "mean": distribution.mean,
            "std": distribution.stddev,
        }

    def evaluate_actions(
        self,
        actor_observations: Mapping[str, torch.Tensor],
        critic_observations: Mapping[str, torch.Tensor],
        pre_sigmoid: torch.Tensor,
        morphology: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate stored actions with gradients for a PPO minibatch."""

        actor_inputs = self.actor_observation_normalizer(actor_observations)
        critic_inputs = self.critic_observation_normalizer(critic_observations)
        features, embedding = self._encode_morphology(morphology)

        distribution = self.actor(actor_inputs, embedding)

        result = {
            "log_prob": self.actor.action_log_prob(distribution, pre_sigmoid.detach()),
            "entropy": self.actor.entropy_estimate(distribution),
            "values": self.critic(critic_inputs, embedding).squeeze(-1),
            "mean": distribution.mean,
            "std": distribution.stddev,
        }

        return result

    @torch.no_grad()
    def value(
        self,
        critic_observations: Mapping[str, torch.Tensor],
        morphology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate critic observations for return bootstrapping."""

        critic_inputs = self.critic_observation_normalizer(critic_observations)
        _, embedding = self._encode_morphology(morphology)

        return self.critic(critic_inputs, embedding).squeeze(-1)

    @torch.no_grad()
    def act_inference(
        self,
        actor_observations: Mapping[str, torch.Tensor],
        morphology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce deterministic excitation for evaluation."""

        actor_inputs = self.actor_observation_normalizer(actor_observations)
        _, embedding = self._encode_morphology(morphology)

        return self.actor.deterministic(actor_inputs, embedding)


def build_actor_critic(
    observations: Mapping[str, torch.Tensor],
    action_dim: int,
    *,
    # Morphology AE param
    encoder_mode: EncoderMode,
    morphology_checkpoint: str | Path | None = None,
    morphology_statistics: MorphologyFeatureNormalizer | None = None,
    morphology_encoder_config: MorphologyEncoderConfig | None = None,
    # Observation head param
    group_hidden_dims: Mapping[str, tuple[int, ...]] | None = None,
    lookahead_frame_hidden_dims: tuple[int, ...] = (128, 64),
    lookahead_fusion_hidden_dims: tuple[int, ...] = (128,),
    # Network param
    hidden_dim: int = 2048,
    num_blocks: int = 3,
    generator_hidden_dim: int = 256,
    head_hidden_dims: tuple[int, ...] = (1024, 512),
    residual_scale: float = 0.1,
    layernorm_eps: float = 1e-5,
    observation_min_std: float = 1e-3,
    init_std: float = 0.5,
    initial_excitation: float = 0.1,
    device: str | torch.device = "cpu",
    # Checkpoint
    policy_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> MorphologyActorCritic:
    """
    Build independent actor/critic backbones and one shared morphology AE.

    observations contains dynamic groups only:
        ordinary groups: (N, D)
        lookahead:       (N, K, D)

    This function inspects shapes but does not fit observation statistics.

    Morphology initialization:
        none:
            No morphology network.
        concat:
            Encode standardized morphology as an observation group; no FiLM.
        rl:
            Restore a full policy checkpoint, or initialize the encoder
            with morphology_encoder_config and training morphology_statistics.

    Morphology statistics must be fitted using training data only.
    """

    valid_modes = {"none", "rl", "concat"}
    if encoder_mode not in valid_modes:
        raise ValueError(f"Unknown encoder mode: {encoder_mode}")

    if not observations:
        raise ValueError("No dynamic observations were provided")

    defaults = {
        "root": (64, 64),
        "joint": (256, 128),
        "muscle": (512, 256),
        "reference": (64, 64),
        "contact": (32, 32),
        "morphology": (64, 128, 256),
    }

    group_names = tuple(name for name in observations if name != "lookahead")
    if encoder_mode == "concat":
        group_names += ("morphology",)

    if group_hidden_dims is None:
        unknown = set(group_names) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown observation groups: {sorted(unknown)}")
        group_hidden_dims = {name: defaults[name] for name in group_names}
    elif set(group_hidden_dims) != set(group_names):
        raise ValueError("group_hidden_dims must match the non-lookahead groups")

    batch_size = None
    for name, values in observations.items():
        expected_ndim = 3 if name == "lookahead" else 2
        if values.ndim != expected_ndim:
            raise ValueError(
                f"{name}: expected {expected_ndim} dimensions, got {tuple(values.shape)}"
            )
        if any(size <= 0 for size in values.shape):
            raise ValueError(f"{name}: observation dimensions must be positive")
        if not values.is_floating_point():
            raise TypeError(f"{name}: observations must be floating point")

        if batch_size is None:
            batch_size = values.shape[0]
        elif values.shape[0] != batch_size:
            raise ValueError("Observation groups have different batch sizes")

    input_dims = {
        name: values.shape[-1]
        for name, values in observations.items()
        if name != "lookahead"
    }
    if encoder_mode == "concat":
        input_dims["morphology"] = FEATURE_DIM
    feature_dims = {name: values.shape[-1] for name, values in observations.items()}

    morphology_encoder = None
    morphology_normalizer = None

    if encoder_mode == "none":
        if any(
            value is not None
            for value in (
                morphology_checkpoint,
                morphology_encoder_config,
                morphology_statistics,
            )
        ):
            raise ValueError(
                "Unconditional mode does not accept morphology configuration"
            )

        # Existing backbones allocate FiLM modules but skip them
        # when embedding=None.
        embedding_dim = MorphologyEncoderConfig().latent_dim

    elif encoder_mode == "concat":
        if morphology_checkpoint is not None or morphology_encoder_config is not None:
            raise ValueError(
                "Concat does not use an morphology pretraining checkpoint or configuration"
            )
        if policy_state_dict is not None:
            if morphology_statistics is not None:
                raise ValueError("RL restoration uses saved morphology statistics")
            morphology_statistics = MorphologyFeatureNormalizer(
                log_mean=tuple(
                    policy_state_dict["morphology_normalizer.log_mean"]
                    .detach()
                    .cpu()
                    .tolist()
                ),
                log_scale=tuple(
                    policy_state_dict["morphology_normalizer.log_scale"]
                    .detach()
                    .cpu()
                    .tolist()
                ),
            )
        if morphology_statistics is None:
            raise ValueError("Concat requires training-set morphology_statistics")
        morphology_normalizer = MorphologyNormalizer(morphology_statistics)
        embedding_dim = FEATURE_DIM

    elif policy_state_dict is not None:
        # Restore morphology from the full RL checkpoint.
        if morphology_checkpoint is not None or morphology_statistics is not None:
            raise ValueError(
                "RL restoration does not accept an external morphology pretraining checkpoint "
                "or morphology statistics"
            )
        if morphology_encoder_config is None:
            raise ValueError(
                "RL restoration requires the saved morphology_encoder configuration"
            )
        if morphology_encoder_config.input_dim != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM} morphology features")

        statistics = MorphologyFeatureNormalizer(
            log_mean=tuple(
                policy_state_dict["morphology_normalizer.log_mean"]
                .detach()
                .cpu()
                .tolist()
            ),
            log_scale=tuple(
                policy_state_dict["morphology_normalizer.log_scale"]
                .detach()
                .cpu()
                .tolist()
            ),
        )

        morphology_encoder = MorphologyEncoder(morphology_encoder_config)
        morphology_normalizer = MorphologyNormalizer(statistics)
        embedding_dim = morphology_encoder_config.latent_dim

    elif morphology_checkpoint is not None:
        raise ValueError(
            "External morphology pretraining checkpoints are not supported"
        )

    else:
        # Initialize the trainable morphology encoder.
        if morphology_encoder_config is None:
            raise ValueError("Random initialization requires morphology_encoder_config")
        if morphology_statistics is None:
            raise ValueError(
                "Random initialization requires training-set morphology_statistics"
            )
        if morphology_encoder_config.input_dim != FEATURE_DIM:
            raise ValueError(
                f"Expected {FEATURE_DIM} morphology features, "
                f"got {morphology_encoder_config.input_dim}"
            )

        morphology_encoder = MorphologyEncoder(morphology_encoder_config)
        morphology_normalizer = MorphologyNormalizer(morphology_statistics)
        embedding_dim = morphology_encoder_config.latent_dim

    def make_backbone() -> GroupedFiLMBackbone:
        # Create fresh modules on every call: actor/critic do not share them.
        lookahead = None
        if "lookahead" in observations:
            values = observations["lookahead"]
            lookahead = LookaheadEncoder(
                frame_dim=values.shape[-1],
                num_frames=values.shape[-2],
                frame_hidden_dims=tuple(lookahead_frame_hidden_dims),
                fusion_hidden_dims=tuple(lookahead_fusion_hidden_dims),
            )

        observation_encoder = GroupedObservationEncoder(
            input_dims=input_dims,
            hidden_dims={name: tuple(group_hidden_dims[name]) for name in group_names},
            lookahead=lookahead,
        )

        return GroupedFiLMBackbone(
            observation_encoder=observation_encoder,
            concatenate_morphology=(encoder_mode == "concat"),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            generator_hidden_dim=generator_hidden_dim,
            head_hidden_dims=tuple(head_hidden_dims),
            residual_scale=residual_scale,
            layernorm_eps=layernorm_eps,
        )

    actor = MuscleMorphologyActor(
        backbone=make_backbone(),
        action_dim=action_dim,
        init_std=init_std,
        initial_excitation=initial_excitation,
    )
    critic = MorphologyCritic(backbone=make_backbone())

    policy = MorphologyActorCritic(
        actor=actor,
        critic=critic,
        actor_observation_normalizer=GroupedObservationNormalizer(
            feature_dims=feature_dims,
            min_std=observation_min_std,
        ),
        critic_observation_normalizer=GroupedObservationNormalizer(
            feature_dims=feature_dims,
            min_std=observation_min_std,
        ),
        encoder_mode=encoder_mode,
        morphology_encoder=morphology_encoder,
        morphology_normalizer=morphology_normalizer,
    )

    if policy_state_dict is not None:
        policy.load_state_dict(policy_state_dict, strict=True)

    return policy.to(device=device)


def load_policy(
    checkpoint_path: str | Path,
    observations: Mapping[str, torch.Tensor],
    action_dim: int,
    *,
    device: str | torch.device = "cpu",
) -> MorphologyActorCritic:
    """Restore an RL policy for inference without fitting statistics."""

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    algorithm = checkpoint["algorithm"]
    encoder_mode = algorithm["encoder_mode"]

    if encoder_mode != config["encoder"]["mode"]:
        raise ValueError("Checkpoint encoder mode disagrees with its configuration")

    morphology_encoder_config = None
    if encoder_mode not in {"none", "concat"}:
        morphology_encoder_config = MorphologyEncoderConfig.from_checkpoint(config)

    policy = build_actor_critic(
        observations=observations,
        action_dim=action_dim,
        encoder_mode=encoder_mode,
        morphology_encoder_config=morphology_encoder_config,
        policy_state_dict=algorithm["policy_state_dict"],
        device=device,
        **config["network"],
    )

    policy.requires_grad_(False)
    policy.eval()
    return policy
