"""Convert grouped environment observations to and from RSL-RL storage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from myoacross.rl.embodiments.features import FEATURE_DIM
from myoacross.rl.networks.actor_critic import (
    EncoderMode,
    MorphologyActorCritic,
)


def unpack_policy_observations(
    observations: Mapping,
    *,
    encoder_mode: EncoderMode,
    group: Literal["actor", "critic"] = "actor",
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Read one policy group from environment or flat storage observations."""

    if encoder_mode not in {
        "none", "rl", "concat"
    }:
        raise ValueError(f"Unknown encoder mode: {encoder_mode}")

    if group not in {"actor", "critic"}:
        raise ValueError(f"Unknown observation group: {group}")

    if group in observations:
        grouped = observations[group]
        if not isinstance(grouped, Mapping):
            raise TypeError(
                f"{group} observations must be grouped; set concatenate_terms=False"
            )
        dynamic = dict(grouped)
    else:
        prefix = f"{group}/"
        dynamic = {
            key[len(prefix):]: value
            for key, value in observations.items()
            if isinstance(key, str) and key.startswith(prefix)
        }

    if not dynamic:
        raise ValueError(f"No {group} observations were provided")

    for name, value in dynamic.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(f"Invalid observation name: {name!r}")

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{group}/{name} must be a tensor")

        if value.ndim < 2:
            raise ValueError(f"{group}/{name} must have a batch and feature dimension")

    first = next(iter(dynamic.values()))
    batch_size = first.shape[0]
    device = first.device

    for name, value in dynamic.items():
        if value.shape[0] != batch_size or value.device != device:
            raise ValueError(f"{group}/{name} has an inconsistent batch size or device")

    if encoder_mode == "none":
        return dynamic, None

    morphology = observations["morphology"]

    if not isinstance(morphology, torch.Tensor):
        raise TypeError("Morphology observations must be a tensor")

    if morphology.shape != (batch_size, FEATURE_DIM):
        raise ValueError(
            f"Expected morphology shape ({batch_size}, {FEATURE_DIM}), "
            f"got {tuple(morphology.shape)}"
        )

    if morphology.device != device:
        raise ValueError("Morphology and dynamic observations must share a device")

    return dynamic, morphology


def pack_policy_observations(
    observations: Mapping,
    *,
    encoder_mode: EncoderMode,
) -> TensorDict:
    """Snapshot observations using flat keys for RSL-RL rollout storage."""

    actor, morphology = unpack_policy_observations(
        observations,
        encoder_mode=encoder_mode,
        group="actor",
    )
    critic, _ = unpack_policy_observations(
        observations,
        encoder_mode=encoder_mode,
        group="critic",
    )

    actor_first = next(iter(actor.values()))
    critic_first = next(iter(critic.values()))

    if (
        actor_first.shape[0] != critic_first.shape[0]
        or actor_first.device != critic_first.device
    ):
        raise ValueError("Actor and critic observations must share batch size and device")

    flat = {
        f"{group}/{name}": value.detach().clone()
        for group, values in (("actor", actor), ("critic", critic))
        for name, value in values.items()
    }

    if morphology is not None:
        flat["morphology"] = morphology.detach().clone()

    return TensorDict(
        flat,
        batch_size=[actor_first.shape[0]],
        device=actor_first.device,
    )


def make_rollout_transition(
    observations: TensorDict,
    policy_output: Mapping[str, torch.Tensor],
) -> RolloutStorage.Transition:
    """
    Record policy inputs and outputs before stepping the environment.

    Store pre-sigmoid actions for stable probability evaluation.
    Rewards and dones are filled after the environment step.
    """
    transition = RolloutStorage.Transition()

    # pack_policy_observations() already created independent snapshots.
    transition.observations = observations

    transition.actions = (
        policy_output["pre_sigmoid"].detach().clone()
    )

    # RSL-RL stores values with shape (num_envs, 1).
    transition.values = (
        policy_output["values"].detach().unsqueeze(-1).clone()
    )
    transition.actions_log_prob = (
        policy_output["log_prob"].detach().clone()
    )

    # Parameters of the Gaussian before the sigmoid transformation.
    transition.distribution_params = (
        policy_output["mean"].detach().clone(),
        policy_output["std"].detach().clone(),
    )

    return transition


class RslCriticAdapter:
    """
    Expose the critic interface used by RSL-RL PPO.compute_returns().

    The policy owns all network parameters. This adapter only delegates
    value evaluation and provides the feedforward hidden-state interface.
    """

    is_recurrent = False

    def __init__(self, policy: MorphologyActorCritic) -> None:
        self.policy = policy

    @torch.no_grad()
    def __call__(self, observations: TensorDict) -> torch.Tensor:
        """Return state values with shape (batch_size, 1)."""

        critic_observations, morphology = unpack_policy_observations(
            observations,
            encoder_mode=self.policy.encoder_mode,
            group="critic",
        )

        values = self.policy.value(
            critic_observations=critic_observations,
            morphology=morphology,
        )

        return values.unsqueeze(-1)

    def get_hidden_state(self) -> None:
        """Feedforward networks have no recurrent hidden state."""
        return None

    def reset(
        self,
        dones: torch.Tensor | None = None,
        *,
        hidden_state=None,
    ) -> None:
        """Accept the reset interface used by RSL-RL."""

        if hidden_state is not None:
            raise ValueError(
                "A feedforward critic does not accept recurrent hidden state"
            )

        