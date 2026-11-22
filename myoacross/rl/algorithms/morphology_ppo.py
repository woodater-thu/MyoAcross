"""Adapt RSL-RL PPO to a shared morphology-conditioned actor–critic."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from myoacross.rl.runners import distributed

from myoacross.rl.networks.actor_critic import MorphologyActorCritic
from myoacross.rl.networks.rsl_adapter import (
    RslCriticAdapter,
    make_rollout_transition,
    pack_policy_observations,
    unpack_policy_observations,
)


class MorphologyPPO(PPO):
    """
    Collect rollouts with a shared morphology encoder.

    Uses RSL-RL rollout storage and return computation.
    This class is driven by our training runner, not OnPolicyRunner.
    """

    def __init__(
        self,
        policy: MorphologyActorCritic,
        storage: RolloutStorage,
        *,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        desired_kl: float | None = 0.01,
    ) -> None:
        if storage.training_type != "rl":
            raise ValueError("PPO requires RL rollout storage")

        if storage.num_envs * storage.num_transitions_per_env < 2:
            raise ValueError("Advantage normalization requires at least two samples")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
            raise ValueError("gamma and lam must be in [0, 1]")

        # Do not call PPO.__init__: it creates an optimizer by chaining
        # separately owned actor and critic parameters.
        self.policy = policy
        self.critic = RslCriticAdapter(policy)
        self.storage = storage

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam

        # Used by the inherited compute_returns().
        self.normalize_advantage_per_mini_batch = False

        # One owner and one optimizer for all trainable parameters.
        self.parameters = [
            parameter
            for parameter in policy.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=learning_rate,
            # Preserve the existing PPO setup without weight decay.
            weight_decay=0.0,
        )

        if num_learning_epochs <= 0 or num_mini_batches <= 0:
            raise ValueError("Epoch and minibatch counts must be positive")

        rollout_size = (storage.num_envs * storage.num_transitions_per_env)
        if rollout_size % num_mini_batches != 0:
            raise ValueError("Rollout size must be divisible by num_mini_batches")

        if not 0 < clip_param < 1:
            raise ValueError("clip_param must be in (0, 1)")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if desired_kl is not None and desired_kl <= 0:
            raise ValueError("desired_kl must be positive or None")
        if min(value_loss_coef, entropy_coef) < 0:
            raise ValueError("Loss coefficients must be nonnegative")

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.transition: RolloutStorage.Transition | None = None
        self.returns_ready = False

    @torch.no_grad()
    def act(self, observations: Mapping) -> torch.Tensor:
        """Save the current policy transition and return excitation."""

        if self.transition is not None:
            raise RuntimeError("Call process_env_step() before sampling another action")
        if self.storage.step >= self.storage.num_transitions_per_env:
            raise RuntimeError("Rollout storage is full")

        packed = pack_policy_observations(
            observations,
            encoder_mode=self.policy.encoder_mode,
        )

        actor_observations, morphology = unpack_policy_observations(
            packed,
            encoder_mode=self.policy.encoder_mode,
            group="actor",
        )
        critic_observations, _ = unpack_policy_observations(
            packed,
            encoder_mode=self.policy.encoder_mode,
            group="critic",
        )

        output = self.policy.act(
            actor_observations=actor_observations,
            critic_observations=critic_observations,
            morphology=morphology,
        )

        self.transition = make_rollout_transition(packed, output)
        self.returns_ready = False

        return output["actions"]

    @torch.no_grad()
    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Store a transition using RSL-RL's timeout bootstrap rule."""

        if self.transition is None:
            raise RuntimeError("Call act() before process_env_step()")

        expected_shape = (self.storage.num_envs,)
        for name, tensor in (
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name}: expected {expected_shape}, "
                    f"got {tuple(tensor.shape)}"
                )

        terminated = terminated.bool()
        truncated = truncated.bool()

        # Match RSL-RL: bootstrap timeouts with the stored current value.
        stored_rewards = rewards.detach().clone()
        stored_rewards += (
            self.gamma
            * self.transition.values.squeeze(-1)
            * truncated.to(dtype=stored_rewards.dtype)
        )

        self.transition.rewards = stored_rewards
        self.transition.dones = (
            terminated | truncated
        ).detach().clone()

        self.storage.add_transition(self.transition)
        self.transition = None

    @torch.no_grad()
    def compute_returns(self, observations: Mapping) -> None:
        """Compute GAE with RSL-RL after collecting a complete rollout."""

        if self.transition is not None:
            raise RuntimeError("The last environment step is not recorded")
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("Collect a complete rollout before computing returns")

        packed = pack_policy_observations(
            observations,
            encoder_mode=self.policy.encoder_mode,
        )

        # Ask inherited GAE for raw advantages only while computing returns;
        # our optimizer always consumes rollout-normalized advantages.
        self.normalize_advantage_per_mini_batch = distributed.world_size() > 1
        try:
            super().compute_returns(packed)
        finally:
            self.normalize_advantage_per_mini_batch = False
        if distributed.world_size() > 1:
            # The inherited GAE leaves advantages raw when per-minibatch
            # normalization is enabled. Normalize once over the global rollout.
            advantages = self.storage.advantages
            count, average, variance = distributed.moments(
                advantages.reshape(-1, 1).double()
            )
            # Match RSL-RL's sample standard deviation and epsilon.
            std = (variance * count / (count - 1).clamp_min(1)).sqrt()
            advantages.sub_(average.to(advantages.dtype)).div_(
                std.to(advantages.dtype) + 1e-8
            )
        self.returns_ready = True

    def train_mode(self) -> None:
        self.policy.train()

    def eval_mode(self) -> None:
        self.policy.eval()

    def update(self) -> dict[str, float]:
        """Optimize PPO."""

        if self.transition is not None or not self.returns_ready:
            raise RuntimeError(
                "Finish the rollout and compute returns before update()"
            )

        self.policy.train()

        totals = {
            "surrogate": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": 0.0,
        }
        num_updates = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for batch in generator:
            actor_observations, morphology = unpack_policy_observations(
                batch.observations,
                encoder_mode=self.policy.encoder_mode,
                group="actor",
            )
            critic_observations, _ = unpack_policy_observations(
                batch.observations,
                encoder_mode=self.policy.encoder_mode,
                group="critic",
            )

            output = self.policy.evaluate_actions(
                actor_observations=actor_observations,
                critic_observations=critic_observations,
                pre_sigmoid=batch.actions,
                morphology=morphology,
            )

            # Keep scalar quantities shaped (batch_size,).
            old_log_prob = batch.old_actions_log_prob.squeeze(-1)
            old_values = batch.values.squeeze(-1)
            returns = batch.returns.squeeze(-1)
            advantages = batch.advantages.squeeze(-1)

            # Exact KL(old || new) for the underlying diagonal Gaussians.
            # Applying the same invertible sigmoid preserves this KL.
            with torch.no_grad():
                old_mean, old_std = batch.old_distribution_params
                new_mean = output["mean"]
                new_std = output["std"]

                kl = (
                    torch.log(new_std / old_std)
                    + (
                        old_std.square()
                        + (old_mean - new_mean).square()
                    ) / (2.0 * new_std.square())
                    - 0.5
                ).sum(dim=-1).mean()

                kl = distributed.mean(kl)
                if not torch.isfinite(kl):
                    raise FloatingPointError("Non-finite policy KL")

                # Match RSL-RL's adaptive learning-rate rule.
                if self.desired_kl is not None:
                    if kl > 2.0 * self.desired_kl:
                        self.learning_rate = max(
                            1e-5, self.learning_rate / 1.5
                        )
                    elif 0.0 < kl < 0.5 * self.desired_kl:
                        self.learning_rate = min(
                            1e-2, self.learning_rate * 1.5
                        )

                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            # PPO clipped surrogate loss.
            ratio = torch.exp(output["log_prob"] - old_log_prob)

            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.maximum(
                surrogate, surrogate_clipped
            ).mean()

            # Value loss, optionally clipped around rollout values.
            values = output["values"]
            value_errors = (values - returns).square()

            if self.use_clipped_value_loss:
                clipped_values = old_values + (
                    values - old_values
                ).clamp(-self.clip_param, self.clip_param)

                clipped_errors = (
                    clipped_values - returns
                ).square()

                value_loss = torch.maximum(
                    value_errors, clipped_errors
                ).mean()
            else:
                value_loss = value_errors.mean()

            entropy = output["entropy"].mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
            )

            distributed.require_finite(loss, "PPO loss")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            distributed.average_gradients(self.parameters)

            # Shared encoder parameters appear only once in this list.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()

            totals["surrogate"] += surrogate_loss.detach().item()
            totals["value"] += value_loss.detach().item()
            totals["entropy"] += entropy.detach().item()
            totals["kl"] += kl.item()
            totals["grad_norm"] += grad_norm.item()
            num_updates += 1

        values = distributed.mean(torch.tensor(
            [total / num_updates for total in totals.values()],
            device=self.parameters[0].device,
            dtype=torch.float64,
        )).tolist()
        metrics = dict(zip(totals, values))
        metrics["learning_rate"] = self.learning_rate

        self.storage.clear()
        self.returns_ready = False

        return metrics

    def save(self) -> dict:
        """
        Snapshot algorithm state after a completed PPO update.

        File paths, experiment configuration and runner state are managed
        by the training runner.
        """
        if (
            self.transition is not None
            or self.storage.step != 0
            or self.returns_ready
        ):
            raise RuntimeError(
                "Save checkpoints only between completed PPO updates"
            )

        from copy import deepcopy

        return {
            "encoder_mode": self.policy.encoder_mode,
            "policy_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.policy.state_dict().items()
            },
            "optimizer_state_dict": deepcopy(
                self.optimizer.state_dict()
            ),
            "ppo_config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "lam": self.lam,
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "clip_param": self.clip_param,
                "value_loss_coef": self.value_loss_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
                "use_clipped_value_loss": self.use_clipped_value_loss,
                "desired_kl": self.desired_kl,
                "normalize_advantage_per_mini_batch": (
                    self.normalize_advantage_per_mini_batch
                ),
            },
        }

    def load(
        self,
        checkpoint: Mapping,
        *,
        load_optimizer: bool = True,
    ) -> None:
        """
        Restore a checkpoint into an already constructed matching policy.

        load_optimizer=True resumes algorithm state.
        load_optimizer=False loads policy parameters and buffers only.
        """
        if (
            self.transition is not None
            or self.storage.step != 0
            or self.returns_ready
        ):
            raise RuntimeError(
                "Load checkpoints only when rollout storage is empty"
            )

        if checkpoint["encoder_mode"] != self.policy.encoder_mode:
            raise ValueError(
                "Checkpoint encoder mode does not match the policy"
            )

        config = checkpoint["ppo_config"]

        if load_optimizer:
            rollout_size = (
                self.storage.num_envs
                * self.storage.num_transitions_per_env
            )
            num_mini_batches = config["num_mini_batches"]

            if (
                num_mini_batches <= 0
                or rollout_size % num_mini_batches != 0
            ):
                raise ValueError(
                    "Current rollout size is incompatible with "
                    "checkpoint num_mini_batches"
                )

        self.policy.load_state_dict(
            checkpoint["policy_state_dict"],
            strict=True,
        )

        if load_optimizer:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            for name in (
                "learning_rate",
                "gamma",
                "lam",
                "num_learning_epochs",
                "num_mini_batches",
                "clip_param",
                "value_loss_coef",
                "entropy_coef",
                "max_grad_norm",
                "use_clipped_value_loss",
                "desired_kl",
                "normalize_advantage_per_mini_batch",
            ):
                setattr(self, name, config[name])
            self.normalize_advantage_per_mini_batch = False

            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
