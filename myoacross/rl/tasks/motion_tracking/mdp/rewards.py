"""Additional rewards for muscle-driven motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math
from collections.abc import Mapping

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.tasks.tracking.mdp import rewards as tracking_rewards

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def joint_position_tracking_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Track selected anatomical hinge-joint positions.

    std is in radians. Reference and simulated positions use the
    same joint ordering and coordinate convention.
    """
    command = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    error = (
        robot.data.joint_pos[:, joint_ids]
        - command.joint_pos[:, joint_ids]
    ).square().mean(dim=-1)

    return torch.exp(-error / std**2)


def joint_velocity_tracking_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track selected hinge-joint velocities; std is in rad/s."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    error = (
        robot.data.joint_vel[:, joint_ids]
        - command.joint_vel[:, joint_ids]
    ).square().mean(dim=-1)

    return torch.exp(-error / std**2)


def excitation_change_mean_square(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """
    Mean squared change of policy excitation between control steps.

    Assumes policy actions are bounded excitations in [0, 1].
    This measures discrete changes, not a time derivative.
    """
    actions = env.action_manager.action
    previous_actions = env.action_manager.prev_action

    return (actions - previous_actions).square().mean(dim=-1)


def make_myofullbody_reward_cfg(
    *,
    body_names: tuple[str, ...],
    joint_names: tuple[str, ...] = (),
    command_name: str = "motion",
    weights: Mapping[str, float] | None = None,
    stds: Mapping[str, float] | None = None,
    excitation_change_weight: float = 0.0,
) -> dict[str, RewardTermCfg]:
    """
    Build mjlab tracking rewards.

    body_names selects anatomical bodies used in body-level rewards.
    Every selected body must also be included in the motion command.

    weights and stds contain optional overrides, e.g. from YAML.
    """
    body_names = tuple(body_names)

    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("body_names must be nonempty and unique")

    functions = {
        "anchor_position":
            tracking_rewards.motion_global_anchor_position_error_exp,
        "anchor_orientation":
            tracking_rewards.motion_global_anchor_orientation_error_exp,
        "body_position":
            tracking_rewards.motion_relative_body_position_error_exp,
        "body_orientation":
            tracking_rewards.motion_relative_body_orientation_error_exp,
        "body_linear_velocity":
            tracking_rewards.motion_global_body_linear_velocity_error_exp,
        "body_angular_velocity":
            tracking_rewards.motion_global_body_angular_velocity_error_exp,
        "joint_position": 
            joint_position_tracking_exp,
        "joint_velocity": 
            joint_velocity_tracking_exp,
    }

    resolved_weights = {
        "anchor_position": 0.5,
        "anchor_orientation": 0.5,
        "body_position": 1.0,
        "body_orientation": 1.0,
        "body_linear_velocity": 1.0,
        "body_angular_velocity": 1.0,
        "joint_position": 0.0,
        "joint_velocity": 0.0,
    }

    resolved_stds = {
        "anchor_position": 0.3,
        "anchor_orientation": 0.4,
        "body_position": 0.3,
        "body_orientation": 0.4,
        "body_linear_velocity": 1.0,
        "body_angular_velocity": 3.14,
        "joint_position": 0.3,
        "joint_velocity": 1.0,
    }

    for label, overrides, defaults in (
        ("weights", weights, resolved_weights),
        ("stds", stds, resolved_stds),
    ):
        if overrides is None:
            continue

        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown reward {label}: {sorted(unknown)}")

        defaults.update(overrides)

    rewards = {}

    for name, func in functions.items():
        weight = float(resolved_weights[name])
        std = float(resolved_stds[name])

        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name}: tracking weight must be finite and nonnegative")

        if not math.isfinite(std) or std <= 0:
            raise ValueError(f"{name}: std must be finite and positive")

        if weight == 0:
            continue

        params = {
            "command_name": command_name,
            "std": std,
        }

        if name.startswith("body_"):
            params["body_names"] = body_names
        elif name.startswith("joint_"):
            if not joint_names:
                raise ValueError("joint_names must be specified when joint tracking rewards are enabled")

            params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=tuple(joint_names),
                preserve_order=True,
            )

        rewards[name] = RewardTermCfg(
            func=func,
            weight=weight,
            params=params,
        )

    excitation_change_weight = float(excitation_change_weight)

    if (
        not math.isfinite(excitation_change_weight)
        or excitation_change_weight < 0
    ):
        raise ValueError("excitation_change_weight must be finite and nonnegative")

    if excitation_change_weight > 0:
        rewards["excitation_change"] = RewardTermCfg(
            func=excitation_change_mean_square,
            weight=-excitation_change_weight,
        )

    return rewards


