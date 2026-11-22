"""Termination conditions for MyoFullBody motion tracking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as common_mdp
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def anchor_orientation_failure(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when anchor rotation error exceeds threshold radians."""
    command = env.command_manager.get_term(command_name)

    error = quat_error_magnitude(
        command.anchor_quat_w,
        command.robot_anchor_quat_w,
    )

    return error > threshold


def motion_finished(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Truncate when the current reference frame reaches the clip endpoint."""
    command = env.command_manager.get_term(command_name)

    return command.time_steps >= command.motion_lengths - 1


def anchor_position_failure(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_ratio: float,
) -> torch.Tensor:
    """Compare global anchor position error against subject height."""
    command = env.command_manager.get_term(command_name)

    error = torch.linalg.vector_norm(
        command.anchor_pos_w - command.robot_anchor_pos_w,
        dim=-1,
    )

    return error > command.subject_height_m * height_ratio


class MeanBodyPositionFailure:
    """Terminate on excessive mean aligned body-position error."""

    def __init__(
        self,
        cfg: TerminationTermCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        command = env.command_manager.get_term(
            cfg.params["command_name"]
        )
        body_names = tuple(cfg.params["body_names"])

        if not body_names or len(set(body_names)) != len(body_names):
            raise ValueError("body_names must be nonempty and unique")

        missing = set(body_names) - set(command.cfg.body_names)
        if missing:
            raise ValueError(f"Bodies missing from motion command: {sorted(missing)}")

        self._body_ids = [
            command.cfg.body_names.index(name)
            for name in body_names
        ]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        body_names: tuple[str, ...],
        height_ratio: float,
    ) -> torch.Tensor:
        command = env.command_manager.get_term(command_name)

        # Refresh alignment using the current robot anchor.
        command.update_relative_body_poses()

        error = torch.linalg.vector_norm(
            command.body_pos_relative_w[:, self._body_ids]
            - command.robot_body_pos_w[:, self._body_ids],
            dim=-1,
        ).mean(dim=-1)

        return error > command.subject_height_m * height_ratio


def make_myofullbody_termination_cfg(
    *,
    body_names: tuple[str, ...],
    anchor_position_ratio: float,
    body_position_ratio: float,
    anchor_orientation_threshold: float = 1.0,
    command_name: str = "motion",
) -> dict[str, TerminationTermCfg]:
    """
    Configure tracking failures and time truncations.

    Position thresholds are fractions of current subject height.
    Orientation threshold is in radians.
    """
    anchor_ratio = float(anchor_position_ratio)
    body_ratio = float(body_position_ratio)
    angle_threshold = float(anchor_orientation_threshold)
    body_names = tuple(body_names)

    for name, value in (
        ("anchor_position_ratio", anchor_ratio),
        ("body_position_ratio", body_ratio),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("body_names must be nonempty and unique")

    if (
        not math.isfinite(angle_threshold)
        or not 0 < angle_threshold <= math.pi
    ):
        raise ValueError(
            "anchor_orientation_threshold must be in (0, pi]"
        )

    return {
        "anchor_position": TerminationTermCfg(
            func=anchor_position_failure,
            params={
                "command_name": command_name,
                "height_ratio": anchor_ratio,
            },
        ),
        "anchor_orientation": TerminationTermCfg(
            func=anchor_orientation_failure,
            params={
                "command_name": command_name,
                "threshold": angle_threshold,
            },
        ),
        "body_position": TerminationTermCfg(
            func=MeanBodyPositionFailure,
            params={
                "command_name": command_name,
                "body_names": body_names,
                "height_ratio": body_ratio,
            },
        ),
        "time_out": TerminationTermCfg(
            func=common_mdp.time_out,
            time_out=True,
        ),
        "motion_finished": TerminationTermCfg(
            func=motion_finished,
            params={"command_name": command_name},
            time_out=True,
        ),
    }
