"""Observations for muscle-driven MyoFullBody motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.envs import mdp as common_mdp
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.tasks.tracking.mdp import observations as tracking_obs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class MuscleStateObservation:
    """
    Read muscle states in the entity's XML actuator order.

    Output layout:
        [lengths, velocities, forces, excitations, activations]

    Each block contains one value per muscle actuator.
    Values are returned without normalization.
    """

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        entity_name = cfg.params.get("entity_name", "robot")
        robot = env.scene[entity_name]
        model = env.sim.mj_model

        # Resolve CPU model metadata only during initialization.
        actuator_ids = robot.data.indexing.ctrl_ids
        actuator_ids_cpu = actuator_ids.detach().cpu().tolist()

        activation_ids = []

        for actuator_id in actuator_ids_cpu:
            if (
                model.actuator_dyntype[actuator_id]
                != mujoco.mjtDyn.mjDYN_MUSCLE
            ):
                raise ValueError(f"Actuator {actuator_id} is not a muscle actuator")

            activation_start = int(model.actuator_actadr[actuator_id])
            activation_count = int(model.actuator_actnum[actuator_id])

            if activation_start < 0 or activation_count != 1:
                raise ValueError(f"Actuator {actuator_id}: expected one muscle activation state")

            activation_ids.append(activation_start)

        self._actuator_ids = actuator_ids.to(
            device=env.device,
            dtype=torch.long,
        )
        self._activation_ids = torch.tensor(
            activation_ids,
            device=env.device,
            dtype=torch.long,
        )


    def __call__(
        self,
        env: ManagerBasedRlEnv,
        entity_name: str = "robot",
    ) -> torch.Tensor:
        # entity_name is consumed when resolving indices in __init__.
        data = env.sim.data
        actuator_ids = self._actuator_ids

        return torch.cat(
            (
                data.actuator_length[:, actuator_ids],
                data.actuator_velocity[:, actuator_ids],
                data.actuator_force[:, actuator_ids],
                data.ctrl[:, actuator_ids],
                data.act[:, self._activation_ids],
            ),
            dim=-1,
        )


def motion_phase(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Normalized progress through the reference clip, shape (num_envs, 1)."""
    command = env.command_manager.get_term(command_name)

    last_frame = (command.motion_lengths - 1).clamp_min(1)

    return (
        command.time_steps.to(dtype=torch.float32) / last_frame
    ).unsqueeze(-1)


def motion_lookahead(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Read future reference states from the motion command."""
    command = env.command_manager.get_term(command_name)
    return command.lookahead


def morphology_features(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """
    Return physical morphology features, shape (num_envs, 18).

    Log transformation and fixed normalization belong to the network.
    """
    command = env.command_manager.get_term(command_name)
    return command.morphology_features


def make_myofullbody_observation_cfg(
    *,
    command_name: str = "motion",
    enable_corruption: bool = False,
    enable_lookahead: bool = True,
    enable_touch: bool = True,
    enable_phase: bool = True,
) -> dict[str, ObservationGroupCfg]:
    """
    Return named dynamic groups for category-specific encoders.

    Actor and critic receive dictionaries.
    Morphology remains a separate physical feature tensor.
    """

    def make_group() -> ObservationGroupCfg:
        terms = {
            "root": ObservationTermCfg(
                func=root_state,
            ),
            "joint": ObservationTermCfg(
                func=joint_tracking_state,
                params={"command_name": command_name},
            ),
            "muscle": ObservationTermCfg(
                func=MuscleStateObservation,
                params={"entity_name": "robot"},
            ),
            "reference": ObservationTermCfg(
                func=reference_state,
                params={
                    "command_name": command_name,
                    "enable_phase": enable_phase,
                },
            ),
        }

        if enable_touch:
            terms["contact"] = ObservationTermCfg(
                func=foot_contact_state,
            )

        if enable_lookahead:
            terms["lookahead"] = ObservationTermCfg(
                func=motion_lookahead,
                params={"command_name": command_name},
            )

        return ObservationGroupCfg(
            terms=terms,
            concatenate_terms=False,
            enable_corruption=enable_corruption,
        )

    return {
        "actor": make_group(),
        "critic": make_group(),
        "morphology": ObservationGroupCfg(
            terms={
                "features": ObservationTermCfg(
                    func=morphology_features,
                    params={"command_name": command_name},
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }


def root_state(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Root gravity direction and local velocities, shape (N, 9)."""
    return torch.cat(
        (
            common_mdp.projected_gravity(env),
            common_mdp.base_lin_vel(env),
            common_mdp.base_ang_vel(env),
        ),
        dim=-1,
    )


def joint_tracking_state(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """
    Actual, reference, and error joint states, shape (N, 6J).

    All positions use the same model joint coordinates.
    No default-pose subtraction is applied.
    """
    command = env.command_manager.get_term(command_name)

    actual_pos = command.robot_joint_pos
    actual_vel = command.robot_joint_vel
    reference_pos = command.joint_pos
    reference_vel = command.joint_vel

    return torch.cat(
        (
            actual_pos,
            actual_vel,
            reference_pos,
            reference_vel,
            reference_pos - actual_pos,
            reference_vel - actual_vel,
        ),
        dim=-1,
    )


def reference_state(
    env: ManagerBasedRlEnv,
    command_name: str,
    enable_phase: bool = True,
) -> torch.Tensor:
    """Current anchor target and optional clip progress, shape (N, 9 or 10)."""
    values = [
        tracking_obs.motion_anchor_pos_b(env, command_name),
        tracking_obs.motion_anchor_ori_b(env, command_name),
    ]

    if enable_phase:
        values.append(motion_phase(env, command_name))

    return torch.cat(values, dim=-1)


def foot_contact_state(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Touch forces ordered as right foot/toes, left foot/toes."""
    return torch.cat(
        [
            common_mdp.builtin_sensor(
                env,
                sensor_name=f"robot/{name}",
            )
            for name in (
                "r_foot",
                "r_toes",
                "l_foot",
                "l_toes",
            )
        ],
        dim=-1,
    )

