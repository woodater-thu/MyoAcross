"""Build scene/MDP configurations and restore offline simulation settings.

Runtime environment creation and bundle switching live in environment.py.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sim.sim import MujocoCfg, SimulationCfg
from mjlab.utils.nan_guard import NanGuardCfg

from myoacross.rl.embodiments.myofullbody_cfg import (
    SubjectSource,
    make_myofullbody_cfg,
)

from myoacross.rl.tasks.motion_tracking.mdp.actions import (
    make_muscle_action_cfg,
)
from myoacross.rl.tasks.motion_tracking.mdp.commands import (
    MyoFullBodyMotionCommandCfg,
)
from myoacross.rl.tasks.motion_tracking.mdp.observations import (
    make_myofullbody_observation_cfg,
)
from myoacross.rl.tasks.motion_tracking.mdp.rewards import (
    make_myofullbody_reward_cfg,
)
from myoacross.rl.tasks.motion_tracking.mdp.terminations import (
    make_myofullbody_termination_cfg,
)

def make_myofullbody_scene_cfg(
    subject: SubjectSource,
    *,
    num_envs: int = 1,
    env_spacing: float = 2.0,
    init_state: EntityCfg.InitialStateCfg | None = None,
) -> SceneCfg:
    """
    Configure the reference scene before simulation initialization.

    The supplied XML provides the model and its existing floor.
    Subsequent subject changes use bundle injection, not scene rebuilding.
    """

    if isinstance(num_envs, bool) or not isinstance(num_envs, int):
        raise TypeError("num_envs must be an integer")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")

    if not math.isfinite(env_spacing) or env_spacing <= 0:
        raise ValueError("env_spacing must be finite and positive")

    return SceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        terrain=None,
        entities={
            "robot": make_myofullbody_cfg(
                subject,
                init_state=init_state,
            ),
        },
    )


def make_myofullbody_env_cfg(
    subject: SubjectSource,
    *,
    motion: MyoFullBodyMotionCommandCfg,
    simulation_config: SimulationCfg,
    decimation: int,
    episode_length_s: float,
    tracking_body_names: tuple[str, ...],
    termination_options: Mapping[str, Any],
    tracking_joint_names: tuple[str, ...] = (),
    observation_options: Mapping[str, Any] | None = None,
    reward_options: Mapping[str, Any] | None = None,
    num_envs: int = 1,
    env_spacing: float = 2.0,
    seed: int = 42,
    init_state: EntityCfg.InitialStateCfg | None = None,
) -> ManagerBasedRlEnvCfg:
    """
    Assemble a single-subject motion-tracking environment configuration.

    Configuration is built once before environment initialization.
    Runtime subject switching must use bundle injection.
    """

    # Validate inputs
    if (
        isinstance(decimation, bool)
        or not isinstance(decimation, int)
        or decimation < 1
    ):
        raise ValueError("decimation must be a positive integer")

    if (
        not math.isfinite(episode_length_s)
        or episode_length_s <= 0
    ):
        raise ValueError("episode_length_s must be finite and positive")

    if motion.entity_name != "robot":
        raise ValueError("Motion command must use entity_name='robot'")

    command_bodies = tuple(motion.body_names)
    tracking_bodies = tuple(tracking_body_names)

    if (
        not command_bodies
        or command_bodies[0] != "Full Body"
        or len(set(command_bodies)) != len(command_bodies)
    ):
        raise ValueError(
            "Motion body_names must be unique and begin with 'Full Body'"
        )

    if motion.anchor_body_name not in command_bodies:
        raise ValueError("Motion anchor must be included in body_names")

    if (
        not tracking_bodies
        or len(set(tracking_bodies)) != len(tracking_bodies)
        or "Full Body" in tracking_bodies
    ):
        raise ValueError("tracking_body_names must contain unique anatomical bodies")

    missing = set(tracking_bodies) - set(command_bodies)
    if missing:
        raise ValueError(f"Tracking bodies missing from motion command: {sorted(missing)}")

    # Make mdp cfgs
    observations = make_myofullbody_observation_cfg(
        command_name="motion",
        **dict(observation_options or {}),
    )

    rewards = make_myofullbody_reward_cfg(
        command_name="motion",
        body_names=tracking_bodies,
        joint_names=tuple(tracking_joint_names),
        **dict(reward_options or {}),
    )

    terminations = make_myofullbody_termination_cfg(
        command_name="motion",
        body_names=tracking_bodies,
        **dict(termination_options),
    )

    return ManagerBasedRlEnvCfg(
        scene=make_myofullbody_scene_cfg(
            subject,
            num_envs=num_envs,
            env_spacing=env_spacing,
            init_state=init_state,
        ),
        sim=deepcopy(simulation_config),
        decimation=decimation,
        episode_length_s=episode_length_s,
        seed=seed,
        actions={"muscle": make_muscle_action_cfg(subject),},
        commands={"motion": deepcopy(motion),},
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        auto_reset=True,
        is_finite_horizon=False,
        scale_rewards_by_dt=True,
    )



def simulation_config_from_dict(
    record: Mapping[str, object],
) -> SimulationCfg:
    """Restore the simulation settings saved by offline generation."""
    options = dict(record)

    mujoco_options = options.pop("mujoco")
    nan_guard_options = options.pop("nan_guard")

    if not isinstance(mujoco_options, Mapping):
        raise TypeError("simulation_config.mujoco must be a mapping")
    if not isinstance(nan_guard_options, Mapping):
        raise TypeError("simulation_config.nan_guard must be a mapping")

    mujoco_options = dict(mujoco_options)

    # JSON stores tuples as lists.
    for name in ("gravity", "disableflags", "enableflags"):
        if name in mujoco_options:
            mujoco_options[name] = tuple(mujoco_options[name])

    if options.get("broadphase_filter") is not None:
        options["broadphase_filter"] = tuple(
            options["broadphase_filter"]
        )

    return SimulationCfg(
        mujoco=MujocoCfg(**mujoco_options),
        nan_guard=NanGuardCfg(**dict(nan_guard_options)),
        **options,
    )
