"""Create the tracking environment and switch its active virtual embodiment.

The environment is created once. Model changes inject prepared Warp parameters
and refresh entity caches. Motion changes reuse the model and reset all worlds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim import SimulationCfg

from myoacross.rl.embodiments.training_data import RolloutSelection
from myoacross.rl.embodiments.bundle import PreparedWarpBundle, WarpBundleSwitcher
from myoacross.rl.embodiments.features import MorphologyFeatures
from myoacross.rl.tasks.motion_tracking.mdp.commands import (
    MyoFullBodyMotionCommand,
    MyoFullBodyMotionCommandCfg,
)
from myoacross.rl.tasks.motion_tracking.myofullbody_env_cfg import (
    make_myofullbody_env_cfg,
)


def create_tracking_environment(
    subject_xml: str | Path,
    motion_path: str | Path,
    *,
    subject_features: MorphologyFeatures,
    subject_height_m: float,
    motion_paths: Sequence[str | Path] | None = None,
    simulation_config: SimulationCfg,
    tracking_body_names: Sequence[str],
    anchor_body_name: str,
    decimation: int,
    episode_length_s: float,
    termination_options: Mapping[str, object],
    num_envs: int,
    seed: int,
    device: str,
    nconmax: int = 128,
    njmax: int = 512,
    tracking_joint_names: Sequence[str] = (),
    observation_options: Mapping[str, object] | None = None,
    reward_options: Mapping[str, object] | None = None,
    lookahead_times_s: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    sampling_mode: str = "adaptive",
    env_spacing: float = 2.0,
) -> ManagerBasedRlEnv:
    """
    Create the initial simulation once.

    XML establishes the compatible scene topology and motion comes from the
    source subject. Features and height describe the selected virtual body;
    inject its matching bundle before collecting policy observations.

    Later subject switches must update the existing environment,
    rather than call this function again.

    Contact and constraint capacities override the offline allocation settings
    before environment creation; model physics settings remain unchanged.
    """
    for name, value in (("nconmax", nconmax), ("njmax", njmax)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    simulation_config = replace(
        simulation_config,
        nconmax=nconmax,
        njmax=njmax,
    )

    subject_xml = Path(subject_xml).expanduser().resolve()
    motion_path = Path(motion_path).expanduser().resolve()

    for path in (subject_xml, motion_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if sampling_mode not in {"adaptive", "uniform", "start"}:
        raise ValueError(f"Unknown motion sampling mode: {sampling_mode}")

    tracking_bodies = tuple(tracking_body_names)

    if not tracking_bodies or len(set(tracking_bodies)) != len(tracking_bodies):
        raise ValueError("tracking_body_names must be non-empty and unique")

    if "Full Body" in tracking_bodies:
        raise ValueError("tracking_body_names should contain anatomical bodies, not the free-root body")

    # The command needs the free-root body first for reference-state reset.
    # The anchor may be needed even when it has no body-tracking reward.
    command_bodies = tuple(dict.fromkeys(
        ("Full Body", anchor_body_name, *tracking_bodies)
    ))

    motion_config = MyoFullBodyMotionCommandCfg(
        entity_name="robot",
        motion_file=str(motion_path),
        motion_files=tuple(str(path) for path in motion_paths) if motion_paths is not None else (),
        anchor_body_name=anchor_body_name,
        body_names=command_bodies,
        subject_features=subject_features,
        subject_height_m=subject_height_m,
        lookahead_times_s=tuple(lookahead_times_s),
        sampling_mode=sampling_mode,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )

    env_config = make_myofullbody_env_cfg(
        subject=subject_xml,
        motion=motion_config,
        simulation_config=simulation_config,
        decimation=decimation,
        episode_length_s=episode_length_s,
        tracking_body_names=tracking_bodies,
        tracking_joint_names=tuple(tracking_joint_names),
        termination_options=termination_options,
        observation_options=observation_options,
        reward_options=reward_options,
        num_envs=num_envs,
        env_spacing=env_spacing,
        seed=seed,
    )

    env = ManagerBasedRlEnv(
        cfg=env_config,
        device=device,
    )
    from myoacross.rl.runners.multi_embodiment.zero_tendon import configure_zero_tendon_elision

    try:
        configure_zero_tendon_elision(env.sim)
    except BaseException:
        env.close()
        raise
    return env


@torch.no_grad()
def refresh_joint_limit_buffers(env: ManagerBasedRlEnv) -> None:
    """Refresh entity limit buffers after injecting model parameters."""
    robot = env.scene["robot"]

    # Entity joint order excludes the free root.
    joint_ids = robot.data.indexing.joint_ids.to(dtype=torch.long)

    # mjlab's model bridge exposes a world dimension.
    limits = env.sim.model.jnt_range[:, joint_ids]

    robot.data.joint_pos_limits.copy_(limits)

    factor = (
        robot.cfg.articulation.soft_joint_pos_limit_factor
        if robot.cfg.articulation is not None
        else 1.0
    )

    midpoint = limits.mean(dim=-1)
    half_range = 0.5 * (limits[..., 1] - limits[..., 0]) * factor

    robot.data.soft_joint_pos_limits[..., 0].copy_(
        midpoint - half_range
    )
    robot.data.soft_joint_pos_limits[..., 1].copy_(
        midpoint + half_range
    )


def switch_training_embodiment(
    env: ManagerBasedRlEnv,
    switcher: WarpBundleSwitcher,
    prepared: PreparedWarpBundle,
    selection: RolloutSelection,
    *,
    inject_model: bool = True,
) -> tuple[dict, dict]:
    """
    Switch all environments between completed PPO training blocks.

    The previous rollout's bootstrap and PPO update must be finished.
    A failure during switching must abort the step loop.
    Set inject_model=False only when prepared is the already active bundle.
    """
    bundle = prepared.bundle

    expected_subject_id = (
        None if bundle.embodiment == "baseline"
        else selection.subject.subject_id
    )
    if bundle.source_subject_id != expected_subject_id:
        raise ValueError("Prepared bundle has the wrong source subject")
    if bundle.variant_id != selection.variant_id:
        raise ValueError("Prepared bundle has the wrong variant")

    command = env.command_manager.get_term("motion")
    if not isinstance(command, MyoFullBodyMotionCommand):
        raise TypeError("Expected MyoFullBodyMotionCommand")

    # Complete previous device work before changing the active model.
    # Explicit synchronization is acceptable at a subject-switch boundary.
    wp.synchronize_device(env.sim.wp_model.body_mass.device)

    command.set_motions(tuple(motion.retarget_path for motion in selection.subject.motions))

    if inject_model:
        switcher.inject(prepared)

        # Ensure Warp copies finish before Torch reads the updated model.
        wp.synchronize_device(env.sim.wp_model.body_mass.device)

        refresh_joint_limit_buffers(env)

        command.set_subject_morphology(
            bundle.features,
            bundle.subject_height_m,
        )

    # Never attribute outgoing failures to the newly selected body/subject.
    command.reset_sampling_statistics()
    # Reset all environments; do not reseed at each switch.
    return env.reset()
