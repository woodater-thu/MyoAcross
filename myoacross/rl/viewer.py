"""Display policy-controlled rollouts using mjlab's native or Viser viewer."""

from copy import deepcopy
import math
from pathlib import Path

import mujoco
import torch

from myoacross.rl.networks.rsl_adapter import (
    unpack_policy_observations,
)
from myoacross.embodiment.measurements import (
    compute_myofullbody_height,
)
from myoacross.rl.embodiments.features import MorphologyFeatures
from myoacross.rl.networks.actor_critic import load_policy
from myoacross.rl.tasks.motion_tracking.environment import (
    create_tracking_environment,
)
from myoacross.rl.tasks.motion_tracking.myofullbody_env_cfg import (
    simulation_config_from_dict,
)


def _initialize_inference(
    subject_xml: str | Path,
    motion_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str = "cuda:0",
    seed: int = 42,
):
    """
    Create one subject environment and restore its policy.

    The caller owns the returned environment and must close it.
    Environment and simulation settings come from the RL checkpoint.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    encoder_mode = checkpoint["algorithm"]["encoder_mode"]
    if "simulation" not in config:
        raise ValueError(
            "Checkpoint has no saved simulation configuration; "
            "older checkpoints require their original training simulation settings"
        )
    environment_options = config["environment"]
    simulation_config = simulation_config_from_dict(config["simulation"])
    del checkpoint

    subject_xml = Path(subject_xml).expanduser().resolve()

    # Measure the standalone subject, before adding scene objects.
    model = mujoco.MjModel.from_xml_path(str(subject_xml))
    features = MorphologyFeatures.from_model(model)
    height = compute_myofullbody_height(model)

    options = deepcopy(dict(environment_options))
    options["num_envs"] = 1
    options["sampling_mode"] = "start"

    observation_options = dict(options.get("observation_options") or {})
    observation_options["enable_corruption"] = False
    options["observation_options"] = observation_options

    env = create_tracking_environment(
        subject_xml=subject_xml,
        motion_path=motion_path,
        subject_features=features,
        subject_height_m=height,
        simulation_config=simulation_config,
        seed=seed,
        device=device,
        **options,
    )

    try:
        command = env.command_manager.get_term("motion")
        # Explicitly disable reference-state initialization noise.
        command.cfg.pose_range = {}
        command.cfg.velocity_range = {}
        command.cfg.joint_position_range = (0.0, 0.0)

        observations, _ = env.reset()

        dynamic, _ = unpack_policy_observations(
            observations,
            encoder_mode=encoder_mode,
        )

        policy = load_policy(
            checkpoint_path,
            observations=dynamic,
            action_dim=env.action_manager.total_action_dim,
            device=device,
        )

        if policy.encoder_mode != encoder_mode:
            raise ValueError("Inference encoder mode differs from checkpoint")

        return env, policy, observations

    except BaseException:
        env.close()
        raise

def view_policy(
    xml_path: str | Path,
    motion_path: str | Path,
    checkpoint_path: str | Path,
    *,
    fps: float = 60.0,
    device: str = "cuda:0",
    viewer: str = "native",
    camera: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Play a deterministic policy with automatic episode resets."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    if camera not in {None, "front-side"}:
        raise ValueError(f"Unknown camera preset: {camera}")

    if viewer not in {"native", "viser"}:
        raise ValueError(f"Unknown viewer: {viewer}")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    env, policy, _ = _initialize_inference(
        subject_xml=xml_path,
        motion_path=motion_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    server = None
    try:
        if camera == "front-side":
            env.cfg.viewer.azimuth = -45.0
            env.cfg.viewer.elevation = -12.0
            env.cfg.viewer.distance = 3.7
            env.cfg.viewer.lookat = (0.0, 0.0, 0.95)

        @torch.inference_mode()
        def policy_action(observations):
            dynamic, morphology = unpack_policy_observations(
                observations,
                encoder_mode=policy.encoder_mode,
            )
            return policy.act_inference(dynamic, morphology)

        if viewer == "viser":
            import viser
            from mjlab.viewer import ViserPlayViewer

            server = viser.ViserServer(
                host=host, port=port,
                label=f"MyoAcross · {Path(checkpoint_path).name}",
            )
            play_viewer = ViserPlayViewer(
                env, policy_action, frame_rate=fps, viser_server=server,
            )
            with server.gui.add_folder("Policy"):
                server.gui.add_markdown(
                    f"**Checkpoint:** `{Path(checkpoint_path).name}`\n\n"
                    f"**Subject:** `{Path(xml_path).parent.name}`\n\n"
                    f"**Motion:** `{Path(motion_path).parent.name}`\n\n"
                    f"**Device:** `{device}`"
                )
        else:
            from mjlab.viewer import NativeMujocoViewer

            play_viewer = NativeMujocoViewer(
                env, policy_action, frame_rate=fps,
                enable_perturbations=False,
            )
        play_viewer.run()
    finally:
        if server is not None:
            server.stop()
        env.close()
