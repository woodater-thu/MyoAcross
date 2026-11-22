"""
Coordinate RL initialization and own the resulting runtime objects.

Data selection, environment operations and observation adaptation live in their
respective modules. This runner collects rollouts, updates PPO and logs metrics.
"""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import OmegaConf
from mjlab.envs import ManagerBasedRlEnv
import json

from rsl_rl.storage import RolloutStorage

from myoacross.rl.runners import distributed

from myoacross.rl.algorithms.morphology_ppo import MorphologyPPO
from myoacross.rl.embodiments.subjects import DatasetRecord, select_target_ids
from myoacross.utils.datalists import get_motion_specs
from myoacross.utils.paths import PROJECT_ROOT, resolve_path
from myoacross.rl.embodiments.training_data import (
    RolloutSelection,
    TrainingData,
)
from myoacross.rl.networks.morphology_encoder import MorphologyEncoderConfig
from myoacross.rl.embodiments.bundle import PreparedWarpBundle, WarpBundleSwitcher
from myoacross.rl.embodiments.features import (
    MorphologyFeatureNormalizer,
    load_morphology_features,
)
from myoacross.rl.networks.actor_critic import (
    EncoderMode,
    MorphologyActorCritic,
    build_actor_critic,
)
from myoacross.rl.networks.rsl_adapter import (
    pack_policy_observations,
    unpack_policy_observations,
)
from myoacross.rl.tasks.motion_tracking.environment import (
    create_tracking_environment,
    switch_training_embodiment,
)
from myoacross.rl.tasks.motion_tracking.motion_library import (
    clear_motion_clip_cache,
    preload_motion_clips,
)
from myoacross.rl.tasks.motion_tracking.myofullbody_env_cfg import (
    simulation_config_from_dict,
)


if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


@dataclass
class TrainingState:
    """Runtime objects created once before the PPO loop."""

    env: ManagerBasedRlEnv
    policy: MorphologyActorCritic
    training_data: TrainingData
    switcher: WarpBundleSwitcher
    selection: RolloutSelection
    prepared_bundle: PreparedWarpBundle
    observations: dict
    body_batch: object | None = None


def initialize_actor_critic(
    observations: Mapping[str, torch.Tensor],
    action_dim: int,
    *,
    encoder_mode: EncoderMode,
    train_index_paths: Sequence[str | Path],
    output_dir: str | Path,
    network_options: Mapping[str, object],
    device: str | torch.device,
    morphology_checkpoint: str | Path | None = None,
    morphology_encoder_config: MorphologyEncoderConfig | None = None,
    resume_checkpoint: Mapping | None = None,
) -> MorphologyActorCritic:
    """
    Initialize a policy for a new training run.

    Paths must already be resolved by the configuration entry point.
    observations contains dynamic groups, excluding morphology.
    The caller must set random seeds before calling this function.
    """
    if encoder_mode not in {"none", "rl", "concat"}:
        raise ValueError(f"Unknown encoder mode: {encoder_mode}")

    output_dir = Path(output_dir).expanduser().resolve()
    statistics_path = output_dir / "morphology_normalizer.json"

    if encoder_mode == "concat" and (
        morphology_checkpoint is not None or morphology_encoder_config is not None
    ):
        raise ValueError(
            "Concat does not use an morphology pretraining checkpoint or configuration"
        )

    statistics = None
    build_options = {"encoder_mode": encoder_mode}

    if resume_checkpoint is not None:
        build_options["policy_state_dict"] = resume_checkpoint["algorithm"][
            "policy_state_dict"
        ]
        if encoder_mode not in {"none", "concat"}:
            build_options["morphology_encoder_config"] = (
                MorphologyEncoderConfig.from_checkpoint(resume_checkpoint["config"])
            )

    elif encoder_mode == "none":
        if morphology_checkpoint is not None:
            raise ValueError(
                "Unconditional mode does not use an morphology pretraining checkpoint"
            )
        if morphology_encoder_config is not None:
            raise ValueError(
                "Unconditional mode does not use an morphology encoder configuration"
            )

    elif morphology_checkpoint is not None:
        raise ValueError(
            "External morphology pretraining checkpoints are not supported"
        )

    else:
        if morphology_encoder_config is None and encoder_mode != "concat":
            raise ValueError("Random initialization requires morphology_encoder_config")

        features, _ = load_morphology_features(train_index_paths)

        statistics = MorphologyFeatureNormalizer.fit(features)

        build_options.update(
            morphology_encoder_config=morphology_encoder_config,
            morphology_statistics=statistics,
        )

    # A statistics file here indicates an existing initialization.
    # Resume must restore the full RL checkpoint instead of refitting.
    if distributed.is_primary() and encoder_mode != "none" and statistics_path.exists():
        raise FileExistsError(statistics_path)

    # Network options must not override initialization arguments.
    reserved = {
        "observations",
        "action_dim",
        "encoder_mode",
        "morphology_checkpoint",
        "morphology_encoder_config",
        "morphology_statistics",
        "device",
    }
    overlap = reserved.intersection(network_options)
    if overlap:
        raise ValueError(
            f"Unexpected initialization keys in network options: {sorted(overlap)}"
        )

    policy = build_actor_critic(
        observations=observations,
        action_dim=action_dim,
        **build_options,
        **dict(network_options),
        device=device,
    )

    if encoder_mode != "none":
        if statistics is None:
            # Export the checkpoint's statistics for inspection.
            # The policy itself already stores them as Torch buffers.
            normalizer = policy.morphology_normalizer
            assert normalizer is not None

            statistics = MorphologyFeatureNormalizer(
                log_mean=tuple(normalizer.log_mean.detach().cpu().tolist()),
                log_scale=tuple(normalizer.log_scale.detach().cpu().tolist()),
            )

        distributed.primary_call(lambda: statistics.save(statistics_path))

    return policy


def initialize_training(
    *,
    train_index_paths: Sequence[str | Path],
    embodiment: str,
    morphology_source: str,
    output_dir: str | Path,
    encoder_mode: EncoderMode,
    network_options: Mapping[str, object],
    environment_options: Mapping[str, object],
    seed: int,
    device: str,
    morphology_checkpoint: str | Path | None = None,
    morphology_encoder_config: MorphologyEncoderConfig | None = None,
    resume_checkpoint: Mapping | None = None,
    motion_set: str = "kit_flat_locomotion_72",
    subject_ids: Sequence[str] | None = None,
    motion_ids: Sequence[str] | None = None,
    retargets_root: str | Path | None = None,
    require_quality_pass: bool = True,
) -> TrainingState:
    """
    Initialize the first virtual embodiment and actor–critic.

    Input paths must already be resolved.
    The caller owns the returned environment and must close it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    subjects = DatasetRecord.from_split(
        split="train",
        embodiment=embodiment,
        motion_set=motion_set,
        subject_ids=subject_ids,
        motion_ids=motion_ids,
        require_quality_pass=require_quality_pass,
        **({"retargets_root": retargets_root} if retargets_root is not None else {}),
    )
    if (
        resume_checkpoint is not None
        and resume_checkpoint["config"]["data"].get("catalog_fingerprint")
        != subjects.fingerprint
    ):
        raise ValueError(
            "Resume dataset fingerprint differs; retarget data or target models changed"
        )

    training_data = TrainingData(
        index_paths=train_index_paths,
        subjects=subjects,
        embodiment=embodiment,
        morphology_source=morphology_source,
        seed=seed,
    )

    selection = training_data.sample()
    bundle = training_data.load_bundle(selection)

    simulation_config = simulation_config_from_dict(
        resume_checkpoint["config"]["simulation"]
        if resume_checkpoint is not None
        else training_data.simulation_config
    )

    env = create_tracking_environment(
        # Preserve the fixed fields used during offline bundle generation.
        # The selected virtual body's parameters are injected before rollout.
        subject_xml=training_data.baseline_path,
        motion_path=selection.subject.motions[0].retarget_path,
        motion_paths=tuple(
            motion.retarget_path for motion in selection.subject.motions
        ),
        subject_features=bundle.features,
        subject_height_m=bundle.subject_height_m,
        simulation_config=simulation_config,
        seed=seed,
        device=device,
        **dict(environment_options),
    )

    try:
        switcher = WarpBundleSwitcher(env.sim.wp_model)

        prepared = switcher.prepare_bundle(
            bundle,
            env.sim.mj_model,
        )

        # Initial observations must come from the injected virtual body.
        observations, _ = switch_training_embodiment(
            env,
            switcher,
            prepared,
            selection,
        )

        dynamic, _ = unpack_policy_observations(
            observations,
            encoder_mode=encoder_mode,
        )

        policy = initialize_actor_critic(
            observations=dynamic,
            action_dim=env.action_manager.total_action_dim,
            encoder_mode=encoder_mode,
            train_index_paths=train_index_paths,
            output_dir=output_dir,
            network_options=network_options,
            device=device,
            morphology_checkpoint=morphology_checkpoint,
            morphology_encoder_config=morphology_encoder_config,
            resume_checkpoint=resume_checkpoint,
        )

        return TrainingState(
            env=env,
            policy=policy,
            training_data=training_data,
            switcher=switcher,
            selection=selection,
            prepared_bundle=prepared,
            observations=observations,
        )

    except BaseException:
        env.close()
        raise


def _save_training_checkpoint(path, algorithm, iteration, config) -> None:
    """Save the policy, optimizer, and configuration without replacing a file."""
    with path.open("xb") as checkpoint_file:
        torch.save(
            {
                "algorithm": algorithm.save(),
                "iteration": iteration,
                "config": config,
            },
            checkpoint_file,
        )


def train_policy(
    state: TrainingState,
    *,
    num_iterations: int,
    num_steps_per_env: int,
    updates_per_embodiment: int,
    algorithm_options: Mapping[str, object],
    output_dir: str | Path,
    resolved_config: Mapping[str, object],
    logs_dir: str | Path | None = None,
    save_interval: int = 100,
    embodiments_per_rank: int = 1,
    save_checkpoints: bool = True,
    wandb_run: Run | None = None,
    resume_checkpoint: Mapping | None = None,
) -> Path:
    """
    Train with equally sized groups of physical embodiments per rank.

    Keep the embodiment for several PPO updates; sample source clips per env.
    The caller owns the environment and must close it after this function.
    Resume restores algorithm state and absolute counters into a new output
    directory. Physics episodes and random generators start afresh.
    """

    for name, value in (
        ("num_iterations", num_iterations),
        ("num_steps_per_env", num_steps_per_env),
        ("updates_per_embodiment", updates_per_embodiment),
        ("save_interval", save_interval),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    output_dir = Path(output_dir).expanduser().resolve()
    start_iteration = 0 if resume_checkpoint is None else resume_checkpoint["iteration"]
    if not 0 <= start_iteration < num_iterations:
        raise ValueError(
            "Resume iteration must be below the total num_iterations target"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir if logs_dir is None else Path(logs_dir).expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "metrics.jsonl"
    final_path = output_dir / "final.pt"

    if distributed.is_primary() and (metrics_path.exists() or final_path.exists()):
        raise FileExistsError(f"Training output already exists in {output_dir}")

    if distributed.is_primary():
        (logs_dir / "config.yaml").write_text(
            OmegaConf.to_yaml(
                OmegaConf.create(dict(resolved_config)), resolve=True, sort_keys=False
            )
        )
    env = state.env
    policy = state.policy
    state.body_batch = getattr(state, "body_batch", None)
    if (
        isinstance(embodiments_per_rank, bool)
        or not isinstance(embodiments_per_rank, int)
        or embodiments_per_rank < 1
        or env.num_envs % embodiments_per_rank
    ):
        raise ValueError("embodiments_per_rank must be positive and divide num_envs")
    if embodiments_per_rank > 1 and state.body_batch is None:
        from myoacross.rl.runners.multi_embodiment.batched import BatchedEmbodiments

        state.body_batch = BatchedEmbodiments(state, embodiments_per_rank)
        state.body_batch.install(state.body_batch.sample())
    if state.body_batch is not None and state.body_batch.count != embodiments_per_rank:
        raise ValueError(
            "Configured embodiment count differs from installed body batch"
        )
    observations = state.observations

    # Preserve the settings actually used, including simulation capacities
    # and an AE architecture potentially restored from a separate checkpoint.
    checkpoint_config = deepcopy(dict(resolved_config))
    checkpoint_config["training"]["embodiments_per_rank"] = embodiments_per_rank
    checkpoint_config["simulation"] = asdict(env.cfg.sim)
    fingerprint = getattr(state.training_data, "catalog_fingerprint", None)
    if fingerprint is not None:
        checkpoint_config["data"]["catalog_fingerprint"] = fingerprint
    if policy.morphology_encoder is not None:
        checkpoint_config["morphology_encoder"] = asdict(
            policy.morphology_encoder.config
        )

    packed = pack_policy_observations(
        observations,
        encoder_mode=policy.encoder_mode,
    )

    storage = RolloutStorage(
        training_type="rl",
        num_envs=env.num_envs,
        num_transitions_per_env=num_steps_per_env,
        obs=packed,
        actions_shape=(env.action_manager.total_action_dim,),
        device=str(env.device),
    )

    algorithm = MorphologyPPO(
        policy=policy,
        storage=storage,
        **dict(algorithm_options),
    )
    if resume_checkpoint is not None:
        algorithm.load(resume_checkpoint["algorithm"], load_optimizer=True)
        if distributed.is_primary():
            print(
                f"Resumed model, normalizers and optimizer at iteration {start_iteration}; "
                f"learning_rate={algorithm.learning_rate:.8g}; target={num_iterations}",
                flush=True,
            )
    algorithm.train_mode()

    writer = (
        metrics_path.open("x", encoding="utf-8")
        if distributed.is_primary()
        else nullcontext(None)
    )
    with writer as metrics_file:
        for iteration in range(start_iteration, num_iterations):
            iteration_start = time.perf_counter()
            # Resample morphology between PPO blocks; clips change on env reset.
            if (
                iteration > 0
                and iteration % updates_per_embodiment == 0
                and state.body_batch is None
            ):
                selection = state.training_data.sample()

                bundle_changed = selection.bundle_path != state.selection.bundle_path
                subject_changed = (
                    selection.subject.subject_id != state.selection.subject.subject_id
                )

                if bundle_changed or subject_changed:
                    prepared = state.prepared_bundle
                    if bundle_changed:
                        bundle = state.training_data.load_bundle(selection)
                        prepared = state.switcher.prepare_bundle(
                            bundle,
                            env.sim.mj_model,
                        )
                        del bundle

                    observations, _ = switch_training_embodiment(
                        env,
                        state.switcher,
                        prepared,
                        selection,
                        inject_model=bundle_changed,
                    )

                    state.selection = selection
                    state.prepared_bundle = prepared
                    state.observations = observations

                    # Do not retain a cache of every GPU bundle.
                    del prepared

            if (
                state.body_batch is not None
                and iteration > 0
                and iteration % updates_per_embodiment == 0
            ):
                observations = state.body_batch.install(state.body_batch.sample())

            # Accumulate on-device; read back once per update.
            reward_sum = torch.zeros((), device=env.device)
            terminated_count = torch.zeros((), device=env.device, dtype=torch.long)
            truncated_count = torch.zeros_like(terminated_count)

            rollout_start = time.perf_counter()

            # Rollout
            for _ in range(num_steps_per_env):
                actions = algorithm.act(observations)

                observations, rewards, terminated, truncated, _ = env.step(actions)

                reward_sum += rewards.detach().sum()
                terminated_count += terminated.sum()
                truncated_count += truncated.sum()

                algorithm.process_env_step(
                    rewards=rewards,
                    terminated=terminated,
                    truncated=truncated,
                )

            state.observations = observations

            # Bootstrap before any embodiment switch or statistics update.
            algorithm.compute_returns(observations)

            # clear() resets the storage cursor, not its tensor contents.
            # These views remain valid until the next rollout overwrites them.
            rollout_observations = storage.observations.flatten(0, 1)

            if torch.device(env.device).type == "cuda":
                torch.cuda.synchronize(env.device)
            rollout_seconds = time.perf_counter() - rollout_start
            update_start = time.perf_counter()

            # Update
            metrics = algorithm.update()

            # Use the collected rollout to update statistics for NEXT rollout.
            actor_observations, _ = unpack_policy_observations(
                rollout_observations,
                encoder_mode=policy.encoder_mode,
                group="actor",
            )
            critic_observations, _ = unpack_policy_observations(
                rollout_observations,
                encoder_mode=policy.encoder_mode,
                group="critic",
            )
            policy.update_observation_statistics(
                actor_observations,
                critic_observations,
            )

            mean_reward, num_terminated, num_truncated = (
                distributed.mean(
                    torch.stack(
                        (
                            reward_sum / (num_steps_per_env * env.num_envs),
                            terminated_count.to(reward_sum.dtype),
                            truncated_count.to(reward_sum.dtype),
                        )
                    )
                )
                .cpu()
                .tolist()
            )
            rollout_metrics = {
                "rollout/mean_step_reward": mean_reward,
                "rollout/terminated_count": round(
                    num_terminated * distributed.world_size()
                ),
                "rollout/truncated_count": round(
                    num_truncated * distributed.world_size()
                ),
            }

            durations = torch.tensor(
                [
                    rollout_seconds,
                    time.perf_counter() - update_start,
                    time.perf_counter() - iteration_start,
                ],
                device=env.device,
                dtype=torch.float64,
            )
            if distributed.world_size() > 1:
                torch.distributed.all_reduce(
                    durations, op=torch.distributed.ReduceOp.MAX
                )
            rollout_seconds, update_seconds, iteration_seconds = durations.tolist()
            global_batch = num_steps_per_env * env.num_envs * distributed.world_size()
            rollout_metrics.update(
                {
                    "perf/rollout_s": rollout_seconds,
                    "perf/update_s": update_seconds,
                    "perf/iteration_s": iteration_seconds,
                    "perf/global_steps_per_second": global_batch / iteration_seconds,
                }
            )
            command = env.command_manager.get_term("motion")
            motion_statistics = {
                "motion_paths": [str(path) for path in command.motion_paths],
                "active_motion_env_counts": torch.bincount(
                    command.motion_ids,
                    minlength=command.motion_library.num_motions,
                )
                .cpu()
                .tolist(),
                "sampled_motion_counts": command.consume_motion_sample_counts()
                .cpu()
                .tolist(),
            }
            selection = {
                "embodiments": state.body_batch.metadata()
                if state.body_batch is not None
                else [
                    {
                        "subject_id": state.selection.subject.subject_id,
                        "variant_id": state.selection.variant_id,
                        "num_envs": env.num_envs,
                    }
                ],
                "rank": distributed.rank(),
                "subject_id": state.selection.subject.subject_id,
                "variant_id": state.selection.variant_id,
                **motion_statistics,
            }
            selections = [selection]
            if distributed.world_size() > 1:
                selections = [None] * distributed.world_size()
                torch.distributed.all_gather_object(selections, selection)

            completed_iterations = iteration + 1
            record = {
                "iteration": completed_iterations,
                "environment_steps": (
                    completed_iterations
                    * num_steps_per_env
                    * env.num_envs
                    * distributed.world_size()
                ),
                "world_size": distributed.world_size(),
                "embodiments_per_rank": embodiments_per_rank,
                "peak_torch_allocated_gib": (
                    torch.cuda.max_memory_allocated(env.device) / 2**30
                    if torch.device(env.device).type == "cuda"
                    else 0.0
                ),
                "peak_torch_reserved_gib": (
                    torch.cuda.max_memory_reserved(env.device) / 2**30
                    if torch.device(env.device).type == "cuda"
                    else 0.0
                ),
                "num_envs_per_rank": env.num_envs,
                "rollout_selections": selections,
                "subject_id": state.selection.subject.subject_id,
                "variant_id": state.selection.variant_id,
                **motion_statistics,
                **metrics,
                **rollout_metrics,
            }

            if metrics_file is not None:
                metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
                metrics_file.flush()

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/iteration": completed_iterations,
                        "train/environment_steps": record["environment_steps"],
                        **{f"ppo/{name}": value for name, value in metrics.items()},
                        **rollout_metrics,
                    },
                    step=completed_iterations,
                )

            if distributed.is_primary():
                print(
                    f"[{completed_iterations}/{num_iterations}] "
                    f"{record['subject_id']}/{record['variant_id']} "
                    f"policy={metrics['surrogate']:.4f} "
                    f"value={metrics['value']:.4f} "
                    f"kl={metrics['kl']:.6f}",
                    flush=True,
                )
            if save_checkpoints and (
                completed_iterations % save_interval == 0
                or completed_iterations == num_iterations
            ):
                checkpoint_path = (
                    final_path
                    if completed_iterations == num_iterations
                    else output_dir / f"checkpoint_{completed_iterations:06d}.pt"
                )

                def save_checkpoint():
                    _save_training_checkpoint(
                        checkpoint_path,
                        algorithm,
                        completed_iterations,
                        checkpoint_config,
                    )

                distributed.primary_call(save_checkpoint)

    return final_path if save_checkpoints else metrics_path


def _create_wandb_run(config: dict, stage: str):
    """Own one stage's W&B run; missing configuration disables logging."""
    if not distributed.is_primary():
        return nullcontext(None)
    options = config.get("logging", {}).get("wandb", {})
    mode = options.get("mode", "disabled")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported W&B mode: {mode}")
    if mode == "disabled":
        return nullcontext(None)

    import wandb

    name = config["name"]
    seed = config["runtime"]["seed"]
    # Do not create the checkpoint directory before its existence check.
    log_dir = Path(config["output"]["logs_dir"]) / "wandb"
    log_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=options.get("project", "myoacross"),
        entity=options.get("entity"),
        mode=mode,
        name=f"{name}/{stage}/seed_{seed}",
        group=name,
        job_type=stage,
        tags=[config["encoder"]["mode"], f"seed_{seed}"],
        config=config,
        dir=str(log_dir),
    )


def discover_training_index_paths(data: dict) -> list[Path]:
    """Resolve and validate the requested training embodiment bank."""
    import json

    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")

    required = {"embodiment", "morphology_source", "bundle_root"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing training data settings: {sorted(missing)}")

    embodiment = data["embodiment"]
    morphology_source = data["morphology_source"]

    if embodiment not in (
        "baseline",
        "height_scale",
        "segment_scale",
    ):
        raise ValueError(f"Unknown embodiment: {embodiment!r}")

    if morphology_source not in ("subject", "virtual"):
        raise ValueError(f"Unknown morphology_source: {morphology_source!r}")

    if embodiment == "baseline" and morphology_source != "subject":
        raise ValueError("Baseline training requires morphology_source='subject'")

    subject_ids = select_target_ids("train", embodiment, data.get("subject_ids"))

    if not subject_ids:
        raise ValueError("The training subject allowlist is empty")

    root = (
        Path(data["bundle_root"])
        .expanduser()
        .resolve()
        / morphology_source
    )

    bank_path = root / "bank.json"
    if not bank_path.is_file():
        raise FileNotFoundError(
            f"Completed embodiment bank not found: {bank_path}. "
            "Run generation for the selected embodiment first."
        )

    bank = json.loads(bank_path.read_text(encoding="utf-8"))

    def validate_source(record: dict, path: Path) -> None:
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object: {path}")

        expected = {
            "embodiment": embodiment,
            "morphology_source": morphology_source,
        }

        for name, value in expected.items():
            if record.get(name) != value:
                raise ValueError(
                    f"{path}: expected {name}={value!r}, got {record.get(name)!r}"
                )

    validate_source(bank, bank_path)

    bank_subjects = bank.get("subject_ids")
    if (
        not isinstance(bank_subjects, list)
        or not all(
            isinstance(subject_id, str) and subject_id for subject_id in bank_subjects
        )
        or len(set(bank_subjects)) != len(bank_subjects)
    ):
        raise ValueError(f"Invalid bank subject IDs: {bank_path}")

    if embodiment == "baseline":
        if bank_subjects:
            raise ValueError("Shared baseline bank must have an empty subject_ids list")
        index_subject_ids = [None]
    else:
        missing_subjects = set(subject_ids) - set(bank_subjects)
        if missing_subjects:
            raise ValueError(
                "Bank is missing registered training subjects: "
                f"{sorted(missing_subjects)}"
            )
        index_subject_ids = subject_ids

    expected_count = bank.get("count_per_subject")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError(f"Invalid count_per_subject: {bank_path}")

    if morphology_source == "subject" and expected_count != 1:
        raise ValueError("Subject banks must contain exactly one model per subject")

    index_paths = []
    common_signature = None

    for subject_id in index_subject_ids:
        index_path = (
            root / "index.json"
            if subject_id is None
            else root / subject_id / "index.json"
        )
        if not index_path.is_file():
            raise FileNotFoundError(index_path)

        record = json.loads(index_path.read_text(encoding="utf-8"))
        validate_source(record, index_path)

        if (
            "source_subject_id" not in record
            or record["source_subject_id"] != subject_id
        ):
            raise ValueError(f"Index subject identity mismatch: {index_path}")

        for name in (
            "baseline_path",
            "simulation_config",
            "sampling",
        ):
            if name not in bank or name not in record:
                raise ValueError(f"Missing {name} in bank or index: {index_path}")
            if record[name] != bank[name]:
                raise ValueError(f"Index {name} differs from bank: {index_path}")

        signature = record.get("model_signature")
        if not isinstance(signature, str) or not signature.strip():
            raise ValueError(f"Invalid model signature: {index_path}")

        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError(f"Incompatible model signatures within bank: {index_path}")

        samples = record.get("samples")
        if (
            not isinstance(samples, list)
            or len(samples) != expected_count
            or record.get("count") != expected_count
        ):
            raise ValueError(f"Index sample count differs from bank: {index_path}")

        variant_ids = set()
        bundle_paths = set()

        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid sample record: {index_path}")

            variant_id = sample.get("variant_id")
            if (
                not isinstance(variant_id, str)
                or not variant_id.strip()
                or variant_id in variant_ids
            ):
                raise ValueError(f"Invalid or duplicate variant ID: {index_path}")
            variant_ids.add(variant_id)

            filename = sample.get("bundle")
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or Path(filename).suffix != ".npz"
            ):
                raise ValueError(f"Expected a bundle NPZ filename: {index_path}")

            bundle_path = index_path.parent / filename
            if bundle_path in bundle_paths:
                raise ValueError(f"Duplicate bundle path: {bundle_path}")
            if not bundle_path.is_file():
                raise FileNotFoundError(bundle_path)
            bundle_paths.add(bundle_path)

        index_paths.append(index_path)

    if morphology_source == "virtual":
        index_paths.extend(
            discover_training_index_paths({**data, "morphology_source": "subject"})
        )
    return index_paths


def run_training(config: dict, config_dir: Path, *, resume: Path | None = None) -> Path:
    """Prepare one simulation per process and execute synchronous PPO."""
    with distributed.training_process(config["runtime"]["device"]) as device:
        required = {
            "name",
            "runtime",
            "encoder",
            "data",
            "network",
            "environment",
            "training",
            "algorithm",
        }
        missing = required - set(config)
        if missing:
            raise ValueError(f"Missing training configuration: {sorted(missing)}")

        name = config["name"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in (".", "..")
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("name must be a non-empty directory name")

        runtime = config["runtime"]
        encoder = config["encoder"]
        mode = encoder["mode"]

        if mode not in {"none", "rl", "concat"}:
            raise ValueError(f"Unknown encoder mode: {mode}")

        config = deepcopy(config)
        config["data"].setdefault("motion_set", "kit_flat_locomotion_72")
        if config["data"].get("motion_ids") is None:
            config["data"]["motion_ids"] = [
                motion.motion_id
                for motion in get_motion_specs(motion_set=config["data"]["motion_set"])
            ]
        config["data"].setdefault("dataset_version", "embodiment_motion_v1")
        config["data"].setdefault("require_quality_pass", True)
        config["data"].setdefault("virtual_reference_mode", "source_subject")
        if config["data"]["dataset_version"] != "embodiment_motion_v1":
            raise ValueError("Unsupported dataset_version")
        if config["data"]["virtual_reference_mode"] != "source_subject":
            raise ValueError(
                "Only source_subject references are supported for virtual bundles"
            )
        resume_checkpoint = None
        start_iteration = 0
        if resume is not None:
            import torch
            from .resume import validate_resume_checkpoint

            resume = resume.expanduser().resolve()
            resume_checkpoint = torch.load(
                resume, map_location="cpu", weights_only=True, mmap=True
            )
            start_iteration = validate_resume_checkpoint(
                resume_checkpoint, config, distributed.world_size()
            )

        index_paths = discover_training_index_paths(config["data"])

        output = config.get("output")
        if not isinstance(output, dict) or set(output) != {"checkpoint_dir", "logs_dir"}:
            raise ValueError("Training requires output.checkpoint_dir and output.logs_dir")
        output_dir = resolve_path(output["checkpoint_dir"], PROJECT_ROOT)
        logs_dir = resolve_path(output["logs_dir"], PROJECT_ROOT)
        if (
            output_dir == logs_dir
            or output_dir in logs_dir.parents
            or logs_dir in output_dir.parents
        ):
            raise ValueError("Checkpoint and log directories must be separate")
        config["output"] = {"checkpoint_dir": str(output_dir), "logs_dir": str(logs_dir)}

        def prepare_output():
            for directory in (output_dir, logs_dir):
                if directory.exists() and any(directory.iterdir()):
                    raise FileExistsError(
                        f"New training run requires an empty directory: {directory}"
                    )
            output_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

        distributed.primary_call(prepare_output)

        checkpoint_value = encoder.get("checkpoint") if resume_checkpoint is None else None
        checkpoint = (
            resolve_path(checkpoint_value, config_dir)
            if checkpoint_value is not None
            else None
        )

        if checkpoint is not None:
            raise ValueError(
                f"{mode} training does not support external pretraining checkpoints"
            )

        morphology_encoder_config = None
        if mode == "rl" and checkpoint is None:
            morphology_encoder_config = MorphologyEncoderConfig(
                **config["morphology_encoder"]
            )

        resolved_config = deepcopy(config)
        if resume is not None:
            resolved_config["resume"] = {
                "checkpoint": str(resume),
                "iteration": start_iteration,
                "environment_state_restored": False,
            }
            if mode == "rl":
                resolved_config["morphology_encoder"] = asdict(
                    MorphologyEncoderConfig.from_checkpoint(resume_checkpoint["config"])
                )
        resolved_config["runtime"]["device"] = device
        resolved_config["distributed"] = {
            "world_size": distributed.world_size(),
            "num_envs_per_rank": config["environment"]["num_envs"],
            "sampling": "independent_per_rank",
        }
        resolved_config["data"]["train_index_paths"] = [str(path) for path in index_paths]
        resolved_config["encoder"]["checkpoint"] = (
            str(checkpoint) if checkpoint is not None else None
        )

        state = initialize_training(
            train_index_paths=index_paths,
            embodiment=config["data"]["embodiment"],
            morphology_source=config["data"]["morphology_source"],
            motion_set=config["data"]["motion_set"],
            subject_ids=config["data"].get("subject_ids"),
            motion_ids=config["data"].get("motion_ids"),
            retargets_root=config["data"].get("retargets_root"),
            require_quality_pass=config["data"]["require_quality_pass"],
            output_dir=output_dir,
            encoder_mode=mode,
            network_options=config["network"],
            environment_options=config["environment"],
            seed=runtime["seed"] + distributed.rank() + start_iteration,
            device=device,
            morphology_checkpoint=checkpoint,
            morphology_encoder_config=morphology_encoder_config,
            resume_checkpoint=resume_checkpoint,
        )

        try:
            preload_motion_clips(
                motion.retarget_path
                for entry in state.training_data.subjects
                for motion in entry.subject.motions
            )
            distributed.broadcast_model(state.policy)
            with _create_wandb_run(resolved_config, stage="rl") as wandb_run:
                if wandb_run is not None:
                    wandb_run.define_metric("train/environment_steps")
                    for pattern in ("ppo/*", "rollout/*", "perf/*"):
                        wandb_run.define_metric(
                            pattern, step_metric="train/environment_steps"
                        )
                return train_policy(
                    state,
                    **config["training"],
                    algorithm_options=config["algorithm"],
                    output_dir=output_dir,
                    logs_dir=logs_dir,
                    resolved_config=resolved_config,
                    wandb_run=wandb_run,
                    resume_checkpoint=resume_checkpoint,
                )
        finally:
            try:
                state.env.close()
            finally:
                clear_motion_clip_cache()
