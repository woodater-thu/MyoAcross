"""Parallel policy evaluation with fixed body groups and per-world task queues."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import gc
from functools import wraps
import numpy as np
from types import SimpleNamespace
import json
import time
import mujoco
import torch
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    yaw_quat,
)
from myoacross.rl.embodiments.subjects import DatasetRecord
from myoacross.rl.embodiments.features import FEATURE_NAMES, FEATURE_UNITS
from myoacross.rl.embodiments.training_data import RolloutSelection
from myoacross.rl.embodiments.bundle import EmbodimentBundle, WarpBundleSwitcher
from myoacross.rl.runners.multi_embodiment.batched import BatchedEmbodiments
from myoacross.rl.networks.actor_critic import load_policy
from myoacross.rl.networks.rsl_adapter import unpack_policy_observations
from myoacross.rl.runners.generate_embodiment_bundles import (
    generate_embodiment_bundles,
)
from myoacross.rl.tasks.motion_tracking.environment import create_tracking_environment
from myoacross.rl.tasks.motion_tracking.myofullbody_env_cfg import (
    simulation_config_from_dict,
)

ERROR_KEYS = (
    "body_position_error_m",
    "aligned_body_position_error_m",
    "body_rotation_error_deg",
    "joint_angle_error_deg",
    "joint_velocity_error_deg_s",
)
METRICS = ("completed", "coverage") + ERROR_KEYS


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


REFERENCE_JOINT_NAMES = tuple(
    f"{name}_{side}"
    for side in ("l", "r")
    for name in (
        "hip_flexion", "hip_adduction", "hip_rotation", "knee_angle",
        "ankle_angle", "subtalar_angle",
    )
)


@dataclass(frozen=True)
class EvaluationCondition:
    target: str = "baseline"
    morphology_magnitude: float = 0.0
    reference_magnitude_deg: float = 0.0
    sample: int = 0


@dataclass(frozen=True)
class EvaluationTask:
    subject_index: int
    motion_index: int
    condition_index: int


class EvaluationQueue:
    """Each contiguous environment group owns one subject's finite task queue."""

    def __init__(self, motion_counts, envs_per_subject, conditions):
        if (
            isinstance(envs_per_subject, bool)
            or not isinstance(envs_per_subject, int)
            or envs_per_subject < 1
        ):
            raise ValueError("envs_per_subject must be a positive integer")
        self.group_size = envs_per_subject
        self.queues = [
            iter(
                [
                    EvaluationTask(s, m, ci)
                    for m in range(count)
                    for ci in range(len(conditions))
                ]
            )
            for s, count in enumerate(motion_counts)
        ]
        self.total = sum(motion_counts) * len(conditions)

    def assign(self, env_ids):
        return {i: next(self.queues[i // self.group_size], None) for i in env_ids}


class MetricAccumulator:
    def __init__(self, n, device):
        self.steps = torch.zeros(n, dtype=torch.long, device=device)
        self.sums = torch.zeros(n, 5, dtype=torch.float64, device=device)
        self.reward = torch.zeros(n, dtype=torch.float64, device=device)

    def add(self, active, errors, reward):
        self.steps += active
        self.sums += torch.where(active[:, None], errors, 0).double()
        self.reward += torch.where(active, reward, 0).double()

    def means(self):
        return self.sums / self.steps.clamp(min=1)[:, None]

    def reset(self, ids):
        self.steps[ids] = 0
        self.sums[ids] = 0
        self.reward[ids] = 0


def _capture_scoring_reference(command):
    """Clone before step: the command advances before step returns."""
    return {
        "pos": command.body_pos_w.clone(),
        "quat": command.body_quat_w.clone(),
        "joint": command.joint_pos.clone(),
        "velocity": command.joint_vel.clone(),
        "anchor_pos": command.anchor_pos_w.clone(),
        "anchor_quat": command.anchor_quat_w.clone(),
    }


def _compute_tracking_errors(command, robot, reference, body_ids, angular):
    """Score post-step states against saved pre-step references."""
    position = command.robot_body_pos_w
    orientation = command.robot_body_quat_w
    anchor = command.robot_anchor_pos_w.clone()
    anchor[:, 2] = reference["anchor_pos"][:, 2]
    rotation = yaw_quat(
        quat_mul(command.robot_anchor_quat_w, quat_inv(reference["anchor_quat"]))
    )
    aligned = anchor[:, None] + quat_apply(
        rotation[:, None].expand_as(reference["quat"]),
        reference["pos"] - reference["anchor_pos"][:, None],
    )
    return torch.stack(
        (
            torch.linalg.vector_norm(
                position[:, body_ids] - reference["pos"][:, body_ids], dim=-1
            ).mean(-1),
            torch.linalg.vector_norm(
                position[:, body_ids] - aligned[:, body_ids], dim=-1
            ).mean(-1),
            torch.rad2deg(
                quat_error_magnitude(
                    orientation[:, body_ids], reference["quat"][:, body_ids]
                )
            ).mean(-1),
            torch.rad2deg(
                (
                    robot.data.joint_pos[:, angular] - reference["joint"][:, angular]
                ).abs()
            ).mean(-1),
            torch.rad2deg(
                (
                    robot.data.joint_vel[:, angular] - reference["velocity"][:, angular]
                ).abs()
            ).mean(-1),
        ),
        dim=-1,
    )


def _initialize_batch(
    catalog, config, checkpoint, output, envs_per_subject, device, seed, bundle_root=None
):
    from mujoco_warp._src import sensor

    if not getattr(sensor, "MYOACROSS_BATCHED_SITE_SIZE", False):
        raise RuntimeError(
            "Batched evaluation needs the isolated backend. Run "
            "python -m myoacross rl evaluate --config <yaml>."
        )
    simulation = simulation_config_from_dict(config["simulation"])
    # Export selected physical models without randomization or fitting.
    first_subject = catalog.get(catalog.subject_ids[0])
    embodiment = json.loads(
        (first_subject.xml_path.parent / "report.json").read_text()
    )["strategy"]
    if bundle_root is None:
        bundle_root = output / "subject_bundles"
        paths = generate_embodiment_bundles(
            {
                None if embodiment == "baseline" else s.subject_id: s.xml_path.parent
                for s in catalog
            },
            bundle_root,
            embodiment=embodiment,
            morphology_source="subject",
            simulation_config=simulation,
        )
    else:
        bank = json.loads((bundle_root / "bank.json").read_text())
        # Runtime contact/constraint capacities override offline allocations.
        simulation_record = json.loads(json.dumps(config["simulation"]))
        physics = {key: value for key, value in simulation_record.items()
                   if key not in {"nconmax", "njmax"}}
        stored_physics = {key: value for key, value in bank["simulation_config"].items()
                          if key not in {"nconmax", "njmax"}}
        if (bank["embodiment"] != embodiment or bank["morphology_source"] != "subject"
                or stored_physics != physics):
            raise ValueError("Stored evaluation bundles differ from checkpoint simulation or embodiment")
        if embodiment == "baseline":
            paths = [bundle_root / "index.json"]
        else:
            if not set(catalog.subject_ids) <= set(bank["subject_ids"]):
                raise ValueError("Stored evaluation bank does not contain selected subjects")
            paths = [bundle_root / sid / "index.json" for sid in catalog.subject_ids]
    bundles, selections = {}, []
    for index in paths:
        record = json.loads(index.read_text())
        path = index.parent / record["samples"][0]["bundle"]
        bundles[path] = EmbodimentBundle.load(path)
        selections.append(
            RolloutSelection(
                catalog.get(
                    "shared"
                    if embodiment == "baseline"
                    else record["source_subject_id"]
                ),
                "subject",
                path,
            )
        )
    # Catalog ordering owns world groups; never depend on export ordering.
    by_subject = {s.subject.subject_id: s for s in selections}
    selections = [by_subject[s] for s in catalog.subject_ids]
    bank = json.loads((bundle_root / "bank.json").read_text())
    first = bundles[selections[0].bundle_path]
    options = deepcopy(config["environment"])
    options["num_envs"] = len(catalog) * envs_per_subject
    options["sampling_mode"] = "start"
    options["observation_options"] = dict(options.get("observation_options") or {})
    options["observation_options"]["enable_corruption"] = False
    env = create_tracking_environment(
        subject_xml=bank["baseline_path"],
        motion_path=selections[0].subject.motions[0].retarget_path,
        subject_features=first.features,
        subject_height_m=first.subject_height_m,
        simulation_config=simulation,
        seed=seed,
        device=device,
        **options,
    )
    try:
        command = env.command_manager.get_term("motion")
        command.cfg.stop_at_motion_end = True
        command.cfg.pose_range = {}
        command.cfg.velocity_range = {}
        command.cfg.joint_position_range = (0.0, 0.0)
        env.cfg.auto_reset = False
        switcher = WarpBundleSwitcher(env.sim.wp_model)
        prepared = switcher.prepare_bundle(first, env.sim.mj_model)
        # Adapter to the existing install API; no training initialization or sampler.
        bank_view = SimpleNamespace(
            subjects=[
                SimpleNamespace(variants=[("subject", s.bundle_path)])
                for s in selections
            ],
            load_bundle=lambda selection: bundles[selection.bundle_path],
        )
        state = SimpleNamespace(
            env=env, prepared_bundle=prepared, training_data=bank_view
        )
        batch = BatchedEmbodiments(state, len(catalog))
        observations = batch.install(selections)
        env.cfg.episode_length_s = (max(command._clip_lengths) + 2) * env.step_dt
        dynamic, _ = unpack_policy_observations(
            observations, encoder_mode=config["encoder"]["mode"]
        )
        policy = load_policy(
            checkpoint,
            observations=dynamic,
            action_dim=env.action_manager.total_action_dim,
            device=device,
        )
        policy.eval()
        return env, policy, batch
    except BaseException:
        env.close()
        raise


def _release_batch_resources(function):
    """Collect unreachable environment cycles after the batch frame exits."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        device = torch.device(kwargs.get("device", "cuda:0"))
        try:
            return function(*args, **kwargs)
        finally:
            if device.type == "cuda" and torch.cuda.is_initialized():
                torch.cuda.synchronize(device)
            gc.collect()
            if device.type == "cuda" and torch.cuda.is_initialized():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(device)
                print(f"[evaluation cleanup] device={device} used_mib={(total-free)/2**20:.1f} "
                      f"torch_allocated_mib={torch.cuda.memory_allocated(device)/2**20:.1f}", flush=True)
    return wrapped


@_release_batch_resources
@torch.no_grad()
def _run_evaluation_batch(
    catalog,
    checkpoint,
    conditions,
    output_dir,
    *,
    envs_per_subject,
    device,
    seed,
    max_steps,
    bundle_root=None,
    signal_collector=None,
):
    output = Path(output_dir)
    config = torch.load(checkpoint, map_location="cpu", weights_only=True)["config"]

    started = time.perf_counter()
    env = None
    completed = []
    write_json(
        output / "progress.json", dict(status="initializing", completed_episodes=0)
    )
    try:
        env, policy, batch = _initialize_batch(
            catalog,
            config,
            checkpoint,
            output,
            envs_per_subject,
            device,
            seed,
            bundle_root=bundle_root,
        )
        subjects = list(catalog)
        if signal_collector is not None:
            signal_collector.initialize(env, catalog)
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        joint_ids = robot.data.indexing.joint_ids.cpu().numpy()
        angular = torch.as_tensor(
            env.sim.mj_model.jnt_type[joint_ids] == mujoco.mjtJoint.mjJNT_HINGE,
            device=device,
        )
        tracking_names = env.termination_manager.get_term_cfg("body_position").params[
            "body_names"
        ]
        body_ids = [command.cfg.body_names.index(n) for n in tracking_names]
        metadata = []
        physical_features = (
            command.morphology_features[::envs_per_subject].cpu().tolist()
        )
        for i, subject in enumerate(subjects):
            metadata.append(
                dict(
                    subject_id=subject.subject_id,
                    physical_model=str(subject.xml_path),
                    true_measurements=physical_features[i],
                    feature_names=FEATURE_NAMES,
                    feature_units=FEATURE_UNITS,
                    references={
                        m.motion_id: dict(
                            path=str(m.retarget_path),
                        )
                        for m in subject.motions
                    },
                )
            )
        write_json(output / "subjects.json", metadata)
        write_json(output / "body_groups.json", batch.metadata())
        queue = EvaluationQueue(
            [len(s.motions) for s in subjects], envs_per_subject, conditions
        )
        current = {}
        active = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        input_features = command.morphology_features.clone()
        joint_offsets = torch.zeros_like(command.joint_pos)
        reference_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
        reference_levels = list(dict.fromkeys(
            (c.reference_magnitude_deg, c.sample) for c in conditions if c.reference_magnitude_deg))
        reference_positions = None
        signs_by_task = {}
        joint_names = list(robot.joint_names)
        if reference_levels:
            selected_joints = [joint_names.index(n) for n in REFERENCE_JOINT_NAMES]
            library = command.motion_library
            frame_offsets = library.offsets.cpu().tolist()
            frame_lengths = library.lengths.cpu().tolist()
            cpu_joints = library.joint_pos.cpu().numpy()
            cpu_positions = library.body_pos_w.cpu().numpy()
            cpu_quaternions = library.body_quat_w.cpu().numpy()
            positions = np.empty((len(reference_levels), len(cpu_joints),
                                  len(command.cfg.body_names), 3), dtype=np.float32)
            clip_index = 0
            for subject in subjects:
                model = mujoco.MjModel.from_xml_path(str(subject.xml_path))
                for motion in subject.motions:
                    start = frame_offsets[clip_index]
                    stop = start + frame_lengths[clip_index]
                    clip = dict(joint_pos=cpu_joints[start:stop],
                                body_pos_w=cpu_positions[start:stop],
                                body_quat_w=cpu_quaternions[start:stop])
                    for ri, (angle, sample) in enumerate(reference_levels):
                        signs = _perturbation_signs(seed, subject.subject_id, motion.motion_id, sample)
                        prepared = _prepare_reference_perturbation(
                            model, joint_names, list(command.cfg.body_names), clip,
                            angle, signs[len(FEATURE_NAMES):])
                        positions[ri, start:stop] = prepared["body_pos_w"]
                    clip_index += 1
            reference_positions = torch.as_tensor(positions, device=device)
            del positions, prepared, clip, model, cpu_joints, cpu_positions, cpu_quaternions
        setup_seconds = time.perf_counter() - started
        totals = torch.ones(env.num_envs, dtype=torch.long, device=device)
        metrics = MetricAccumulator(env.num_envs, device)
        offsets = [0]
        for subject in subjects:
            offsets.append(offsets[-1] + len(subject.motions))

        def refill(ids):
            """Reset only finished worlds; exhausted worlds remain excluded from scoring."""
            assignments = queue.assign(ids)
            bindings, flags = [], []
            for world, task in assignments.items():
                current.pop(world, None)
                flags.append(task is not None)
                input_features[world] = command.morphology_features[world]
                joint_offsets[world].zero_()
                reference_ids[world] = -1
                if task is not None:
                    current[world] = task
                    condition = conditions[task.condition_index]
                    subject = subjects[task.subject_index]
                    motion = subject.motions[task.motion_index]
                    signs = _perturbation_signs(seed, subject.subject_id, motion.motion_id, condition.sample)
                    signs_by_task[task.subject_index, task.motion_index, condition.sample] = signs
                    input_features[world] = torch.as_tensor(_prepare_morphology_perturbation(
                        physical_features[task.subject_index], condition.morphology_magnitude,
                        signs[:len(FEATURE_NAMES)]), device=device)
                    if condition.reference_magnitude_deg:
                        reference_ids[world] = reference_levels.index(
                            (condition.reference_magnitude_deg, condition.sample))
                        joint_offsets[world, selected_joints] = torch.as_tensor(
                            np.deg2rad(condition.reference_magnitude_deg) * signs[len(FEATURE_NAMES):],
                            dtype=joint_offsets.dtype, device=device)
                    bindings.append(offsets[task.subject_index] + task.motion_index)
                else:
                    # Safe reference for unused slots, with the same physical subject.
                    bindings.append(offsets[world // envs_per_subject])
            selected = torch.tensor(ids, device=device, dtype=torch.long)
            active[selected] = torch.tensor(flags, device=device)
            command.set_evaluation_motion_ids(
                selected, torch.tensor(bindings, device=device)
            )
            observations, _ = env.reset(env_ids=selected)
            if signal_collector is not None:
                signal_collector.on_reset(env, assignments)
            totals[selected] = command.motion_lengths[selected] - 1
            metrics.reset(selected)
            return observations

        observations = refill(list(range(env.num_envs)))
        run_started = time.perf_counter()
        batch_steps = 0
        last_report = 0
        while current:
            dynamic, morphology = unpack_policy_observations(
                observations, encoder_mode=policy.encoder_mode
            )
            reference = _capture_scoring_reference(command)
            if reference_positions is not None:
                ids = (active & (reference_ids >= 0)).nonzero().flatten()
                frames = command.time_steps[ids, None] + command._lookahead_frame_offsets[None, :]
                frames = torch.minimum(frames, command.motion_lengths[ids, None] - 1)
                frames += command.motion_library.offsets[command.motion_ids[ids, None]]
                dynamic = _apply_reference_perturbation(
                    dynamic, command, ids, command.joint_pos[ids] + joint_offsets[ids],
                    reference_positions[reference_ids[ids, None], frames]
                    + env.scene.env_origins[ids, None, None, :])
            action = policy.act_inference(
                dynamic, input_features if morphology is not None else None
            )
            if signal_collector is not None:
                signal_collector.before_step(env, current)
            observations, rewards, terminated, truncated, _ = env.step(action)
            if signal_collector is not None:
                signal_collector.after_step(env, current)
            errors = _compute_tracking_errors(command, robot, reference, body_ids, angular)
            # One device check, not dozens of CPU copies of full trajectories.
            finite = torch.isfinite(errors).all(-1) & torch.isfinite(rewards)
            if not bool((finite | ~active).all()):
                raise RuntimeError("Nonfinite active evaluation state or reward")
            metrics.add(active, errors, rewards)
            batch_steps += 1
            capped = (
                metrics.steps >= max_steps
                if max_steps is not None
                else torch.zeros_like(active)
            )
            done = terminated | truncated | (active & capped)
            ids = done.nonzero().flatten().cpu().tolist()
            if not ids:
                continue
            selected = torch.tensor(ids, device=device, dtype=torch.long)
            counts = metrics.steps[selected].cpu().tolist()
            means = metrics.means()[selected].cpu().tolist()
            sums = metrics.reward[selected].cpu().tolist()
            horizons = totals[selected].cpu().tolist()
            failed = terminated[selected].cpu().tolist()
            timeouts = truncated[selected].cpu().tolist()
            terms = {
                n: env.termination_manager.get_term(n)[selected].cpu().tolist()
                for n in env.termination_manager.active_terms
            }
            for j, world in enumerate(ids):
                task = current.get(world)
                if task is None:
                    continue
                reasons = [n for n, values in terms.items() if values[j]]
                reason = (
                    "failed"
                    if failed[j]
                    else "completed"
                    if timeouts[j] and "motion_finished" in reasons
                    else "time_limit"
                    if timeouts[j] and "time_out" in reasons
                    else "truncated"
                    if timeouts[j]
                    else "max_steps"
                )
                result = dict(
                    subject_index=task.subject_index,
                    motion_index=task.motion_index,
                    condition_index=task.condition_index,
                    steps=counts[j],
                    duration_s=counts[j] * env.step_dt,
                    reference_duration_s=horizons[j] * env.step_dt,
                    coverage=counts[j] / horizons[j],
                    completed=reason == "completed",
                    end_reason=reason,
                    termination_terms=";".join(reasons),
                    reward_sum=sums[j],
                    **dict(zip(ERROR_KEYS, means[j])),
                )
                completed.append(result)
                if signal_collector is not None:
                    signal_collector.on_episode_end(env, world, task, result)
            observations = refill(ids)
            if len(completed) - last_report >= env.num_envs or not current:
                last_report = len(completed)
                write_json(
                    output / "progress.json",
                    dict(
                        status="running",
                        completed_episodes=len(completed),
                        total_episodes=queue.total,
                        active_environments=len(current),
                        rollout_seconds=time.perf_counter() - run_started,
                    ),
                )
        rollout_seconds = time.perf_counter() - run_started
        if len(completed) != queue.total:
            raise RuntimeError("Incomplete evaluation task coverage")
        lookup = {
            (
                r["subject_index"],
                r["motion_index"],
                r["condition_index"],
            ): r
            for r in completed
        }
        if len(lookup) != queue.total:
            raise RuntimeError("Duplicate evaluation tasks")
        rows = []
        for si, subject in enumerate(subjects):
            for mi, motion in enumerate(subject.motions):
                baseline = lookup[si, mi, 0]
                for ci, condition in enumerate(conditions):
                    result = lookup[si, mi, ci]
                    metrics_record = {k: v for k, v in result.items()
                                      if k not in {"subject_index", "motion_index", "condition_index"}}
                    values = physical_features[si]
                    details = {}
                    if len(conditions) > 1:
                        signs = signs_by_task[si, mi, condition.sample]
                        values = _prepare_morphology_perturbation(
                            values, condition.morphology_magnitude, signs[:len(FEATURE_NAMES)]).tolist()
                        details = dict(perturbation_target=condition.target,
                                       morphology_magnitude=condition.morphology_magnitude,
                                       reference_magnitude_deg=condition.reference_magnitude_deg,
                                       sample=condition.sample,
                                       morphology_factors=json.dumps((1 + condition.morphology_magnitude
                                                                     * signs[:len(FEATURE_NAMES)]).tolist()),
                                       reference_joint_names=json.dumps(REFERENCE_JOINT_NAMES),
                                       reference_offsets_deg=json.dumps((condition.reference_magnitude_deg
                                                                         * signs[len(FEATURE_NAMES):]).tolist()))
                    rows.append(dict(
                        subject_id=subject.subject_id, motion_id=motion.motion_id,
                        category=motion.category, source_subject_id=motion.source_subject_id,
                        parameter=None if ci == 0 else condition.target,
                        relative_change=0.0, variant="baseline" if ci == 0 else
                        f"{condition.target}_m{condition.morphology_magnitude:g}_r{condition.reference_magnitude_deg:g}_sample{condition.sample}",
                        unit=None, true_value=None, input_value=None,
                        input_measurements=json.dumps(values),
                        physical_model=str(subject.xml_path), reference_path=str(motion.retarget_path),
                        **details, **metrics_record,
                        **{"delta_" + k: float(result[k]) - float(baseline[k]) for k in METRICS}))
        transitions = sum(r["steps"] for r in completed)
        timing = dict(
            setup_seconds=setup_seconds,
            rollout_seconds=rollout_seconds,
            total_seconds=time.perf_counter() - started,
            transitions=transitions,
            useful_transitions_per_second=transitions / rollout_seconds,
            batch_steps=batch_steps,
            num_envs=env.num_envs,
            simulated_transitions=batch_steps * env.num_envs,
        )
        write_json(output / "timing.json", timing)
        write_json(
            output / "progress.json",
            dict(
                status="finished",
                completed_episodes=len(completed),
                total_episodes=queue.total,
                completed_trials=len(rows),
            ),
        )
        return rows
    except Exception as error:
        write_json(
            output / "progress.json",
            dict(status="failed", error=repr(error), completed_episodes=len(completed)),
        )
        raise
    finally:
        if env is not None:
            env.close()


def _load_body_catalog(catalog, bundle_root):
    """Select stored physical models and retain original motion records."""
    subjects, geometry = [], {}
    for subject in catalog:
        record = json.loads((bundle_root / subject.subject_id / "index.json").read_text())
        if record["source_subject_id"] != subject.subject_id:
            raise ValueError("Stored test bundle belongs to a different subject")
        model_path = Path(record["source_directory"]) / "model.xml"
        values = dict(zip(FEATURE_NAMES, record["samples"][0]["features"]["values"]))
        thigh_r, thigh_l = values["thigh_length_r"], values["thigh_length_l"]
        shank_r, shank_l = values["shank_length_r"], values["shank_length_l"]
        leg_r, leg_l = thigh_r + shank_r, thigh_l + shank_l
        geometry[subject.subject_id] = dict(
            thigh_asymmetry_pct=200 * abs(thigh_r - thigh_l) / (thigh_r + thigh_l),
            shank_asymmetry_pct=200 * abs(shank_r - shank_l) / (shank_r + shank_l),
            leg_length_asymmetry_pct=200 * abs(leg_r - leg_l) / (leg_r + leg_l),
            leg_length_difference_mm=1000 * abs(leg_r - leg_l),
        )
        subjects.append(replace(subject, xml_path=model_path))
    return DatasetRecord(subjects), geometry


def _build_conditions(perturbation):
    """Schedule one clean baseline and paired multivariate bias samples."""
    conditions = [EvaluationCondition()]
    if perturbation is None:
        return conditions
    allowed = {"target", "morphology_magnitudes", "reference_magnitudes_deg", "num_samples"}
    if not isinstance(perturbation, dict) or set(perturbation) - allowed:
        raise ValueError("Invalid perturbation settings")
    target = perturbation.get("target")
    if target not in {"morphology", "reference", "combined"}:
        raise ValueError("perturbation.target must be morphology, reference, or combined")
    samples = perturbation.get("num_samples", 3)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("num_samples must be a positive integer")
    morphology = perturbation.get("morphology_magnitudes", [0.02, 0.05, 0.10])
    reference = perturbation.get("reference_magnitudes_deg", [2.5, 5.0, 10.0])
    for values, label in ((morphology, "morphology_magnitudes"),
                          (reference, "reference_magnitudes_deg")):
        if (not isinstance(values, list) or not values
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not np.isfinite(v) or v <= 0 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{label} must contain unique positive finite magnitudes")
    if any(v >= 1 for v in morphology):
        raise ValueError("morphology_magnitudes must be below one")
    if target == "combined" and len(morphology) != len(reference):
        raise ValueError("Combined magnitudes must have matching lengths")
    if target == "combined":
        levels = list(zip(morphology, reference))
    elif target == "morphology":
        levels = [(m, 0.0) for m in morphology]
    else:
        levels = [(0.0, r) for r in reference]
    conditions.extend(
        EvaluationCondition(target, float(m), float(r), sample)
        for m, r in levels
        for sample in range(samples)
    )
    return conditions


def _perturbation_signs(seed, subject_id, motion_id, sample):
    """Use task identity, independent of policy, world count, and scheduling."""
    subject, motion = subject_id.encode(), motion_id.encode()
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, sample, len(subject), *subject, len(motion), *motion]))
    return rng.choice([-1, 1], len(FEATURE_NAMES) + len(REFERENCE_JOINT_NAMES))


def _prepare_morphology_perturbation(features, magnitude, signs):
    """Prepare episode-fixed physical measurements before policy normalization."""
    values = np.asarray(features, dtype=np.float32)
    if magnitude == 0:
        return values
    return (values * (1.0 + magnitude * np.asarray(signs))).astype(np.float32)


def _prepare_reference_perturbation(
    model, joint_names, body_names, clip, magnitude_deg, signs
):
    """Compute biased lower-limb reference poses without changing the clean clip."""
    if magnitude_deg == 0:
        return clip
    joint_ids = np.array([model.joint(n).id for n in joint_names])
    body_ids = np.array([model.body(n).id for n in body_names])
    selected = np.array([joint_names.index(n) for n in REFERENCE_JOINT_NAMES])
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or model.jnt_bodyid[free[0]] != body_ids[0]:
        raise ValueError("Reference perturbation requires the free-root body first")
    root = model.jnt_qposadr[free[0]]
    joints = clip["joint_pos"].copy()
    joints[:, selected] += np.deg2rad(magnitude_deg) * np.asarray(signs)
    positions = np.empty_like(clip["body_pos_w"])
    data = mujoco.MjData(model)
    for frame, values in enumerate(joints):
        data.qpos[model.jnt_qposadr[joint_ids]] = values
        data.qpos[root:root + 3] = clip["body_pos_w"][frame, 0]
        data.qpos[root + 3:root + 7] = clip["body_quat_w"][frame, 0]
        mujoco.mj_kinematics(model, data)
        positions[frame] = data.xpos[body_ids]
    return {**clip, "joint_pos": joints, "body_pos_w": positions}


def _apply_reference_perturbation(
    dynamic, command, ids, joint_targets, future_positions
):
    """Replace policy targets only; keep clean observations and command buffers."""
    result = dict(dynamic)
    joint = dynamic["joint"].clone()
    count = joint_targets.shape[-1]
    joint[ids, 2 * count:3 * count] = joint_targets
    joint[ids, 4 * count:5 * count] = joint_targets - joint[ids, :count]
    result["joint"] = joint
    if "lookahead" in dynamic:
        lookahead = dynamic["lookahead"].clone()
        rotation = command.robot_anchor_quat_w[ids, None, None, :].expand(
            *future_positions.shape[:-1], 4)
        position = quat_apply_inverse(
            rotation, future_positions - command.robot_anchor_pos_w[ids, None, None, :]
        )
        lookahead[ids, :, :position.shape[-2] * 3] = position.flatten(-2)
        result["lookahead"] = lookahead
    return result


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_trials(rows):
    """Average baseline motions per subject and weight subjects equally."""
    baseline = [r for r in rows if r["parameter"] is None]
    subjects = []
    for subject in sorted({r["subject_id"] for r in baseline}):
        selected = [r for r in baseline if r["subject_id"] == subject]
        subjects.append(
            dict(
                subject_id=subject,
                num_trials=len(selected),
                **{k: float(np.mean([r[k] for r in selected])) for k in METRICS},
            )
        )
    categories = []
    for category in sorted({r["category"] for r in baseline}):
        selected = [r for r in baseline if r["category"] == category]
        categories.append(
            dict(
                category=category,
                num_trials=len(selected),
                **{k: float(np.mean([r[k] for r in selected])) for k in METRICS},
            )
        )
    summary = dict(
        num_subjects=len(subjects),
        num_trials=len(baseline),
        **{k: float(np.mean([r[k] for r in subjects])) for k in METRICS},
    )
    return subjects, categories, summary


def _summarize_sensitivity(rows):
    """Average samples and motions per subject, then weight subjects equally."""
    fields = ("perturbation_target", "morphology_magnitude", "reference_magnitude_deg")
    keys = METRICS + tuple("delta_" + k for k in METRICS)
    results = []
    for condition in sorted({tuple(r[k] for k in fields) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in fields) == condition]
        subjects = sorted({r["subject_id"] for r in selected})
        subject_means = []
        for subject in subjects:
            trials = [r for r in selected if r["subject_id"] == subject]
            subject_means.append({key: np.mean([r[key] for r in trials]) for key in keys})
        values = {key: float(np.mean([r[key] for r in subject_means])) for key in keys}
        results.append(dict(
            zip(fields, condition), num_subjects=len(subjects),
            num_trials=len(selected), **values,
            completion_drop=-values["delta_completed"],
        ))
    return results


def evaluate_policy(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    target_embodiment: str | None = None,
    subject_ids: list[str] | None = None,
    motion_ids: list[str] | None = None,
    perturbation: dict | None = None,
    bundle_root: str | Path | None = None,
    envs_per_subject: int = 64,
    device: str = "cuda:0",
    seed: int = 42,
    max_steps: int | None = None,
    retargets_root: str | Path | None = None,
    catalog: DatasetRecord | None = None,
    signal_collector=None,
    allow_external_motions: bool = False,
    resume: bool = False,
) -> Path:
    """Run parallel trials with optional input biases or physical body conditions."""
    external_catalog = catalog is not None
    if allow_external_motions and not external_catalog:
        raise ValueError("External motions require an explicit catalog")
    if target_embodiment is not None and target_embodiment not in (
        "baseline",
        "height_scale",
        "segment_scale",
    ):
        raise ValueError(f"Unknown target_embodiment: {target_embodiment!r}")
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    if (
        isinstance(envs_per_subject, bool)
        or not isinstance(envs_per_subject, int)
        or envs_per_subject < 1
    ):
        raise ValueError("envs_per_subject must be a positive integer")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer or null")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    conditions = _build_conditions(perturbation)
    del saved
    if config["data"].get("dataset_version") != "embodiment_motion_v1":
        raise ValueError("Checkpoint predates the embodiment × motion dataset")
    allowed_motions = config["data"].get("motion_ids")
    if motion_ids is not None:
        if (
            not isinstance(motion_ids, list)
            or not motion_ids
            or any(not isinstance(m, str) for m in motion_ids)
            or len(set(motion_ids)) != len(motion_ids)
        ):
            raise ValueError("motion_ids must be a nonempty list of unique IDs")
        if not allow_external_motions and allowed_motions is not None and not set(motion_ids) <= set(allowed_motions):
            raise ValueError(
                "Selected motions are outside the checkpoint motion selection"
            )
    embodiment = (
        target_embodiment
        if target_embodiment is not None
        else config["data"]["embodiment"]
    )
    body_conditions = None
    if bundle_root is not None:
        from myoacross.utils.paths import PROJECT_ROOT, resolve_path

        bundle_root = resolve_path(str(bundle_root), PROJECT_ROOT)
        manifest_path = bundle_root / "conditions.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if (manifest["embodiment"] != embodiment or manifest["split"] != split
                    or embodiment != "segment_scale" or split != "test"):
                raise ValueError("Stored physical conditions require matching segment_scale test subjects")
            if perturbation is not None:
                raise ValueError("Evaluate physical asymmetry separately from policy-input perturbations")
            body_conditions = manifest["conditions"]
            if not body_conditions or body_conditions[0]["name"] != "original":
                raise ValueError("Stored body conditions must start with the original model")

    selected_root = retargets_root or config["data"].get("retargets_root")
    if catalog is None:
        catalog = DatasetRecord.from_split(
            split=split,
            embodiment=embodiment,
            subject_ids=subject_ids,
            motion_set=config["data"].get("motion_set", "kit_flat_locomotion_72"),
            motion_ids=motion_ids if motion_ids is not None else allowed_motions,
            require_quality_pass=config["data"].get("require_quality_pass", True),
            **({"retargets_root": selected_root} if selected_root else {}),
        )
    else:
        if catalog.split != split:
            raise ValueError("Supplied catalog split differs from evaluation split")
        if subject_ids is not None:
            catalog = catalog.select(subject_ids)
        for subject in catalog:
            ids = {motion.motion_id for motion in subject.motions}
            if not allow_external_motions and allowed_motions is not None and not ids <= set(allowed_motions):
                raise ValueError("Catalog motions are outside the checkpoint motion selection")
            if motion_ids is not None and ids != set(motion_ids):
                raise ValueError("Catalog motions differ from explicit motion selection")
    if signal_collector is not None and (perturbation is not None or body_conditions is not None):
        raise ValueError("Signal collection requires ordinary catalog evaluation")
    actual_motion_ids = (motion_ids if motion_ids is not None else allowed_motions)
    if external_catalog:
        actual_motion_ids = list(dict.fromkeys(m.motion_id for s in catalog for m in s.motions))
    output = Path(output_dir).expanduser().resolve()
    if resume and body_conditions is None:
        raise ValueError("Resume requires fixed physical body conditions")
    output.mkdir(parents=True, exist_ok=resume)
    evaluation_config = dict(
            checkpoint=str(checkpoint),
            split=split,
            training_embodiment=config["data"]["embodiment"],
            target_embodiment=embodiment,
            subjects=list(catalog.subject_ids),
            motion_ids=actual_motion_ids,
            training_motion_ids=allowed_motions,
            allow_external_motions=allow_external_motions,
            perturbation=perturbation,
            bundle_root=str(bundle_root) if bundle_root is not None else None,
            body_conditions=body_conditions,
            reference_mode="source_subject",
            conditions=([dict(parameter=None, relative_change=0.0)] if perturbation is None
                        else [asdict(c) for c in conditions]),
            envs_per_subject=envs_per_subject,
            device=device,
            seed=seed,
            max_steps=max_steps,
            checkpoint_config=config,
            reference_convention="pre-step scoring reference paired with post-step state",
            input_space="physical_before_log_standardization",
            fixed_within_episode=True,
        )
    if resume:
        previous = json.loads((output / "config.json").read_text())
        if previous != json.loads(json.dumps(evaluation_config)):
            raise ValueError("Resume configuration differs from existing evaluation")
    else:
        write_json(output / "config.json", evaluation_config)
    if body_conditions is not None:
        all_rows, summaries, originals = [], [], {}
        for settings in body_conditions:
            condition = settings["name"]
            side = settings["shortened_side"]
            shortening = settings["relative_shortening"]
            condition_root = bundle_root / condition
            selected_catalog, geometry = _load_body_catalog(catalog, condition_root)
            destination = output / condition
            destination.mkdir(exist_ok=resume)
            progress_file = destination / "progress.json"
            finished = (resume and progress_file.exists()
                        and json.loads(progress_file.read_text()).get("status") == "finished"
                        and (destination / "summary.json").exists())
            if finished:
                with (destination / "trials.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                expected = {(subject.subject_id, motion.motion_id)
                            for subject in selected_catalog for motion in subject.motions}
                if (len(rows) != len(expected) or
                        {(r["subject_id"], r["motion_id"]) for r in rows} != expected):
                    raise ValueError(f"Incomplete saved trials for {condition}")
                for row in rows:
                    row["parameter"] = None if row["parameter"] == "" else row["parameter"]
                    for key in METRICS:
                        row[key] = (float(row[key] == "True") if row[key] in ("True", "False")
                                    else float(row[key]))
                print(f"[evaluation resume] verified {condition}: {len(rows)} trials", flush=True)
            else:
                rows = _run_evaluation_batch(
                    selected_catalog, checkpoint, conditions, destination,
                    envs_per_subject=envs_per_subject, device=device, seed=seed, max_steps=max_steps,
                    bundle_root=condition_root,
                )
            for row in rows:
                identity = row["subject_id"], row["motion_id"]
                if side is None:
                    originals[identity] = row.copy()
                row.update(body_condition=condition, shortened_side=side,
                           relative_shortening=shortening, **geometry[row["subject_id"]])
                row.update({"body_delta_" + key: float(row[key]) - float(originals[identity][key])
                            for key in METRICS})
            subjects, categories, summary = _summarize_trials(rows)
            if not finished:
                _write_csv(destination / "trials.csv", rows)
                _write_csv(destination / "subjects.csv", subjects)
                _write_csv(destination / "categories.csv", categories)
                write_json(destination / "summary.json", summary)
            summaries.append(dict(body_condition=condition, shortened_side=side,
                                  relative_shortening=shortening, **summary,
                                  **{"body_delta_" + key: summary[key] - summaries[0][key]
                                     if summaries else 0.0 for key in METRICS},
                                  **{key: float(np.mean([record[key] for record in geometry.values()]))
                                     for key in next(iter(geometry.values()))}))
            all_rows.extend(rows)
        _write_csv(output / "trials.csv", all_rows)
        _write_csv(output / "body_conditions.csv", summaries)
        original_rows = [row for row in all_rows if row["body_condition"] == "original"]
        subjects, categories, summary = _summarize_trials(original_rows)
        _write_csv(output / "subjects.csv", subjects)
        _write_csv(output / "categories.csv", categories)
        write_json(output / "summary.json", summary)
        return output
    rows = _run_evaluation_batch(
        catalog,
        checkpoint,
        conditions,
        output,
        envs_per_subject=envs_per_subject,
        device=device,
        seed=seed,
        max_steps=max_steps,
        bundle_root=bundle_root,
        **({"signal_collector": signal_collector} if signal_collector is not None else {}),
    )
    _write_csv(output / "trials.csv", rows)
    subjects, categories, summary = _summarize_trials(rows)
    _write_csv(output / "subjects.csv", subjects)
    _write_csv(output / "categories.csv", categories)
    write_json(output / "summary.json", summary)
    if signal_collector is not None:
        signal_collector.finalize()
    if perturbation is not None:
        sensitivity = _summarize_sensitivity(rows)
        _write_csv(output / "sensitivity.csv", sensitivity)
    return output
