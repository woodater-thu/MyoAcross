"""MyoFullBody extensions to mjlab motion-tracking commands."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import (
    MotionCommand,
    MotionCommandCfg,
)
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
)

from myoacross.rl.embodiments.features import (
    FEATURE_DIM,
    MorphologyFeatures,
)
from myoacross.rl.tasks.motion_tracking.motion_library import MotionLibrary

@dataclass(kw_only=True)
class MyoFullBodyMotionCommandCfg(MotionCommandCfg):
    """Configuration for the MyoFullBody tracking command."""

    # Avoid the official default ±0.52 rad joint perturbation initially.
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    # Evaluation keeps the final frame without resampling the robot state.
    stop_at_motion_end: bool = False
    # Future reference times relative to the current frame, in seconds.
    lookahead_times_s: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
    # Subject-specific height for termination
    subject_height_m: float
    # Subject-specific features for morphology encoder
    subject_features: MorphologyFeatures
    motion_files: tuple[str, ...] = ()

    def build(
        self,
        env: ManagerBasedRlEnv,
    ) -> MyoFullBodyMotionCommand:
        return MyoFullBodyMotionCommand(self, env)

    
class MyoFullBodyMotionCommand(MotionCommand):
    """
    Reuse mjlab motion tracking with explicit muscle-state reset.

    Each environment samples a clip from the current source subject.
    All environments retain the same physical embodiment within a block.
    """

    def __init__(
        self,
        cfg: MyoFullBodyMotionCommandCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        robot = env.scene[cfg.entity_name]

        # Official MotionCommand uses the first selected body for root reset.
        if not cfg.body_names or cfg.body_names[0] != "Full Body":
            raise ValueError("body_names must begin with 'Full Body' for root reset")

        if cfg.anchor_body_name not in cfg.body_names:
            raise ValueError("anchor_body_name must be included in body_names")

        if not np.isfinite(env.step_dt) or env.step_dt <= 0:
            raise ValueError("Environment step_dt must be positive")

        # Reuse mjlab's entity indexing, buffers, metrics and adaptive kernel.
        # Its bootstrap single-clip loader is released after installing ours.
        if cfg.motion_files:
            cfg.motion_file = cfg.motion_files[0]
        super().__init__(cfg, env)
        self.motion_ids = torch.zeros_like(self.time_steps)
        self.set_motions(cfg.motion_files or (cfg.motion_file,))
        del self.motion
        del self.bin_count

        height = float(cfg.subject_height_m)
        if not np.isfinite(height) or height <= 0:
            raise ValueError("subject_height_m must be finite and positive")

        # Every environment uses the same subject within one rollout.
        self.subject_height_m = torch.full(
            (self.num_envs,),
            height,
            dtype=torch.float32,
            device=self.device,
        )

        # Physical values: 17 lengths in meters and mass in kilograms.
        # All parallel environments share one subject within a rollout.
        feature_values = torch.tensor(
            cfg.subject_features.values,
            dtype=torch.float32,
            device=self.device,
        )

        self.morphology_features = (
            feature_values.unsqueeze(0)
            .expand(self.num_envs, FEATURE_DIM)
            .clone()
        )

        # Actuator IDs and activation-state addresses are different indices.
        # Resolve their mapping once, while the model topology is fixed.
        model = env.sim.mj_model
        ctrl_ids = robot.data.indexing.ctrl_ids.detach().cpu().tolist()

        activation_ids = []
        for actuator_id in ctrl_ids:
            start = int(model.actuator_actadr[actuator_id])
            count = int(model.actuator_actnum[actuator_id])

            if start >= 0 and count > 0:
                activation_ids.extend(range(start, start + count))

        self._activation_ids = torch.tensor(
            activation_ids,
            dtype=torch.long,
            device=self.device,
        )

    def _update_command(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Advance references; episode termination owns end-of-clip resets."""
        if env_ids is None:
            self.time_steps.add_(1)
        else:
            self.time_steps[env_ids] += 1
        self.time_steps.copy_(torch.minimum(self.time_steps, self.motion_lengths - 1))

        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self._flat_bin_counts.lerp_(self._flat_current, self.cfg.adaptive_alpha)
            self._flat_current.zero_()

    def _write_reference_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Use official kinematic reset, then clear muscle state."""
        super()._write_reference_state_to_sim(
            env_ids,
            root_pos,
            root_ori,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
        )

        # Clear the selected environments' activation states.
        if self._activation_ids.numel() > 0:
            self._env.sim.data.act[
                env_ids[:, None],
                self._activation_ids[None, :],
            ] = 0.0

        # Clear the actual ctrl buffer as well as official actuator targets.
        ctrl = torch.zeros(
            (env_ids.numel(), self.robot.num_actuators),
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.write_ctrl_to_sim(ctrl, env_ids=env_ids)

    @property
    def lookahead(self) -> torch.Tensor:
        """
        Future reference states in the robot's current anchor frame.

        Per future frame:
            body positions:     3 * num_tracked_bodies
            anchor orientation: 6
            anchor velocities:  6
            time offset:        1, in seconds

        Returns shape (num_envs, num_future_frames, 3 * num_bodies + 13).
        """
        local_frames = (
            self.time_steps[:, None]
            + self._lookahead_frame_offsets[None, :]
        )
        local_frames = torch.minimum(local_frames, self.motion_lengths[:, None] - 1)
        frame_ids = self.motion_library.offsets[self.motion_ids, None] + local_frames

        # Shapes: (num_envs, num_future_frames, num_bodies, 3/4).
        body_pos_w = self.motion_library.body_pos_w[frame_ids]
        body_pos_w = (
            body_pos_w
            + self._env.scene.env_origins[:, None, None, :]
        )

        num_envs, num_future, num_bodies, _ = body_pos_w.shape

        # Express all future positions in the same current robot frame.
        anchor_pos_w = self.robot_anchor_pos_w
        anchor_quat_w = self.robot_anchor_quat_w

        body_frame_quat = anchor_quat_w[:, None, None, :].expand(
            -1, num_future, num_bodies, -1
        )

        body_pos_b = quat_apply_inverse(
            body_frame_quat.reshape(-1, 4),
            (
                body_pos_w - anchor_pos_w[:, None, None, :]
            ).reshape(-1, 3),
        ).reshape(num_envs, num_future, num_bodies * 3)

        anchor_index = self.motion_anchor_body_index

        future_quat_w = self.motion_library.body_quat_w[
            frame_ids, anchor_index
        ]
        future_lin_vel_w = self.motion_library.body_lin_vel_w[
            frame_ids, anchor_index
        ]
        future_ang_vel_w = self.motion_library.body_ang_vel_w[
            frame_ids, anchor_index
        ]

        current_quat = anchor_quat_w[:, None, :].expand(
            -1, num_future, -1
        ).reshape(-1, 4)

        relative_quat = quat_mul(
            quat_conjugate(current_quat),
            future_quat_w.reshape(-1, 4),
        )

        # Match mjlab's existing 6D orientation representation.
        orientation = matrix_from_quat(relative_quat)[..., :2]
        orientation = orientation.reshape(num_envs, num_future, 6)

        lin_vel_b = quat_apply_inverse(
            current_quat,
            future_lin_vel_w.reshape(-1, 3),
        ).reshape(num_envs, num_future, 3)

        ang_vel_b = quat_apply_inverse(
            current_quat,
            future_ang_vel_w.reshape(-1, 3),
        ).reshape(num_envs, num_future, 3)

        # Actual offset after clamping to the trajectory endpoint.
        time_offsets = (
            local_frames - self.time_steps[:, None]
        ).to(dtype=body_pos_b.dtype) / self.motion_fps

        return torch.cat(
            (
                body_pos_b,
                orientation,
                lin_vel_b,
                ang_vel_b,
                time_offsets.unsqueeze(-1),
            ),
            dim=-1,
        )

    def set_subject_morphology(
        self,
        features: MorphologyFeatures,
        height_m: float,
    ) -> None:
        """
        Update physical morphology data between rollouts.

        Call after injecting the matching bundle and before resetting.
        Existing device buffers retain their shapes and storage.
        """
        height = float(height_m)
        if not np.isfinite(height) or height <= 0:
            raise ValueError("Subject height must be finite and positive")

        if not isinstance(features, MorphologyFeatures):
            raise TypeError("features must be MorphologyFeatures")

        values = torch.tensor(
            features.values,
            dtype=self.morphology_features.dtype,
            device=self.device,
        )

        self.morphology_features.copy_(
            values.unsqueeze(0).expand_as(self.morphology_features)
        )
        self.subject_height_m.fill_(height)

    @torch.no_grad()
    def set_motion(self, motion_path: str | Path) -> None:
        """Single-clip evaluation uses the same library and reset path."""
        self.set_motions((motion_path,))

    @torch.no_grad()
    def set_motions(
        self, motion_paths: Sequence[str | Path], *,
        motion_ranges: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Install references between blocks; caller must reset before stepping."""
        paths = tuple(Path(path).expanduser().resolve() for path in motion_paths)
        ranges = None
        if motion_ranges is not None:
            if len(motion_ranges) != 2 or any(
                value.shape != (self.num_envs,) or value.dtype != torch.long
                for value in motion_ranges
            ):
                raise ValueError("Motion ranges require one int64 start and count per environment")
            starts, counts = (value.to(self.device).detach().clone() for value in motion_ranges)
            if not bool(((starts >= 0) & (counts > 0) & (starts < len(paths))
                         & (counts <= len(paths) - starts)).all()):
                raise ValueError("Motion ranges must stay inside the reference library")
            ranges = (starts, counts)
        if hasattr(self, "motion_library") and paths == self.motion_paths:
            # Library caching must not retain an outgoing body's sampling limits.
            self._motion_ranges = ranges
            self._evaluation_motion_ids = None
            return
        library = MotionLibrary(
            paths, self.body_indexes, num_joints=len(self.robot.joint_names),
            num_bodies=len(self.robot.body_names), step_dt=self._env.step_dt,
            device=self.device,
        )
        times = np.asarray(self.cfg.lookahead_times_s, dtype=np.float64)
        if (times.ndim != 1 or times.size == 0 or not np.isfinite(times).all()
                or np.any(times <= 0) or np.any(np.diff(times) <= 0)):
            raise ValueError("Lookahead times must be finite, positive, strictly increasing")
        offsets = np.rint(times * library.fps).astype(np.int64)
        if np.any(offsets < 1) or np.any(np.diff(offsets) <= 0):
            raise ValueError("Lookahead times must map to distinct positive frame offsets")
        lengths = library.lengths.cpu().tolist()
        widths = [int(length // library.fps) + 1 for length in lengths]
        flat_counts = torch.zeros(sum(widths), device=self.device)
        flat_current = torch.zeros_like(flat_counts)
        bin_widths = torch.tensor(widths, device=self.device, dtype=torch.long)
        bin_offsets = bin_widths.cumsum(0) - bin_widths
        counts = list(flat_counts.split(widths))
        current = list(flat_current.split(widths))
        frame_offsets = torch.as_tensor(offsets, dtype=torch.long, device=self.device)
        sample_counts = torch.zeros(library.num_motions, dtype=torch.long, device=self.device)
        valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        last_reset_step = torch.full_like(self.time_steps, self._env.common_step_counter)

        # Commit only after all data and auxiliary buffers are prepared.
        self.motion_library = library
        self.motion_fps = library.fps
        self.cfg.motion_file = str(paths[0])
        self.cfg.motion_files = tuple(map(str, paths))
        self._clip_lengths = lengths
        self._flat_bin_counts = flat_counts
        self._flat_current = flat_current
        self._bin_widths = bin_widths
        self._bin_offsets = bin_offsets
        self.bin_failed_count = counts
        self._current_bin_failed = current
        self._lookahead_frame_offsets = frame_offsets
        self._motion_sample_counts = sample_counts
        self._episode_valid = valid
        self._last_reset_step = last_reset_step
        self._motion_ranges = ranges
        self._evaluation_motion_ids = None
        self.motion_ids.zero_()
        self.time_steps.zero_()
        self._pending_forward = False
        for values in self.metrics.values():
            values.zero_()

    @property
    def motion_paths(self) -> tuple[Path, ...]:
        return self.motion_library.paths

    @property
    def motion_lengths(self) -> torch.Tensor:
        return self.motion_library.lengths[self.motion_ids]

    @property
    def reference_frame_indices(self) -> torch.Tensor:
        return self.motion_library.offsets[self.motion_ids] + self.time_steps

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion_library.joint_pos[self.reference_frame_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion_library.joint_vel[self.reference_frame_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (self.motion_library.body_pos_w[self.reference_frame_indices]
                + self._env.scene.env_origins[:, None, :])

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion_library.body_quat_w[self.reference_frame_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion_library.body_lin_vel_w[self.reference_frame_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion_library.body_ang_vel_w[self.reference_frame_indices]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    def reset_sampling_statistics(self) -> None:
        """Invalidate outgoing episodes before a full embodiment reset."""
        self._flat_bin_counts.zero_()
        self._flat_current.zero_()
        self._episode_valid.zero_()
        self._motion_sample_counts.zero_()

    def consume_motion_sample_counts(self) -> torch.Tensor:
        counts = self._motion_sample_counts.clone()
        self._motion_sample_counts.zero_()
        return counts

    def set_evaluation_motion_ids(self, env_ids: torch.Tensor, motion_ids: torch.Tensor) -> None:
        """Bind explicit clips for subsequent resets without changing other worlds."""
        if self.cfg.sampling_mode != "start":
            raise ValueError("Explicit evaluation requires start sampling")
        if (env_ids.ndim != 1 or env_ids.dtype != torch.long
                or motion_ids.shape != env_ids.shape or motion_ids.dtype != torch.long
                or env_ids.device != self.motion_ids.device or motion_ids.device != env_ids.device):
            raise ValueError("Evaluation bindings require matching device-local int64 vectors")
        if (env_ids.numel() != torch.unique(env_ids).numel()
                or (env_ids < 0).any() or (env_ids >= self.num_envs).any()
                or (motion_ids < 0).any() or (motion_ids >= self.motion_library.num_motions).any()):
            raise ValueError("Invalid environment or motion IDs")
        if self._motion_ranges is not None:
            starts, counts = self._motion_ranges
            if ((motion_ids < starts[env_ids]) | (motion_ids >= starts[env_ids] + counts[env_ids])).any():
                raise ValueError("Evaluation motion belongs to another subject")
        if self._evaluation_motion_ids is None:
            self._evaluation_motion_ids = self.motion_ids.clone()
        self._evaluation_motion_ids[env_ids] = motion_ids

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self.cfg.sampling_mode == "adaptive":
            failed = self._env.termination_manager.terminated[env_ids]
            failed = failed & self._episode_valid[env_ids]
            # mjlab retains terminated flags across manual resets until its
            # next termination compute. Consume an outgoing episode only once.
            failed = failed & (self._last_reset_step[env_ids] < self._env.common_step_counter)
            ids = env_ids[failed]
            clips = self.motion_ids[ids]
            widths = self._bin_widths[clips]
            bins = torch.minimum(
                self.time_steps[ids] * widths // self.motion_library.lengths[clips],
                widths - 1,
            )
            bins = bins + self._bin_offsets[clips]
            self._flat_current.add_(torch.bincount(bins, minlength=self._flat_current.numel()))
        if self._evaluation_motion_ids is not None:
            self.motion_ids[env_ids] = self._evaluation_motion_ids[env_ids]
        elif self._motion_ranges is None:
            self.motion_ids[env_ids] = torch.randint(
                self.motion_library.num_motions, (env_ids.numel(),), device=self.device,
            )
        else:
            starts, counts = self._motion_ranges
            self.motion_ids[env_ids] = starts[env_ids] + (
                torch.rand(env_ids.numel(), device=self.device) * counts[env_ids]
            ).long()
        self._motion_sample_counts.add_(torch.bincount(
            self.motion_ids[env_ids], minlength=self.motion_library.num_motions,
        ))
        # Parent uses our frame samplers and reference properties, then calls
        # our muscle reset. No physics model or bundle changes here.
        super()._resample_command(env_ids)
        self._episode_valid[env_ids] = True
        self._last_reset_step[env_ids] = self._env.common_step_counter

    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        self.time_steps[env_ids] = (
            torch.rand(env_ids.numel(), device=self.device)
            * (self.motion_lengths[env_ids] - 1)
        ).long()
        self.metrics["sampling_entropy"][env_ids] = 1.0
        self.metrics["sampling_top1_bin"][env_ids] = 0.5
        self.metrics["sampling_top1_prob"][env_ids] = (
            1.0 / (self.motion_lengths[env_ids] - 1)
        )

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        active_clips = torch.unique(self.motion_ids[env_ids], sorted=True).cpu().tolist()
        for clip_id in active_clips:
            length, counts = self._clip_lengths[clip_id], self.bin_failed_count[clip_id]
            ids = env_ids[self.motion_ids[env_ids] == clip_id]
            probabilities = counts + self.cfg.adaptive_uniform_ratio / counts.numel()
            padded = torch.nn.functional.pad(
                probabilities[None, None], (0, self.kernel.numel() - 1), mode="replicate",
            )
            probabilities = torch.nn.functional.conv1d(padded, self.kernel[None, None]).flatten()
            probabilities = probabilities / probabilities.sum()
            bins = torch.multinomial(probabilities, ids.numel(), replacement=True)
            self.time_steps[ids] = (
                (bins + torch.rand(ids.numel(), device=self.device))
                / counts.numel() * (length - 1)
            ).long().clamp(max=length - 2)
            entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
            self.metrics["sampling_entropy"][ids] = (
                entropy / math.log(counts.numel()) if counts.numel() > 1 else 1.0
            )
            pmax, imax = probabilities.max(0)
            self.metrics["sampling_top1_prob"][ids] = pmax
            self.metrics["sampling_top1_bin"][ids] = imax.float() / counts.numel()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        """Scrub each selected environment within its own clip, without noise."""
        self.time_steps[env_ids] = torch.minimum(
            torch.full_like(env_ids, max(int(frame), 0)), self.motion_lengths[env_ids] - 1,
        )
        self._write_reference_state_to_sim(
            env_ids, self.body_pos_w[env_ids, 0], self.body_quat_w[env_ids, 0],
            self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0],
            self.joint_pos[env_ids], self.joint_vel[env_ids],
        )
        self._episode_valid[env_ids] = False

    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None) -> None:
        """Reuse mjlab's scrubber protocol with selected-clip frame limits."""
        self._scrubber_env_idx = get_env_idx
        with server.gui.add_folder(name.capitalize()):
            scrubber = server.gui.add_slider(
                "Frame", min=0, max=int(self.motion_lengths[get_env_idx()].item()) - 1,
                step=1, initial_value=0,
            )

            @scrubber.on_update
            def on_frame(_):
                idx = get_env_idx()
                last = int(self.motion_lengths[idx].item()) - 1
                self.time_steps[idx] = min(max(int(scrubber.value), 0), last)
                if on_change is not None:
                    on_change()

            all_envs = server.gui.add_checkbox("All envs", initial_value=True)
            start = server.gui.add_button("Start Here")

            @start.on_click
            def on_start(_):
                if request_action is not None:
                    request_action("CUSTOM", {"type": "gui_reset", "all_envs": all_envs.value})

        self._scrubber_handles = (scrubber, all_envs, start)
        self._set_scrubber_disabled(True)

    def debug_vis(self, visualizer) -> None:
        # Runs in the viewer only, including when reference ghost drawing is off.
        if hasattr(self, "_scrubber_handles"):
            last = int(self.motion_lengths[self._scrubber_env_idx()].item()) - 1
            scrubber = self._scrubber_handles[0]
            if scrubber.max != last:
                scrubber.value = min(int(scrubber.value), last)
                scrubber.max = last
        super().debug_vis(visualizer)
