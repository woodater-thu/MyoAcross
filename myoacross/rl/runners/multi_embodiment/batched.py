"""Headless, same-topology morphology batches in one MuJoCo-Warp simulation.

Mesh assets remain shared because this model uses them only for rendering.
Collision geometry, tendon sites, inertial parameters and touch volumes are
per-world. Requires the isolated sensor adapter from
python -m myoacross.rl.runners.multi_embodiment.backend; the installed environment is untouched.
"""
from __future__ import annotations

import numpy as np
import torch
import mujoco
import warp as wp

from myoacross.rl.embodiments.bundle import WarpBundleSwitcher
from myoacross.rl.tasks.motion_tracking.environment import refresh_joint_limit_buffers


@torch.no_grad()
def copy_body_groups(target: torch.Tensor, source: torch.Tensor) -> None:
    """Broadcast K rows into contiguous world groups without an expanded temporary."""
    if (target.ndim != source.ndim or target.shape[1:] != source.shape[1:]
            or source.shape[0] == 0 or target.shape[0] % source.shape[0]):
        raise ValueError("Body rows must evenly divide matching world rows")
    # view() must alias the simulator buffer; reshape() could silently copy it.
    target.view(source.shape[0], target.shape[0] // source.shape[0], *target.shape[1:]).copy_(
        source.unsqueeze(1)
    )


class BatchedEmbodiments:
    """Install K physical bodies, assigning equal contiguous groups of worlds."""

    SHARED = {"mesh_vert", "mesh_normal", "mesh_pos", "mesh_quat", "dof_length"}

    def __init__(self, state, count: int):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or state.env.num_envs % count:
            raise ValueError("Body count must be positive and divide num_envs")
        self.available_count = len({p for entry in state.training_data.subjects for _, p in entry.variants})
        if self.available_count < count:
            raise ValueError("Requested more distinct bodies than the training bank contains")
        from mujoco_warp._src import sensor
        if not getattr(sensor, "MYOACROSS_BATCHED_SITE_SIZE", False):
            raise RuntimeError("Run with the isolated batched-site-size backend adapter")
        self.state, self.count = state, count
        self.env = state.env
        self.group_size = self.env.num_envs // count
        model = self.env.sim.mj_model
        mesh = np.flatnonzero(model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH))
        if (np.any(model.geom_contype[mesh]) or np.any(model.geom_conaffinity[mesh])
                or np.isin(model.pair_geom1, mesh).any()
                or np.isin(model.pair_geom2, mesh).any()):
            raise ValueError("Shared meshes must not participate in collisions")
        if model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
            raise ValueError("Batched bodies currently require sleeping disabled")
        allowed = {int(mujoco.mjtSensor.mjSENS_TOUCH),
                   int(mujoco.mjtSensor.mjSENS_FRAMELINVEL),
                   int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL)}
        if not set(model.sensor_type.tolist()) <= allowed:
            raise ValueError("This headless adapter has not validated the model's sensor types")
        self.signature = state.prepared_bundle.bundle.model_signature
        self.layouts = {k: (v.shape, v.dtype) for k, v in state.prepared_bundle.bundle.arrays.items()}
        self.fields = tuple(n for n in WarpBundleSwitcher.FIELD_NAMES
                            if n not in self.SHARED and n not in {"site_size", "stat.meaninertia"})
        # Nested mean inertia and the adapter's site volumes need explicit expansion.
        sim = self.env.sim
        sim.wp_model.stat.meaninertia = wp.array(
            np.full(self.env.num_envs, float(sim.wp_model.stat.meaninertia.numpy()[0]), dtype=np.float32),
            device=self.env.device)
        old = sim.wp_model.site_size
        sim.wp_model.site_size = wp.array(
            np.repeat(old.numpy()[None], self.env.num_envs, axis=0),
            dtype=old.dtype, device=self.env.device)
        sim.expand_model_fields(self.fields)  # refresh bridges/sensors and recapture CUDA graphs
        self.group_ids = torch.arange(count, device=self.env.device).repeat_interleave(self.group_size)
        self.selections = ()

    def sample(self):
        """Retain the existing subject-uniform sampler, rejecting duplicate bodies."""
        selections, seen = [], set()
        while len(selections) < self.count:
            selected = self.state.training_data.sample()
            if selected.bundle_path not in seen:
                seen.add(selected.bundle_path)
                selections.append(selected)
        return selections

    @torch.no_grad()
    def install(self, selections):
        if len(selections) != self.count:
            raise ValueError("Wrong number of selections")
        if len({s.bundle_path for s in selections}) != self.count:
            raise ValueError("Body selections must be distinct")
        bundles = [self.state.training_data.load_bundle(s) for s in selections]
        for b in bundles:
            if b.model_signature != self.signature or {k: (v.shape, v.dtype) for k, v in b.arrays.items()} != self.layouts:
                raise ValueError("Incompatible body topology or bundle layout")
            for name, value in b.arrays.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite body parameter: {name}")
        # Deduplicate references for bodies derived from the same source subject.
        paths, ranges = [], {}
        for s in selections:
            key = s.subject.subject_id
            if key not in ranges:
                start = len(paths)
                paths.extend(m.retarget_path for m in s.subject.motions)
                ranges[key] = (start, len(paths) - start)
        command = self.env.command_manager.get_term("motion")
        starts, counts = zip(*(ranges[s.subject.subject_id] for s in selections))
        # Validate/load references before changing any live physics parameters.
        command.set_motions(paths, motion_ranges=tuple(
            torch.tensor(x, device=self.env.device)[self.group_ids] for x in (starts, counts)
        ))
        sim = self.env.sim
        wp.synchronize_device(self.env.device)
        for name in (*self.fields, "site_size", "stat.meaninertia"):
            target = sim.wp_model.stat.meaninertia if name == "stat.meaninertia" else getattr(sim.wp_model, name)
            values = np.stack([b.arrays[name] if name == "site_size" else b.arrays[name][0] for b in bundles])
            source = torch.as_tensor(values, device=self.env.device)
            copy_body_groups(wp.to_torch(target), source)
        command.morphology_features.copy_(torch.tensor(
            [b.features.values for b in bundles], device=self.env.device)[self.group_ids])
        command.subject_height_m.copy_(torch.tensor(
            [b.subject_height_m for b in bundles], device=self.env.device)[self.group_ids])
        refresh_joint_limit_buffers(self.env)
        command.reset_sampling_statistics()
        self.selections = tuple(selections)
        wp.synchronize_device(self.env.device)
        self.state.observations, _ = self.env.reset()
        self.state.selection = selections[0]
        # Verify physical fields and source-reference ownership before rollout.
        for name in ("body_mass", "body_pos", "site_size", "actuator_gainprm"):
            actual = wp.to_torch(getattr(sim.wp_model, name))[::self.group_size]
            expected = torch.tensor(np.stack([
                b.arrays[name] if name == "site_size" else b.arrays[name][0] for b in bundles]), device=self.env.device)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.validate_references()
        return self.state.observations

    def validate_references(self):
        command = self.env.command_manager.get_term("motion")
        starts, counts = command._motion_ranges
        if not bool(((command.motion_ids >= starts) & (command.motion_ids < starts + counts)).all()):
            raise AssertionError("A world sampled another body's source subject")

    def metadata(self):
        return [{"subject_id": s.subject.subject_id, "variant_id": s.variant_id,
                 "bundle_path": str(s.bundle_path), "num_envs": self.group_size}
                for s in self.selections]
