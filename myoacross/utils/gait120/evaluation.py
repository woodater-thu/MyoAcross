"""Capture physiological signals through the existing parallel evaluator hooks."""

from pathlib import Path
import json
import numpy as np
import warp as wp
from .signals import JOINT_NAMES, MUSCLE_NAMES, phase_profile
from mm_evalkit.workflows.gait.cycles import extract_valid_strides_flexible


class SolverIntervalGRF:
    """Read solver forces after each native physics step, before forward()."""

    def __init__(self, env):
        import warp as wp
        import torch
        from .grf import accumulate_foot_forces

        self.wp = wp
        self.torch = torch
        self.kernel = accumulate_foot_forces
        model = env.sim.mj_model
        side = np.full(model.ngeom, -1, dtype=np.int32)
        ground = np.zeros(model.ngeom, dtype=np.int32)
        import mujoco

        for i in range(model.ngeom):
            body = model.body(model.geom_bodyid[i]).name.rsplit("/", 1)[-1]
            if body in ("calcn_l", "toes_l"):
                side[i] = 0
            elif body in ("calcn_r", "toes_r"):
                side[i] = 1
            ground[i] = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        if not np.any(side == 0) or not np.any(side == 1) or (not ground.any()):
            raise ValueError("GRF collection requires both feet and a plane ground")
        device = env.sim.wp_data.contact.geom.device
        self.side = wp.array(side, dtype=wp.int32, device=device)
        self.ground = wp.array(ground, dtype=wp.int32, device=device)
        capacity = env.sim.wp_data.contact.geom.shape[0]
        self.contact_ids = wp.array(
            np.arange(capacity, dtype=np.int32), dtype=wp.int32, device=device
        )
        self.forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=device)
        self.net = wp.zeros((env.num_envs, 2), dtype=wp.vec3, device=device)
        self.gravity = torch.as_tensor(
            model.opt.gravity.copy(), dtype=torch.float32, device=env.device
        )
        self.end_time = None
        self.mass = None

    def capture(self, env):
        import mujoco_warp

        wp = self.wp
        data = env.sim.wp_data
        self.net.zero_()
        mujoco_warp.contact_force(
            env.sim.wp_model, data, self.contact_ids, True, self.forces
        )
        wp.launch(
            self.kernel,
            dim=self.contact_ids.size,
            inputs=[
                data.nacon,
                data.contact.geom,
                data.contact.worldid,
                self.forces,
                self.side,
                self.ground,
            ],
            outputs=[self.net],
        )
        self.end_time = wp.to_torch(data.time).clone()
        self.mass = wp.to_torch(env.sim.wp_model.body_mass).sum(-1).clone()
        vertical = wp.to_torch(self.net) @ (-self.gravity / self.gravity.norm())
        self.torque = env.scene["robot"].data.qfrc_actuator.clone()
        return vertical.sum(-1) / (self.mass * self.gravity.norm())

    def values(self):
        vertical = self.wp.to_torch(self.net) @ (-self.gravity / self.gravity.norm())
        return vertical / (self.mass[:, None] * self.gravity.norm())


def capture_grf_substep(env):
    return env._gait120_grf.capture(env)


class Gait120SignalCollector:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.buffers = {}
        self.audit = []

    def initialize(self, env, catalog):
        import torch
        from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg

        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.subjects = list(catalog)
        robot = env.scene["robot"]
        self.joint_ids = [robot.joint_names.index(n) for n in JOINT_NAMES]
        model = env.sim.mj_model
        muscle_ids = [
            next(
                (
                    i
                    for i in range(model.nu)
                    if model.actuator(i).name.rsplit("/", 1)[-1] == n
                )
            )
            for n in MUSCLE_NAMES
        ]
        self.act_ids = torch.as_tensor(
            model.actuator_actadr[muscle_ids].copy(), device=env.device
        )
        if bool((self.act_ids < 0).any()):
            raise ValueError("Selected muscles have no activation state")
        self.touch_ids = [
            model.sensor(
                next(
                    (
                        model.sensor(i).name
                        for i in range(model.nsensor)
                        if model.sensor(i).name.rsplit("/", 1)[-1] == n
                    )
                )
            ).adr[0]
            for n in ("l_foot", "l_toes", "r_foot", "r_toes")
        ]
        env._gait120_grf = SolverIntervalGRF(env)
        metrics = dict(env.cfg.metrics)
        metrics["gait120_vertical_grf_bw"] = MetricsTermCfg(
            func=capture_grf_substep, per_substep=True
        )
        env.metrics_manager = MetricsManager(metrics, env)
        self.physics_dt = float(env.physics_dt)
        self.audit = []

    def on_reset(self, env, assignments):
        for world, task in assignments.items():
            if task is not None:
                self.buffers[world] = dict(
                    joints=[], muscle=[], contacts=[], grf=[], reference_steps=[]
                )

    def before_step(self, env, current):
        import warp as wp

        joints = (
            env.scene["robot"].data.joint_pos[:, self.joint_ids].detach().cpu().numpy()
        )
        act = wp.to_torch(env.sim.wp_data.act)[:, self.act_ids].detach().cpu().numpy()
        touch = (
            wp.to_torch(env.sim.wp_data.sensordata)[:, self.touch_ids]
            .detach()
            .cpu()
            .numpy()
        )
        frames = (
            env.command_manager.get_term("motion").time_steps.detach().cpu().numpy()
        )
        for world in current:
            buf = self.buffers[world]
            buf["reference_steps"].append(int(frames[world]))
            buf["joints"].append(np.rad2deg(joints[world].copy()))
            buf["muscle"].append(act[world].copy())
            (left, right) = (
                np.any(touch[world, :2] > 0.05),
                np.any(touch[world, 2:] > 0.05),
            )
            buf["contacts"].append(
                2 if left and right else 0 if left else 1 if right else -1
            )
        robot = env.scene["robot"]
        t = wp.to_torch(env.sim.wp_data.time).detach().cpu().numpy()
        tau = robot.data.qfrc_actuator[:, self.joint_ids].detach().cpu().numpy()
        root = wp.to_torch(env.sim.wp_data.qpos)[:, :3].detach().cpu().numpy()
        for world in current:
            b = self.buffers[world]
            for key, value in [
                ("time", float(t[world])),
                ("torque", tau[world].copy()),
                ("root", root[world].copy()),
            ]:
                b.setdefault(key, []).append(value)

    def after_step(self, env, current):
        grf = env._gait120_grf.values().detach().cpu().numpy()
        for world in current:
            self.buffers[world]["grf"].append(grf[world].copy())
        grf = env._gait120_grf
        times = grf.end_time.detach().cpu().numpy()
        force = wp.to_torch(grf.net).detach().cpu().numpy()
        torque = grf.torque[:, self.joint_ids].detach().cpu().numpy()
        for world in current:
            b = self.buffers[world]
            for key, value in [
                ("force_time", float(times[world])),
                ("grf_xyz", force[world].copy()),
                ("solver_torque", torque[world].copy()),
            ]:
                b.setdefault(key, []).append(value)

    def on_episode_end(self, env, world, task, result):
        sid = self.subjects[task.subject_index].subject_id
        out = self.output_dir / sid
        out.mkdir(parents=True, exist_ok=True)
        b = self.buffers.pop(world)
        name = self.subjects[task.subject_index].motions[task.motion_index].motion_id
        a = np.asarray(b["muscle"])
        q = np.asarray(b["joints"])
        tau = np.asarray(b["torque"])
        grf = np.asarray(b["grf"])
        xyz = np.asarray(b["grf_xyz"])
        solver_tau = np.asarray(b["solver_torque"])
        contacts = np.asarray(b["contacts"])
        t = np.asarray(b["time"], dtype=float)
        ft = np.asarray(b["force_time"], dtype=float)
        dt = float(env.step_dt)
        assert len(t) == len(ft) == len(a) == len(q) == len(tau) == len(grf)
        assert np.allclose(ft - t, dt, atol=2e-05), (ft - t).tolist()
        assert len(t) < 2 or np.allclose(np.diff(t), dt, atol=2e-05)
        for v in [a, q, tau, grf, xyz, solver_tau]:
            assert np.isfinite(v).all()
        tc = np.arange(len(t)) * dt
        grid = np.arange(0, (len(t) - 1) * dt + 1e-09, 0.01)
        near = np.minimum(np.rint(grid / dt).astype(int), len(t) - 1)
        ev = extract_valid_strides_flexible(
            contacts[near],
            side="right",
            allow_implicit_start=False,
            allow_implicit_end=False,
            expected_n=None,
        )

        def interp(v, clock):
            return np.stack(
                [np.interp(grid, clock, v[:, k]) for k in range(v.shape[1])], axis=1
            )

        signals = {
            "activation": interp(a, tc),
            "joint_angles_deg": interp(q, tc),
            "actuator_torque_nm": interp(tau, tc),
            "grf_bw": interp(grf, ft - t[0]),
        }
        valid = [
            e
            for e in ev
            if grid[e.start] >= ft[0] - t[0] - 1e-06
            and grid[e.end - 1] <= ft[-1] - t[0] + 1e-06
        ]
        assert len(valid) == len(ev), (
            "Force samples do not cover the selected complete cycles"
        )
        cycles = {
            k: np.stack([phase_profile(v[e.start : e.end]) for e in ev])
            if ev
            else np.empty((0, v.shape[1], 100))
            for (k, v) in signals.items()
        }
        dest = out / "signals"
        dest.mkdir(exist_ok=True)
        path = dest / (name + ".npz")
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(
            path,
            time_sim_sec=t,
            time_episode_sec=t - t[0],
            activation=a,
            joint_angles_deg=q,
            actuator_torque_nm=tau,
            contacts=contacts,
            reference_steps=np.asarray(b["reference_steps"]),
            root_position_m=np.asarray(b["root"]),
            force_interval_end_sim_sec=ft,
            force_interval_start_sim_sec=ft - self.physics_dt,
            grf_bw=grf,
            grf_world_xyz_n=xyz,
            solver_actuator_torque_nm=solver_tau,
            muscle_names=MUSCLE_NAMES,
            joint_names=JOINT_NAMES,
            foot_names=["left", "right"],
            model_mass_kg=float(env._gait120_grf.mass[world]),
            physics_dt=self.physics_dt,
            control_dt=dt,
        )
        np.savez_compressed(
            out / (name + "_cycles.npz"),
            **cycles,
            names=MUSCLE_NAMES,
            joint_names=JOINT_NAMES,
        )
        info = dict(
            completed=bool(result["completed"]),
            end_reason=str(result["end_reason"]),
            samples=len(t),
            cycles=[dict(start=e.start, end=e.end) for e in ev],
            max_interval_alignment_error_sec=float(np.max(np.abs(ft - t - dt))),
        )
        (out / (name + "_events.json")).write_text(json.dumps(info, indent=2))
        self.audit.append(dict(subject=sid, motion=name, **info))

    def finalize(self):
        (self.output_dir / "signal_audit.json").write_text(
            json.dumps(self.audit, indent=2)
        )
        (self.output_dir / "signal_protocol.json").write_text(
            json.dumps(
                dict(
                    state_sampling="pre_control_step; activation, joint angles and actuator moments",
                    force_sampling="last native physics interval; actual interval end timestamps",
                    cycle_method="complete right heel-strike cycles; 100 points; no optimized shift",
                    torque_definition="actuator generalized moment in Nm, not human inverse-dynamics net moment",
                    joint_names=JOINT_NAMES,
                    muscle_names=MUSCLE_NAMES,
                ),
                indent=2,
            )
        )
