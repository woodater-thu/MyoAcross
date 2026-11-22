"""
Exact performance paths for GMR-Fit retargeting.

Every function here reproduces an upstream computation (GMR, mink or
MuJoCo) with cheaper numerics; none changes the optimization problem.
Disable them with ``performance.fast_ik: false`` or
``performance.fast_frames: false`` to run the upstream code for A/B checks.
"""

import sys

import daqp
import mink
import mujoco
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


_DAQP_INFINITY = 1e30

# Process-wide switch read at call time by the mink patches below.
ENABLED = {"ik": False}


def _compute_qp_objective(configuration, tasks, damping):
    # mink exposes the objective assembly only through its solve_ik module.
    return sys.modules["mink.solve_ik"]._compute_qp_objective(
        configuration, tasks, damping
    )


def supports_box_qp(solver, limits, constraints) -> bool:
    """Return whether the QP matches the MyoFullBody GMR setup handled here."""
    return (
        solver == "daqp"
        and not constraints
        and limits is not None
        and len(limits) == 1
        and type(limits[0]) is mink.ConfigurationLimit
        and len(limits[0]._ball_qposadr) == 0
    )


def solve_ik_box(configuration, tasks, dt, damping, limit) -> np.ndarray:
    """
    Solve mink's differential IK QP with joint limits as DAQP simple bounds.

    mink stacks ConfigurationLimit as dense rows G = [P; -P] (244 x 128 for
    MyoFullBody). The same constraint is lb <= dq <= ub, which DAQP handles
    natively. DoFs that no task touches have a diagonal Hessian row and a
    zero gradient; when zero lies inside their bounds their optimal step is
    exactly zero, so they are removed from the solve.
    """
    model = configuration.model
    nv = model.nv

    hessian, gradient = _compute_qp_objective(configuration, tasks, damping)

    q = configuration.q
    step_to_upper = np.empty(nv)
    step_from_lower = np.empty(nv)
    mujoco.mj_differentiatePos(model, step_to_upper, 1.0, q, limit.upper)
    mujoco.mj_differentiatePos(model, step_from_lower, 1.0, limit.lower, q)

    upper = np.full(nv, _DAQP_INFINITY)
    lower = np.full(nv, -_DAQP_INFINITY)
    indices = limit.indices
    upper[indices] = limit.gain * step_to_upper[indices]
    lower[indices] = -limit.gain * step_from_lower[indices]

    off_diagonal = hessian.copy()
    off_diagonal.flat[:: nv + 1] = 0.0
    coupled = (
        np.any(off_diagonal != 0.0, axis=1)
        | (gradient != 0.0)
        | (lower > 0.0)
        | (upper < 0.0)
    )
    active = np.flatnonzero(coupled)

    delta_q = np.zeros(nv)
    if len(active):
        solution, _, exitflag, _ = daqp.solve(
            np.ascontiguousarray(hessian[np.ix_(active, active)]),
            np.ascontiguousarray(gradient[active]),
            np.zeros((0, len(active))),
            np.ascontiguousarray(upper[active]),
            np.ascontiguousarray(lower[active]),
            np.zeros(len(active), dtype=np.intc),
        )
        if exitflag <= 0:
            raise mink.exceptions.NoSolutionFound("daqp")
        delta_q[active] = solution

    return delta_q / dt


def install_equality_mask_patch() -> None:
    """
    Skip mink's per-call np.isin when a task covers every model equality.

    The mask is then simply "is an equality row", identical to upstream.
    """
    from mink.tasks.equality_constraint_task import EqualityConstraintTask

    original = EqualityConstraintTask._update_active_constraints
    if getattr(original, "_myoacross_fast", False):
        return

    def update_active_constraints(self, configuration):
        if not ENABLED["ik"] or len(self._eq_ids) != configuration.model.neq:
            return original(self, configuration)
        data = configuration.data
        self._mask = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        self.cost = self._cost[self._eq_id_to_pos[data.efc_id[self._mask]]]

    update_active_constraints._myoacross_fast = True
    EqualityConstraintTask._update_active_constraints = update_active_constraints


def install_vectorized_targets(retargeter) -> None:
    """
    Replace GMR's per-body SciPy target update with one batched update.

    Reproduces GeneralMotionRetargeting.update_targets for unit human scale
    tables: offset rotation R*=R^H dR, position p*=p^H+R* d, ground offset,
    optional lowest-point grounding, then FrameTask targets.
    """
    if any(scale != 1.0 for scale in retargeter.human_scale_table.values()):
        raise ValueError("Vectorized targets require unit human scale factors")

    tables = []
    if retargeter.use_ik_match_table1:
        tables.append(retargeter.human_body_to_task1)
    if retargeter.use_ik_match_table2:
        tables.append(retargeter.human_body_to_task2)

    human_names = list(retargeter.human_scale_table)
    row = {name: index for index, name in enumerate(human_names)}
    task_rows = [(task, row[name]) for table in tables for name, task in table.items()]
    cache = {"offsets": None, "rotation": None, "position": None}

    def offsets():
        learned = retargeter.learned_offsets
        if cache["rotation"] is not None and cache["offsets"] is learned:
            return cache["rotation"], cache["position"]

        if (
            learned is not None
            and "pos_offsets" in learned
            and "rot_offsets" in learned
        ):
            global_rot = learned.get("global_rot_offset")
            global_rotation = (
                Rotation.from_quat(global_rot, scalar_first=True)
                if global_rot is not None
                else Rotation.identity()
            )
            # Compose one body at a time, exactly as GMR does per frame.
            rotation = Rotation.concatenate(
                [
                    global_rotation
                    * Rotation.from_quat(
                        learned["rot_offsets"][name], scalar_first=True
                    )
                    for name in human_names
                ]
            )
            position = np.array(
                [learned["pos_offsets"][name] for name in human_names],
                dtype=float,
            )
        else:
            rotation = Rotation.concatenate(
                [retargeter.rot_offsets1[name] for name in human_names]
            )
            position = np.array(
                [retargeter.pos_offsets1[name] for name in human_names],
                dtype=float,
            )

        cache.update(offsets=learned, rotation=rotation, position=position)
        return rotation, position

    root_name = retargeter.human_root_name
    root_row = row[root_name]

    def update_targets(human_data, offset_to_ground=False):
        rotation_offsets, position_offsets = offsets()

        # Unit scaling in GMR is (p - p_root) * 1 + p_root, evaluated in the
        # source dtype (float32 for SMPL-H joints); keep the same rounding.
        root_position = np.asarray(human_data[root_name][0])
        positions = np.array([np.asarray(human_data[name][0]) for name in human_names])
        positions = (positions - root_position) + root_position
        positions[root_row] = root_position

        # GMR applies the rotation offset, round-trips through a quaternion,
        # then rotates the local position offset with the updated rotation.
        quaternions = (
            Rotation.from_quat(
                np.array([np.asarray(human_data[name][1]) for name in human_names]),
                scalar_first=True,
            )
            * rotation_offsets
        ).as_quat(scalar_first=True)
        # SciPy's single-rotation apply rounds differently from its batched
        # apply; keep GMR's per-body call so targets stay bit-identical.
        positions = positions + np.array(
            [
                Rotation.from_quat(quaternion, scalar_first=True).apply(offset)
                for quaternion, offset in zip(quaternions, position_offsets)
            ]
        )

        positions[:, 2] -= retargeter.ground_offset
        if offset_to_ground:
            positions[:, 2] -= positions[:, 2].min()

        for task, index in task_rows:
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(quaternions[index]),
                    positions[index],
                )
            )

    retargeter.update_targets = update_targets


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _slerp_rotvec(rotvec1, rotvec2, alpha) -> np.ndarray:
    """Batched copy of GMR's slerp(): shortest arc, lerp when dot > 0.9995."""
    q1 = Rotation.from_rotvec(rotvec1).as_quat()
    q2 = Rotation.from_rotvec(rotvec2).as_quat()
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    dot = np.sum(q1 * q2, axis=-1)
    flip = dot < 0.0
    q2 = np.where(flip[:, None], -q2, q2)
    dot = np.where(flip, -dot, dot)

    lerp = q1 + alpha[:, None] * (q2 - q1)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.sin(theta) / np.sin(theta_0)
        s0 = np.cos(theta) - dot * s1
    slerp = s0[:, None] * q1 + s1[:, None] * q2

    quaternions = np.where((dot > 0.9995)[:, None], lerp, slerp)
    return Rotation.from_quat(quaternions).as_rotvec()


def smplh_frames(smplh_data, body_model, smplh_output, tgt_fps=30):
    """
    Batched equivalent of GMR get_smplh_data_offline_fast.

    Same FPS alignment (frame_skip, endpoint-spanning linspace, per-joint
    SLERP of local rotations, linear joint positions) and same per-frame
    dictionaries; SciPy is called once per joint instead of once per frame.
    """
    from smplx.joint_names import JOINT_NAMES

    src_fps = _to_numpy(smplh_data["mocap_frame_rate"]).item()
    num_frames = smplh_data["pose_body"].shape[0]

    global_orient = _to_numpy(smplh_output.global_orient).reshape(num_frames, 3)
    full_pose = _to_numpy(smplh_output.full_pose).reshape(num_frames, -1, 3)
    joints = _to_numpy(smplh_output.joints).squeeze()
    parents = [int(parent) for parent in _to_numpy(body_model.parents)]
    joint_names = JOINT_NAMES[: len(parents)]

    if tgt_fps < src_fps:
        frame_skip = int(src_fps / tgt_fps)
        new_num_frames = num_frames // frame_skip

        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames - 1, new_num_frames)
        index1 = np.floor(target_time).astype(int)
        index2 = np.minimum(index1 + 1, num_frames - 1)
        alpha = target_time - index1

        global_orient = _slerp_rotvec(
            global_orient[index1], global_orient[index2], alpha
        )
        num_pose_joints = full_pose.shape[1]
        full_pose = _slerp_rotvec(
            full_pose[index1].reshape(-1, 3),
            full_pose[index2].reshape(-1, 3),
            np.repeat(alpha, num_pose_joints),
        ).reshape(new_num_frames, num_pose_joints, 3)
        joints = interp1d(original_time, joints, axis=0, kind="linear")(target_time)

        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        aligned_fps = tgt_fps

    orientations = []
    for index in range(len(joint_names)):
        if index == 0:
            orientation = Rotation.from_rotvec(global_orient)
        else:
            orientation = orientations[parents[index]] * Rotation.from_rotvec(
                full_pose[:, index]
            )
        orientations.append(orientation)
    quaternions = np.stack(
        [orientation.as_quat(scalar_first=True) for orientation in orientations],
        axis=1,
    )

    frames = []
    for frame_joints, frame_quaternions in zip(joints, quaternions):
        frames.append(
            {
                name: (frame_joints[index], frame_quaternions[index])
                for index, name in enumerate(joint_names)
            }
        )

    return frames, aligned_fps
