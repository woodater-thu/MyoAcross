"""Signal names and phase normalization shared by references and evaluation."""

import numpy as np
from mm_evalkit.workflows.gait.channels import common_joint_names
from mm_evalkit.datasets.gait120 import Gait120_muscle_map

JOINT_NAMES = tuple(common_joint_names)
MUSCLE_MAP = dict(Gait120_muscle_map)
MUSCLE_NAMES = tuple(MUSCLE_MAP)
EMG_CHANNELS = (*MUSCLE_NAMES, "soleus_lateral")
EMG_MAP = dict(MUSCLE_MAP, soleus_lateral="SoleusLateralis")
SAGITTAL_JOINTS = tuple(
    f"{joint}_{side}"
    for side in ("r", "l")
    for joint in ("hip_flexion", "knee_angle", "ankle_angle")
)


def phase_profile(values):
    from mm_evalkit.workflows.gait.cycles import interpolate_gait_cycle

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("A phase profile requires at least two finite time samples")
    return np.stack(
        [
            interpolate_gait_cycle(values, i, [(0, len(values))])[0][0]
            for i in range(values.shape[1])
        ]
    )


def stance_profiles(time, force, valid=None):
    """Normalize fully observed contacts; exclude clipped or <0.2 s contacts."""
    time, force = np.asarray(time), np.asarray(force)
    changes = np.diff(np.r_[False, force > 0.05, False].astype(int))
    profiles = []
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if start == 0 or end == len(force) or time[end - 1] - time[start] < 0.2:
            continue
        if valid is not None and not valid[start:end].all():
            continue
        profiles.append(phase_profile(force[start:end, None])[0])
    return profiles


def pearson(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if (
        not (np.isfinite(first).all() and np.isfinite(second).all())
        or min(np.ptp(first), np.ptp(second)) <= 1e-12
    ):
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])
