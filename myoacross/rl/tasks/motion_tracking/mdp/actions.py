"""Muscle excitation actions using mjlab's XML actuator interface."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from mjlab.envs.mdp.actions import TendonEffortActionCfg

from myoacross.rl.embodiments.myofullbody_cfg import (
    SubjectSource,
    load_myofullbody_spec,
)


def make_muscle_action_cfg(
    subject: SubjectSource,
    *,
    entity_name: str = "robot",
) -> TendonEffortActionCfg:
    """
    Pass policy excitation directly to XML-defined muscle actuators.

    One action dimension corresponds to one actuated tendon.
    Ordering follows the XML actuator order.

    The policy produces excitation in [0, 1].
    Clipping provides a final boundary guard.
    """
    spec = load_myofullbody_spec(subject)

    tendon_names = []
    seen_targets = set()

    for actuator in spec.actuators:
        if actuator.trntype != mujoco.mjtTrn.mjTRN_TENDON:
            raise ValueError(f"{actuator.name!r}: expected tendon transmission")

        if (
            actuator.dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
            or actuator.gaintype != mujoco.mjtGain.mjGAIN_MUSCLE
        ):
            raise ValueError(f"{actuator.name!r}: expected a muscle actuator")

        target = actuator.target
        if not target:
            raise ValueError(f"{actuator.name!r}: missing tendon target")
        if target in seen_targets:
            raise ValueError(f"Duplicate muscle tendon target: {target!r}")

        control_range = np.asarray(actuator.ctrlrange, dtype=np.float64,)
        if (
            not np.isfinite(control_range).all()
            or control_range[0] > 0.0
            or control_range[1] < 1.0
        ):
            raise ValueError(f"{actuator.name!r}: control range must support [0, 1], got {control_range}")

        seen_targets.add(target)
        tendon_names.append(target)

    if not tendon_names:
        raise ValueError("No muscle actuators found")

    return TendonEffortActionCfg(
        entity_name=entity_name,
        # Tendon actions resolve tendon names in this mjlab interface.
        actuator_names=tuple(re.escape(name) for name in tendon_names),
        preserve_order=True,
        scale=1.0,
        offset=0.0,
        clip={".*": (0.0, 1.0)},
    )

