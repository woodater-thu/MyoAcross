"""Build the reference MyoFullBody entity used to initialize mjlab."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import mujoco

from mjlab.actuator.actuator import CommandField, TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.mujoco import detect_command_field

from .subjects import SubjectRecord


SubjectSource = SubjectRecord | str | Path | mujoco.MjSpec


def _resolve_xml_path(subject: SubjectSource) -> Path:
    """Resolve a subject record or explicit XML path."""

    value = (
        subject.xml_path
        if isinstance(subject, SubjectRecord)
        else subject
    )
    path = Path(value).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"MyoFullBody XML does not exist: {path}")

    return path


def load_myofullbody_spec(
    subject: SubjectSource,
) -> mujoco.MjSpec:
    """Return a fresh specification for mjlab to modify."""

    if isinstance(subject, mujoco.MjSpec):
        return subject.copy()

    path = _resolve_xml_path(subject)
    return mujoco.MjSpec.from_file(str(path))


def _make_xml_actuator_cfgs(
    spec: mujoco.MjSpec,
) -> tuple[XmlActuatorCfg, ...]:
    """
    Group existing XML actuators by transmission and command field.

    Preserve the XML-defined dynamics, gains and control limits.
    Target expressions match literal object names.
    """

    transmission_map = {
        mujoco.mjtTrn.mjTRN_JOINT: TransmissionType.JOINT,
        mujoco.mjtTrn.mjTRN_JOINTINPARENT: TransmissionType.JOINT,
        mujoco.mjtTrn.mjTRN_TENDON: TransmissionType.TENDON,
        mujoco.mjtTrn.mjTRN_SITE: TransmissionType.SITE,
    }

    groups: dict[
        tuple[TransmissionType, CommandField], list[str]
    ] = defaultdict(list)

    seen_targets: set[tuple[TransmissionType, str]] = set()

    for actuator in spec.actuators:
        transmission = transmission_map.get(actuator.trntype)
        if transmission is None:
            raise ValueError(f"Unsupported transmission for actuator {actuator.name!r}: {actuator.trntype}")

        target = actuator.target
        if not target:
            raise ValueError(f"Actuator {actuator.name!r} has no named target")

        target_key = (transmission, target)
        if target_key in seen_targets:
            raise ValueError(
                "XmlActuatorCfg requires a unique actuator per target; "
                f"duplicate target: {target!r}"
            )
        seen_targets.add(target_key)

        command_field = detect_command_field(actuator)
        groups[(transmission, command_field)].append(target)

    if not groups:
        raise ValueError("MyoFullBody XML contains no actuators")

    return tuple(
        XmlActuatorCfg(
            target_names_expr=tuple(
                re.escape(target) for target in targets
            ),
            transmission_type=transmission,
            command_field=command_field,
        )
        for (transmission, command_field), targets in groups.items()
    )


def make_myofullbody_cfg(
    subject: SubjectSource,
    *,
    init_state: EntityCfg.InitialStateCfg | None = None,
) -> EntityCfg:
    """
    Create the reference entity configuration for environment initialization.

    Parse the XML here without compiling it.
    Subsequent morphology changes must use the bundle injection path,
    rather than rebuilding this entity.
    """
    
    spec = load_myofullbody_spec(subject)

    actuator_cfgs = _make_xml_actuator_cfgs(spec)

    if init_state is None:
        init_state = EntityCfg.InitialStateCfg(
            # Use the first XML keyframe when present.
            # Otherwise use zero scalar-joint positions.
            joint_pos=None if len(spec.keys) > 0 else {".*": 0.0},
            joint_vel={".*": 0.0},
        )

    return EntityCfg(
        # Return a fresh specification whenever mjlab builds the entity.
        spec_fn=lambda: spec.copy(),
        articulation=EntityArticulationInfoCfg(
            actuators=actuator_cfgs,
        ),
        init_state=init_state,
    )