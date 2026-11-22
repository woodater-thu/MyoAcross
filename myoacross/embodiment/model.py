"""Base class for subject-specific embodiment scaling."""

import hashlib
from abc import ABC, abstractmethod
import mujoco
import numpy as np

from myoacross.embodiment.measurements import (
    compute_myofullbody_landmarks,
    compute_myofullbody_measurements,
    compute_myofullbody_height,
)

# Auxiliary bodies whose baseline mass and inertia remain fixed
FIXED_INERTIA_BODIES: frozenset[str] = frozenset(
    {
        "Full Body",
        "head_attach",
        "bimanual_attachment",
        "root",
        "thorax",
        "cervical_spine",
        "clavphant_r",
        "scapphant_r",
        "humphant_r",
        "humphant1_r",
        "clavphant_l",
        "scapphant_l",
        "humphant_l",
        "humphant1_l",
    }
)


def model_sha256(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


class OptimizationFailure(RuntimeError):
    """Carry diagnostics from an unsuccessful scale optimization."""

    def __init__(self, report: dict) -> None:
        super().__init__(report["error"])
        self.report = report


class EmbodimentModel(ABC):
    """Common inputs and interface for embodiment scaling strategies."""

    def __init__(
        self,
        baseline_spec: mujoco.MjSpec,
    ) -> None:
        """Retain an independent baseline description and compiled model."""

        self.baseline_spec = baseline_spec.copy()
        self.baseline_model = self.baseline_spec.copy().compile()

    @abstractmethod
    def solve(self) -> dict[str, float]:
        """
        Return positive, finite scales for every non-world baseline body:
        {'body_name': body_scale, ...}
        This method is realized by subclass
        """

        raise NotImplementedError("Subclasses must implement solve()")

    def build(
        self,
        body_scales: dict[str, float],
        target_mass_kg: float,
    ) -> mujoco.MjSpec:
        spec = self.scale_geometry(body_scales)
        spec = self.scale_inertia(spec, body_scales, target_mass_kg)
        spec = self.scale_muscles(spec)
        return spec

    def scale_geometry(
        self,
        body_scales: dict[str, float],
    ) -> mujoco.MjSpec:
        """Update geometry of MyoFullBody."""

        model = self.baseline_model

        # Validate body scales
        body_names = {model.body(i).name for i in range(1, model.nbody)}
        missing = body_names - set(body_scales)
        unknown = set(body_scales) - body_names

        if missing:
            raise KeyError(f"Missing body scales: {sorted(missing)}")
        if unknown:
            raise KeyError(f"Unknown bodies in scales: {sorted(unknown)}")

        scales = {name: float(value) for name, value in body_scales.items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise ValueError("Body scales must be finite and positive")
        if scales.get("Full Body") != 1.0:
            raise ValueError("The 'Full Body' scale must be 1.0")

        spec = self.baseline_spec.copy()
        mesh_scales: dict[str, float] = {}
        joint_scales: dict[str, float] = {}

        # These rules target the current MyoFullBody structure
        if any(list(body.frames) for body in spec.bodies):
            raise ValueError("Geometry scaling does not yet support explicit frames")
        if len(spec.keys):
            raise ValueError("Geometry scaling does not yet support keyframes")

        for body in spec.bodies:
            if body.name == "world":
                continue

            body_id = model.body(body.name).id
            scale = scales[body.name]
            parent_id = int(model.body_parentid[body_id])
            parent_scale = 1.0 if parent_id == 0 else scales[model.body(parent_id).name]

            # Parent dimensions control the child's connection offset
            body.pos[:] *= parent_scale

            # Keep physical inertia values fixed during geometry fitting
            body.explicitinertial = True
            body.mass = float(model.body_mass[body_id])
            body.inertia[:] = model.body_inertia[body_id]
            body.fullinertia[:] = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
            body.ipos[:] = model.body_ipos[body_id] * scale
            body.iquat[:] = model.body_iquat[body_id]

            # Scale visual, collision and wrapping geometry
            for geom in body.geoms:
                geom.pos[:] *= scale
                if np.isfinite(geom.fromto).all():
                    geom.fromto[:] *= scale

                if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                    previous_scale = mesh_scales.get(geom.meshname)
                    if previous_scale is not None and previous_scale != scale:
                        raise ValueError(
                            f"Mesh {geom.meshname!r} is shared by bodies with different scales"
                        )

                    mesh_scales[geom.meshname] = scale
                else:
                    geom.size[:] *= scale

            # Scale muscle attachment points and other landmarks
            for site in body.sites:
                site.pos[:] *= scale
                site.size[:] *= scale
                if np.isfinite(site.fromto).all():
                    site.fromto[:] *= scale

            # Scale local anchors and translational joint coordinates
            for joint in body.joints:
                joint.pos[:] *= scale
                joint_scales[joint.name] = 1.0

                if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                    joint_scales[joint.name] = parent_scale
                    joint.range[:] *= parent_scale
                    joint.ref *= parent_scale
                    joint.springref *= parent_scale
                    joint.margin *= parent_scale

        # Scale each mesh asset once
        for mesh in spec.meshes:
            if mesh.name in mesh_scales:
                if np.any(mesh.refpos):
                    raise ValueError(
                        f"Mesh {mesh.name!r} has an unsupported nonzero reference position"
                    )

                mesh.scale[:] *= mesh_scales[mesh.name]

        # Update joint polynomial constraints for scaled coordinates
        for equality in spec.equalities:
            if equality.type != mujoco.mjtEq.mjEQ_JOINT:
                raise ValueError(f"Unsupported equality constraint: {equality.name!r}")

            output_scale = joint_scales[equality.name1]
            input_scale = joint_scales[equality.name2] if equality.name2 else 1.0
            equality.data[:5] *= output_scale / input_scale ** np.arange(5)

        return spec

    def scale_inertia(
        self,
        spec: mujoco.MjSpec,
        body_scales: dict[str, float],
        target_mass_kg: float,
    ) -> mujoco.MjSpec:
        """Scale mass and inertia to match the explicitly supplied mass."""

        model = self.baseline_model

        # Body scales have already been validated by scale_geometry().
        missing = FIXED_INERTIA_BODIES - set(body_scales)
        if missing:
            raise KeyError(f"Missing auxiliary bodies: {sorted(missing)}")

        # Compute fixed auxiliary mass and geometrically scaled tissue mass.
        auxiliary_mass = 0.0
        scaled_mass = 0.0

        for body_id in range(1, model.nbody):
            name = model.body(body_id).name
            mass = float(model.body_mass[body_id])

            if name in FIXED_INERTIA_BODIES:
                auxiliary_mass += mass
            else:
                scaled_mass += mass * float(body_scales[name]) ** 3

        if not np.isfinite(scaled_mass) or scaled_mass <= 0.0:
            raise ValueError(f"Invalid geometrically scaled mass: {scaled_mass}")

        # Match the target mass.
        target_mass = float(target_mass_kg)
        if not np.isfinite(target_mass) or target_mass <= auxiliary_mass:
            raise ValueError(
                f"Target mass must be finite and exceed auxiliary mass: {target_mass}"
            )

        density_scale = (target_mass - auxiliary_mass) / scaled_mass
        if not np.isfinite(density_scale) or density_scale <= 0.0:
            raise ValueError(f"Invalid density scale: {density_scale}")

        # Explicit inertias must not be clamped or automatically balanced.
        spec.compiler.boundmass = 0.0
        spec.compiler.boundinertia = 0.0
        spec.compiler.balanceinertia = False

        for body in spec.bodies:
            if body.name == "world":
                continue

            body_id = model.body(body.name).id

            # Preserve artificial mass and inertia on auxiliary bodies.
            if body.name in FIXED_INERTIA_BODIES:
                body.mass = float(model.body_mass[body_id])
                body.inertia[:] = model.body_inertia[body_id]
                continue

            scale = float(body_scales[body.name])
            mass = float(model.body_mass[body_id] * scale**3 * density_scale)
            inertia = model.body_inertia[body_id] * scale**5 * density_scale

            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"Invalid scaled mass for body {body.name!r}: {mass}")
            if not np.isfinite(inertia).all() or np.any(inertia <= 0.0):
                raise ValueError(
                    f"Invalid scaled inertia for body {body.name!r}: {inertia}"
                )

            # Handle roundoff near the principal-inertia triangle boundary.
            largest = int(np.argmax(inertia))
            other_sum = float(np.sum(np.delete(inertia, largest)))

            if inertia[largest] > other_sum * (1.0 + 1e-10):
                raise ValueError(
                    f"Principal inertias violate the triangle inequality "
                    f"for body {body.name!r}"
                )

            if inertia[largest] >= other_sum * (1.0 - 1e-12):
                inertia[largest] = other_sum * (1.0 - 1e-12)

            body.mass = mass
            body.inertia[:] = inertia

        return spec

    def scale_muscles(
        self,
        spec: mujoco.MjSpec,
    ) -> mujoco.MjSpec:
        """Scale tendon and muscle parameters using reference-pose path lengths."""

        baseline = self.baseline_model
        model = spec.copy().compile()

        # Evaluate both models in their own reference configurations
        baseline_data = mujoco.MjData(baseline)
        data = mujoco.MjData(model)
        mujoco.mj_forward(baseline, baseline_data)
        mujoco.mj_forward(model, data)

        tendon_scales: dict[str, float] = {}

        for tendon in spec.tendons:
            baseline_id = baseline.tendon(tendon.name).id
            tendon_id = model.tendon(tendon.name).id
            baseline_length = float(baseline_data.ten_length[baseline_id])
            length = float(data.ten_length[tendon_id])

            if not np.isfinite(baseline_length) or baseline_length <= 0.0:
                raise ValueError(
                    f"Invalid baseline length for tendon {tendon.name!r}: {baseline_length}"
                )
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError(
                    f"Invalid scaled length for tendon {tendon.name!r}: {length}"
                )

            scale = length / baseline_length
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Invalid scale for tendon {tendon.name!r}: {scale}")

            tendon_scales[tendon.name] = scale
            reference = self.baseline_spec.tendon(tendon.name)

            # Scale display width and explicit spring lengths
            tendon.width = reference.width * scale
            springlength = np.asarray(reference.springlength)
            tendon.springlength[:] = np.where(
                springlength >= 0.0, springlength * scale, springlength
            )

        for actuator in spec.actuators:
            reference = self.baseline_spec.actuator(actuator.name)

            if reference.trntype != mujoco.mjtTrn.mjTRN_TENDON:
                raise ValueError(
                    f"Actuator {actuator.name!r} does not use a tendon transmission"
                )
            if (
                reference.gaintype != mujoco.mjtGain.mjGAIN_MUSCLE
                or reference.biastype != mujoco.mjtBias.mjBIAS_MUSCLE
            ):
                raise ValueError(
                    f"Actuator {actuator.name!r} does not use muscle gain and bias"
                )

            lengthrange = np.asarray(reference.lengthrange)
            active_force = float(reference.gainprm[2])
            passive_force = float(reference.biasprm[2])

            if not np.isfinite(lengthrange).all() or lengthrange[0] >= lengthrange[1]:
                raise ValueError(
                    f"Invalid baseline length range for muscle {actuator.name!r}"
                )
            if not np.isfinite(active_force) or active_force <= 0.0:
                raise ValueError(
                    f"Muscle {actuator.name!r} requires a positive explicit gain force"
                )
            if not np.isfinite(passive_force) or passive_force <= 0.0:
                raise ValueError(
                    f"Muscle {actuator.name!r} requires a positive explicit bias force"
                )

            scale = tendon_scales[reference.target]

            # Preserve normalized muscle properties and activation dynamics
            actuator.lengthrange[:] = lengthrange * scale

            # Assume muscle cross-sectional area scales with squared length
            actuator.gainprm[2] = active_force * scale**2
            actuator.biasprm[2] = passive_force * scale**2

        return spec

    def validate(
        self,
        model: mujoco.MjModel,
        body_scales: dict[str, float],
        target_mass_kg: float,
    ) -> dict:
        """Validate scaling against the explicitly supplied target mass."""

        baseline = self.baseline_model

        # Validate body correspondence.
        baseline_names = [baseline.body(i).name for i in range(1, baseline.nbody)]
        model_names = [model.body(i).name for i in range(1, model.nbody)]

        if model_names != baseline_names:
            raise ValueError("Compiled body names or ordering differ from baseline")
        if set(body_scales) != set(baseline_names):
            raise ValueError("Body scales do not match the baseline bodies")
        if body_scales.get("Full Body") != 1.0:
            raise ValueError("The 'Full Body' scale must be 1.0")

        # Compute expected mass and inertia scaling.
        auxiliary_mass = 0.0
        scaled_mass = 0.0

        for body_id in range(1, baseline.nbody):
            name = baseline.body(body_id).name
            scale = float(body_scales[name])

            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Invalid scale for body {name!r}: {scale}")

            if name in FIXED_INERTIA_BODIES:
                auxiliary_mass += float(baseline.body_mass[body_id])
            else:
                scaled_mass += float(baseline.body_mass[body_id]) * scale**3

        if not np.isfinite(scaled_mass) or scaled_mass <= 0.0:
            raise ValueError(f"Invalid geometrically scaled mass: {scaled_mass}")

        target_mass = float(target_mass_kg)
        if not np.isfinite(target_mass) or target_mass <= auxiliary_mass:
            raise ValueError(
                f"Target mass must be finite and exceed auxiliary mass: {target_mass}"
            )

        density_scale = (target_mass - auxiliary_mass) / scaled_mass
        if not np.isfinite(density_scale) or density_scale <= 0.0:
            raise ValueError(f"Invalid density scale: {density_scale}")

        # Check each body's mass and inertia.
        for body_id in range(1, model.nbody):
            name = model.body(body_id).name
            scale = float(body_scales[name])
            correction = density_scale

            if name in FIXED_INERTIA_BODIES:
                scale = 1.0
                correction = 1.0

            expected_mass = baseline.body_mass[body_id] * scale**3 * correction
            expected_inertia = baseline.body_inertia[body_id] * scale**5 * correction

            mass = model.body_mass[body_id]
            inertia = model.body_inertia[body_id]

            if not np.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"Invalid compiled mass for body {name!r}")
            if not np.isfinite(inertia).all() or np.any(inertia <= 0.0):
                raise ValueError(f"Invalid compiled inertia for body {name!r}")

            if not np.isclose(mass, expected_mass, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"Compiled mass does not match the scaling rule for body {name!r}"
                )
            if not np.allclose(inertia, expected_inertia, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"Compiled inertia does not match the scaling rule "
                    f"for body {name!r}"
                )

        # Check the final total mass.
        total_mass = float(np.sum(model.body_mass[1:]))

        if not np.isclose(total_mass, target_mass, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"Compiled total mass {total_mass} "
                f"does not match target mass {target_mass}"
            )

        # Validate tendon and actuator correspondence.
        for kind, count in (
            ("tendon", baseline.ntendon),
            ("actuator", baseline.nu),
        ):
            baseline_names = [getattr(baseline, kind)(i).name for i in range(count)]
            model_count = model.ntendon if kind == "tendon" else model.nu
            model_names = [getattr(model, kind)(i).name for i in range(model_count)]

            if model_names != baseline_names:
                raise ValueError(
                    f"Compiled {kind} names or ordering differ from baseline"
                )

        # Compute reference-pose tendon length ratios.
        baseline_data = mujoco.MjData(baseline)
        data = mujoco.MjData(model)
        mujoco.mj_forward(baseline, baseline_data)
        mujoco.mj_forward(model, data)

        if not np.isfinite(baseline_data.ten_length).all() or np.any(
            baseline_data.ten_length <= 0.0
        ):
            raise ValueError("Invalid baseline tendon lengths")

        if not np.isfinite(data.ten_length).all() or np.any(data.ten_length <= 0.0):
            raise ValueError("Invalid compiled tendon lengths")

        tendon_scales = data.ten_length / baseline_data.ten_length

        # Check tendon widths and compiled spring lengths.
        for tendon_id in range(model.ntendon):
            name = model.tendon(tendon_id).name
            scale = tendon_scales[tendon_id]

            expected_width = baseline.tendon_width[tendon_id] * scale
            expected_springlength = baseline.tendon_lengthspring[tendon_id] * scale

            if not np.isclose(
                model.tendon_width[tendon_id],
                expected_width,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Compiled width does not match the scaling rule "
                    f"for tendon {name!r}"
                )

            if not np.allclose(
                model.tendon_lengthspring[tendon_id],
                expected_springlength,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Compiled spring length does not match the scaling rule "
                    f"for tendon {name!r}"
                )

        # Check muscle transmissions.
        if not np.array_equal(model.actuator_trntype, baseline.actuator_trntype):
            raise ValueError(
                "Compiled actuator transmission types differ from baseline"
            )

        if not np.array_equal(model.actuator_trnid, baseline.actuator_trnid):
            raise ValueError(
                "Compiled actuator transmission targets differ from baseline"
            )

        # Check muscle length ranges, force parameters and dynamics.
        for actuator_id in range(model.nu):
            name = model.actuator(actuator_id).name
            tendon_id = int(baseline.actuator_trnid[actuator_id, 0])
            scale = tendon_scales[tendon_id]

            expected_gainprm = baseline.actuator_gainprm[actuator_id].copy()
            expected_biasprm = baseline.actuator_biasprm[actuator_id].copy()
            expected_gainprm[2] *= scale**2
            expected_biasprm[2] *= scale**2

            expected_parameters = {
                "lengthrange": (baseline.actuator_lengthrange[actuator_id] * scale),
                "gainprm": expected_gainprm,
                "biasprm": expected_biasprm,
                "dynprm": baseline.actuator_dynprm[actuator_id],
            }

            for parameter, expected in expected_parameters.items():
                actual = getattr(model, f"actuator_{parameter}")[actuator_id]

                if not np.allclose(actual, expected, rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"Compiled {parameter} does not match the scaling "
                        f"rule for muscle {name!r}"
                    )

        # Compute actual measurements and optional subject target errors.
        landmarks, landmark_names = compute_myofullbody_landmarks(model)
        measurements = compute_myofullbody_measurements(landmarks, landmark_names)
        measurements["height"] = compute_myofullbody_height(model)

        targets = getattr(self, "measurements", {})
        measurement_report = {}

        for name, value in measurements.items():
            entry = {"predicted": float(value)}

            if name in targets:
                target = float(targets[name])
                entry["target"] = target
                entry["error_m"] = float(value - target)

            measurement_report[name] = entry

        return {
            "mass_inertia_checks_passed": True,
            "body_mass_kg": total_mass,
            "target_body_mass_kg": target_mass,
            "body_mass_error_kg": total_mass - target_mass,
            "density_scale": float(density_scale),
            "tendon_muscle_checks_passed": True,
            "measurements": measurement_report,
        }
