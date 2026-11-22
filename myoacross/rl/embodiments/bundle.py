"""Serializable model arrays and metadata for one embodiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import ctypes
import dataclasses
import hashlib
import json

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from myoacross.rl.embodiments.features import MorphologyFeatures
from myoacross.rl.embodiments.virtual.parameters import (
    REGION_NAMES,
    GenerationParameters,
    HeightScaleParameters,
    SegmentScaleParameters,
)


@dataclass(slots=True)
class EmbodimentBundle:
    """CPU model arrays, measured features and source metadata."""

    source_subject_id: str | None
    variant_id: str

    embodiment: str
    morphology_source: str
    parameters: GenerationParameters | None

    features: MorphologyFeatures
    subject_height_m: float
    model_signature: str
    arrays: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        for name in (
            "variant_id",
            "model_signature",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if self.embodiment not in (
            "baseline",
            "height_scale",
            "segment_scale",
        ):
            raise ValueError(
                f"Unsupported embodiment: {self.embodiment!r}"
            )

        if self.morphology_source not in ("subject", "virtual"):
            raise ValueError(
                f"Unsupported morphology_source: "
                f"{self.morphology_source!r}"
            )

        if self.embodiment == "baseline":
            if self.source_subject_id is not None:
                raise ValueError(
                    "Shared baseline bundles must have source_subject_id=None"
                )
            if self.morphology_source != "subject":
                raise ValueError(
                    "Baseline requires morphology_source='subject'"
                )
        elif (
            not isinstance(self.source_subject_id, str)
            or not self.source_subject_id.strip()
        ):
            raise ValueError(
                "Height and segment bundles require a non-empty "
                "source_subject_id"
            )

        if self.morphology_source == "subject":
            if self.parameters is not None:
                raise ValueError(
                    "Subject bundles must not contain generation parameters"
                )
        else:
            parameter_types = {
                "height_scale": HeightScaleParameters,
                "segment_scale": SegmentScaleParameters,
            }
            expected_type = parameter_types[self.embodiment]

            if not isinstance(self.parameters, expected_type):
                raise TypeError(
                    f"{self.embodiment} requires "
                    f"{expected_type.__name__}"
                )

        if not isinstance(self.features, MorphologyFeatures):
            raise TypeError("features must be MorphologyFeatures")

        self.subject_height_m = float(self.subject_height_m)
        if (
            not np.isfinite(self.subject_height_m)
            or self.subject_height_m <= 0.0
        ):
            raise ValueError(
                "subject_height_m must be finite and positive"
            )

        if self.parameters is not None:
            if not np.isclose(
                self.features.body_mass_kg,
                self.parameters.body_mass_kg,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError(
                    "Feature mass does not match generation target mass"
                )

        if not self.arrays:
            raise ValueError("Bundle arrays must not be empty")

        copied = {}
        for name, value in self.arrays.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Array names must be non-empty strings"
                )

            array = np.array(value, copy=True, order="C")
            if array.dtype.kind not in "biuf":
                raise ValueError(
                    f"Unsupported dtype for {name}: {array.dtype}"
                )

            array.setflags(write=False)
            copied[name] = array

        self.arrays = copied

    def save(self, path: str | Path) -> None:
        """Save one bundle without overwriting an existing file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.parameters is None:
            parameter_record = None
        elif isinstance(self.parameters, SegmentScaleParameters):
            parameter_record = {
                "region_names": list(REGION_NAMES),
                "region_scales": list(self.parameters.region_scales),
                "body_mass_kg": self.parameters.body_mass_kg,
            }
        elif isinstance(self.parameters, HeightScaleParameters):
            parameter_record = {
                "uniform_scale": self.parameters.uniform_scale,
                "body_mass_kg": self.parameters.body_mass_kg,
            }
        else:
            raise TypeError(
                f"Unsupported parameters: {type(self.parameters).__name__}"
            )

        array_names = sorted(self.arrays)
        metadata = {
            "source_subject_id": self.source_subject_id,
            "variant_id": self.variant_id,
            "embodiment": self.embodiment,
            "morphology_source": self.morphology_source,
            "parameters": parameter_record,
            "features": self.features.to_dict(),
            "subject_height_m": self.subject_height_m,
            "model_signature": self.model_signature,
            "array_names": array_names,
        }

        payload = {
            f"array_{index}": self.arrays[name]
            for index, name in enumerate(array_names)
        }
        payload["metadata"] = np.asarray(
            json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
            )
        )

        with path.open("xb") as file:
            np.savez_compressed(file, **payload)

    @classmethod
    def load(cls, path: str | Path) -> EmbodimentBundle:
        """Load and validate bundle metadata and CPU arrays."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].item())

            embodiment = metadata["embodiment"]
            morphology_source = metadata["morphology_source"]
            record = metadata["parameters"]

            if record is None:
                parameters = None
            elif not isinstance(record, dict):
                raise ValueError(
                    "Generation parameters must be an object or null"
                )
            elif embodiment == "segment_scale":
                if set(record) != {
                    "region_names",
                    "region_scales",
                    "body_mass_kg",
                }:
                    raise ValueError(
                        "Invalid segment-scale parameter fields"
                    )

                if tuple(record["region_names"]) != REGION_NAMES:
                    raise ValueError(
                        "Region names or order do not match"
                    )

                parameters = SegmentScaleParameters(
                    region_scales=tuple(record["region_scales"]),
                    body_mass_kg=record["body_mass_kg"],
                )
            elif embodiment == "height_scale":
                if set(record) != {
                    "uniform_scale",
                    "body_mass_kg",
                }:
                    raise ValueError(
                        "Invalid height-scale parameter fields"
                    )

                parameters = HeightScaleParameters(
                    uniform_scale=record["uniform_scale"],
                    body_mass_kg=record["body_mass_kg"],
                )
            else:
                raise ValueError(
                    f"{embodiment!r} cannot contain generation parameters"
                )

            array_names = metadata["array_names"]
            if (
                not isinstance(array_names, list)
                or not all(
                    isinstance(name, str) and name
                    for name in array_names
                )
                or len(set(array_names)) != len(array_names)
            ):
                raise ValueError("Invalid bundle array names")

            expected_keys = {
                "metadata",
                *(f"array_{i}" for i in range(len(array_names))),
            }
            if set(archive.files) != expected_keys:
                raise ValueError(
                    "Bundle archive contents do not match metadata"
                )

            arrays = {
                name: archive[f"array_{index}"]
                for index, name in enumerate(array_names)
            }

        return cls(
            source_subject_id=metadata["source_subject_id"],
            variant_id=metadata["variant_id"],
            embodiment=embodiment,
            morphology_source=morphology_source,
            parameters=parameters,
            features=MorphologyFeatures.from_dict(
                metadata["features"]
            ),
            subject_height_m=metadata["subject_height_m"],
            model_signature=metadata["model_signature"],
            arrays=arrays,
        )


class WarpBundleSwitcher:
    """
    Extract, validate, prepare and inject virtual embodiment parameters.

    Bind to a Warp model after its array layout has been finalized.
    Extraction, signature checks and device allocation belong to offline
    processing or initialization.

    Online injection copies values into existing model arrays. 
    The caller must order these copies before simulation, 
    reset state and initialize the new subject's reference trajectory.
    """

    FIELD_NAMES = (
        # Body geometry and inertia.
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_inertia",

        # Joint geometry and constraints.
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "eq_data",

        # Geom geometry and bounding volumes.
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_aabb",
        "geom_rbound",

        # Tendon path sites.
        "site_pos",
        "site_quat",
        "site_size",

        # Muscle and tendon parameters.
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_lengthrange",
        "tendon_lengthspring",

        # Derived reference constants.
        "body_subtreemass",
        "body_invweight0",
        "dof_invweight0",
        "dof_length",
        "tendon_length0",
        "tendon_invweight0",
        "actuator_acc0",
        "stat.meaninertia",

        # Camera and light transforms.
        "cam_quat",
        "cam_pos0",
        "cam_poscom0",
        "cam_mat0",
        "light_dir",
        "light_pos0",
        "light_poscom0",
        "light_dir0",

        # Mesh resources stored in the Warp model.
        "mesh_vert",
        "mesh_normal",
        "mesh_pos",
        "mesh_quat",
    )

    def __init__(self, model: mjwarp.Model) -> None:
        self.model = model

        # Resolve fields now so missing fields fail during initialization.
        self._arrays = {
            name: self._resolve_array(name)
            for name in self.FIELD_NAMES
        }

    def _resolve_array(self, name: str) -> wp.array:
        """Resolve both direct and nested fields."""

        value = self.model

        for component in name.split("."):
            value = getattr(value, component)

        if not isinstance(value, wp.array):
            raise TypeError(f"{name} is not a Warp array")

        return value

    def extract_arrays(self) -> dict[str, np.ndarray]:
        """Copy selected model arrays to CPU for offline storage."""

        return {
            name: np.array(array.numpy(), copy=True, order="C")
            for name, array in self._arrays.items()
        }
    
    def validate_array_layout(
        self,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        """
        Check field names, NumPy shapes and dtypes.

        This checks array layout only, not model topology or configuration.
        Intended for offline validation or initialization.
        """

        expected = set(self.FIELD_NAMES)
        actual = set(arrays)

        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Bundle fields do not match: missing={missing}, extra={extra}")

        for name, target in self._arrays.items():
            source = arrays[name]

            if not isinstance(source, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array")

            # Warp vec3/quat/matrix element dimensions appear explicitly in the NumPy representation.
            reference = target.numpy()

            if source.shape != reference.shape:
                raise ValueError(f"{name}: expected shape {reference.shape}, got {source.shape}")

            if source.dtype != reference.dtype:
                raise ValueError(f"{name}: expected dtype {reference.dtype}, got {source.dtype}")

    def compute_model_signature(
        self,
        mj_model: mujoco.MjModel,
    ) -> str:
        """
        Hash the fixed model configuration and injectable array layout.

        mj_model must be the CPU model corresponding to self.model.
        Run offline or during initialization, not during rollout.
        """

        def describe(value, path: str):
            if isinstance(value, wp.array):
                array = np.ascontiguousarray(value.numpy())

                record = {
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }

                # Injectable values may differ between virtual subjects.
                # Their layout must remain compatible.
                if path not in self.FIELD_NAMES:
                    record["sha256"] = hashlib.sha256(
                        array.tobytes()
                    ).hexdigest()

                return record

            if dataclasses.is_dataclass(value):
                return {
                    field.name: describe(
                        getattr(value, field.name),
                        (
                            f"{path}.{field.name}"
                            if path
                            else field.name
                        ),
                    )
                    for field in dataclasses.fields(value)
                }

            if isinstance(value, (tuple, list)):
                return [
                    describe(item, f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]

            # Handles fixed Warp vector/matrix values.
            if isinstance(value, ctypes.Array):
                return np.asarray(value).tolist()

            if isinstance(value, np.generic):
                return value.item()

            if value is None or isinstance(
                value, (bool, int, float, str)
            ):
                return value

            # Do not silently ignore unrecognized configuration fields.
            raise TypeError(f"Unsupported signature field {path}: {type(value)}")

        # Warp does not retain all CPU object-name information.
        names = {"names": hashlib.sha256(bytes(mj_model.names)).hexdigest(),}

        for name in dir(mj_model):
            if name.startswith("name_") and name.endswith("adr"):
                array = np.ascontiguousarray(getattr(mj_model, name))
                names[name] = hashlib.sha256(array.tobytes()).hexdigest()

        record = {
            "versions": {
                name: version(name)
                for name in (
                    "mujoco",
                    "mujoco-warp",
                    "warp-lang",
                    "mjlab",
                )
            },
            "fields": list(self.FIELD_NAMES),
            "names": names,
            "model": describe(self.model, ""),
        }

        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def validate_bundle(
        self,
        bundle: EmbodimentBundle,
        mj_model: mujoco.MjModel,
    ) -> None:
        """Validate a bundle before preparing it for online injection."""

        signature = self.compute_model_signature(mj_model)

        if bundle.model_signature != signature:
            raise ValueError(
                "Morphology bundle is incompatible with this model: "
                f"expected {signature}, "
                f"got {bundle.model_signature}"
            )

        self.validate_array_layout(bundle.arrays)

    def _check_target_arrays(self) -> None:
        """Ensure the model still owns the arrays resolved at initialization."""

        for name, original in self._arrays.items():
            if self._resolve_array(name) is not original:
                raise RuntimeError(
                    f"Model array {name} was replaced; "
                    "recreate the switcher and prepare bundles again"
                )
                
    def prepare_bundle(
        self,
        bundle: EmbodimentBundle,
        mj_model: mujoco.MjModel,
    ) -> PreparedWarpBundle:
        """Validate and upload a bundle before online use."""

        self._check_target_arrays()
        self.validate_bundle(bundle, mj_model)

        arrays = {
            name: wp.array(
                data=bundle.arrays[name],
                dtype=target.dtype,
                shape=target.shape,
                device=target.device,
            )
            for name, target in self._arrays.items()
        }

        return PreparedWarpBundle(
            bundle=bundle,
            _switcher=self,
            _arrays=arrays,
        )

    def inject(
        self,
        prepared: PreparedWarpBundle,
    ) -> None:
        """
        Copy prepared values into existing model arrays.

        Does not reset simulation state or recompute model constants.
        Copies must be ordered before subsequent simulation operations.
        """

        if prepared._switcher is not self:
            raise ValueError("Prepared bundle belongs to another switcher")

        # Check all target arrays before performing any writes.
        self._check_target_arrays()

        for name, target in self._arrays.items():
            wp.copy(target, prepared._arrays[name])


@dataclass(frozen=True, slots=True)
class PreparedWarpBundle:
    """
    Device-resident parameter copies prepared for one switcher.

    These arrays are sources for injection, not the arrays read directly
    by the simulation. Preparation allocates device memory; reuse the
    prepared object when switching back to the same virtual subject.

    Keep the number of resident bundles bounded to control memory usage.
    """

    bundle: EmbodimentBundle

    # Prepared arrays belong to one specific switcher.
    _switcher: WarpBundleSwitcher
    _arrays: dict[str, wp.array]
