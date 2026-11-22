"""Offline generation of virtual MyoFullBody embodiments."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import json

import mujoco
import numpy as np

from myoacross.embodiment.model import EmbodimentModel
from myoacross.embodiment.segment_scale.model import BODY_REGION_MAP

from myoacross.rl.embodiments.features import MorphologyFeatures
from myoacross.rl.embodiments.virtual.parameters import (
    REGION_NAMES,
    GenerationParameters,
    HeightScaleParameters,
    SegmentScaleParameters,
)
from myoacross.rl.embodiments.virtual.sampling import MorphologySampler


GeneratedEmbodiment = tuple[
    GenerationParameters,
    mujoco.MjSpec,
    mujoco.MjModel,
    MorphologyFeatures,
    dict,
]


class VirtualEmbodiment(EmbodimentModel):
    """
    Build a body from absolute template scales and target mass.

    Geometry, inertia and muscle scaling are inherited from
    EmbodimentModel. Features are measured from the generated model.
    """

    def __init__(
        self,
        baseline_spec: mujoco.MjSpec,
        parameters: GenerationParameters,
        source_subject_id: str,
    ) -> None:
        if not isinstance(
            parameters,
            (SegmentScaleParameters, HeightScaleParameters),
        ):
            raise TypeError(
                f"Unsupported generation parameters: "
                f"{type(parameters).__name__}"
            )

        subject_id = str(source_subject_id).strip()
        if not subject_id:
            raise ValueError("source_subject_id must not be empty")

        super().__init__(baseline_spec)

        self.parameters = parameters
        self.source_subject_id = subject_id

    def solve(self) -> dict[str, float]:
        """Expand generation parameters into per-body scales."""
        if isinstance(self.parameters, SegmentScaleParameters):
            region_scales = dict(
                zip(REGION_NAMES, self.parameters.region_scales)
            )
            region_scales["global_root"] = 1.0

            return {
                body_name: region_scales[region_name]
                for body_name, region_name in BODY_REGION_MAP.items()
            }

        # Match HeightScaleModel: scale every non-world body uniformly,
        # except the fixed global root.
        body_scales = {}

        for body_id in range(1, self.baseline_model.nbody):
            body_name = mujoco.mj_id2name(
                self.baseline_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )

            if not body_name:
                raise ValueError(
                    f"Body {body_id} has no name"
                )

            body_scales[body_name] = (
                1.0
                if body_name == "Full Body"
                else self.parameters.uniform_scale
            )

        if "Full Body" not in body_scales:
            raise KeyError(
                "MyoFullBody is missing the 'Full Body' root"
            )

        return body_scales

    def generate(
        self,
    ) -> tuple[
        mujoco.MjSpec,
        mujoco.MjModel,
        MorphologyFeatures,
        dict,
    ]:
        """Build, validate and measure one standalone model."""
        body_scales = self.solve()
        target_mass = self.parameters.body_mass_kg

        spec = self.build(body_scales, target_mass)
        model = spec.copy().compile()

        validation = self.validate(
            model,
            body_scales,
            target_mass,
        )

        # validate() measures the generated model in its reference pose.
        measurements = {
            name: entry["predicted"]
            for name, entry in validation["measurements"].items()
        }
        features = MorphologyFeatures.from_measurements(
            measurements,
            validation["body_mass_kg"],
        )

        return spec, model, features, validation


class VirtualEmbodimentGenerator:
    """
    Generate variants around one validated subject.

    The source XML supplies measured features and actual mass.
    Its report supplies absolute scales and the original template path.

    Every variant is rebuilt from the original template. Scaling does
    not accumulate across variants.
    """

    def __init__(
        self,
        subject_dir: str | Path,
        sampler: MorphologySampler,
    ) -> None:
        self.subject_dir = Path(subject_dir).expanduser().resolve()
        self.sampler = sampler

        model_path = self.subject_dir / "model.xml"
        report_path = self.subject_dir / "report.json"

        for path in (model_path, report_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )

        if report.get("status") != "passed":
            raise ValueError(
                "The source subject must have passed validation"
            )

        self.strategy = report.get("strategy")
        if self.strategy not in {"segment_scale", "height_scale"}:
            raise ValueError(
                f"Unsupported virtual embodiment strategy: "
                f"{self.strategy!r}"
            )

        subject_id = report.get("subject_id")
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError(
                "Source report must contain a nonempty subject_id"
            )
        self.source_subject_id = subject_id

        source_model = mujoco.MjModel.from_xml_path(
            str(model_path)
        )
        self.source_features = MorphologyFeatures.from_model(
            source_model
        )

        reported_mass = float(report["mass"]["actual_kg"])
        if not np.isclose(
            self.source_features.body_mass_kg,
            reported_mass,
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(
                "Source XML mass does not match its report"
            )

        mass = self.source_features.body_mass_kg

        if self.strategy == "segment_scale":
            region_scales = report["scales"]["regions"]

            if set(region_scales) != set(REGION_NAMES):
                raise ValueError(
                    "Source region names do not match REGION_NAMES"
                )

            self.center = SegmentScaleParameters(
                region_scales=tuple(
                    region_scales[name]
                    for name in REGION_NAMES
                ),
                body_mass_kg=mass,
            )
        else:
            self.center = HeightScaleParameters(
                uniform_scale=report["scales"]["uniform"],
                body_mass_kg=mass,
            )

        baseline_path = Path(
            report["baseline_path"]
        ).expanduser()

        if not baseline_path.is_absolute():
            baseline_path = self.subject_dir / baseline_path

        self.baseline_path = baseline_path.resolve()
        if not self.baseline_path.is_file():
            raise FileNotFoundError(self.baseline_path)

        self._baseline_spec = mujoco.MjSpec.from_file(
            str(self.baseline_path)
        )

    def generate_one(
        self,
        rng: np.random.Generator,
    ) -> GeneratedEmbodiment:
        """Sample one body; propagate construction/validation errors."""
        parameters = self.sampler.sample(self.center, rng)

        embodiment = VirtualEmbodiment(
            baseline_spec=self._baseline_spec,
            parameters=parameters,
            source_subject_id=self.source_subject_id,
        )

        spec, model, features, validation = embodiment.generate()

        return parameters, spec, model, features, validation

    def generate(
        self,
        count: int,
        rng: np.random.Generator,
    ) -> Iterator[GeneratedEmbodiment]:
        """Yield bodies individually without retaining the whole bank."""
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError("count must be a positive integer")

        for _ in range(count):
            yield self.generate_one(rng)
            