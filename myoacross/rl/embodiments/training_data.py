"""Associate embodiment bundles with training subjects and motions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from myoacross.rl.embodiments.subjects import (
    DatasetRecord,
    SubjectRecord,
)
from myoacross.rl.embodiments.bundle import EmbodimentBundle
from myoacross.rl.embodiments.features import (
    MorphologyFeatures,
    load_morphology_features,
)


@dataclass(frozen=True, slots=True)
class TrainingSubject:
    """One training subject and its available model variants."""

    subject: SubjectRecord
    variants: tuple[tuple[str, Path], ...]


@dataclass(frozen=True, slots=True)
class RolloutSelection:
    """Source subject and model held fixed across PPO updates."""

    subject: SubjectRecord
    variant_id: str
    bundle_path: Path


class TrainingData:
    """Sample original bodies, with generated variants for virtual training."""

    def __init__(
        self,
        index_paths: Sequence[str | Path],
        subjects: DatasetRecord,
        *,
        embodiment: str,
        morphology_source: str,
        seed: int,
    ) -> None:
        self.catalog_fingerprint = getattr(subjects, "fingerprint", None)
        if subjects.split != "train":
            raise ValueError("RL training requires a training DatasetRecord")

        if embodiment not in (
            "baseline",
            "height_scale",
            "segment_scale",
        ):
            raise ValueError(f"Unknown embodiment: {embodiment!r}")

        if morphology_source not in ("subject", "virtual"):
            raise ValueError(f"Unknown morphology_source: {morphology_source!r}")

        if embodiment == "baseline" and morphology_source != "subject":
            raise ValueError("Baseline requires morphology_source='subject'")

        paths = tuple(sorted(Path(path).expanduser().resolve() for path in index_paths))
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Training index paths must be nonempty and unique")

        # Reuse feature and duplicate-sample validation.
        _, source_ids = load_morphology_features(paths)

        expected_subjects = set(subjects.subject_ids)
        if embodiment == "baseline":
            if len(paths) != 1 or source_ids:
                raise ValueError(
                    "Baseline requires one shared index without source subjects"
                )
        elif set(source_ids) != expected_subjects:
            raise ValueError(
                "Index subjects do not match the training dataset: "
                f"missing={sorted(expected_subjects - set(source_ids))}, "
                f"extra={sorted(set(source_ids) - expected_subjects)}"
            )

        self.embodiment = embodiment
        self.morphology_source = morphology_source

        entries = {}
        index_subjects_seen = set()
        bundle_records = {}
        baseline_path = None
        simulation_config = None
        model_signature = None

        for index_path in paths:
            record = json.loads(index_path.read_text(encoding="utf-8"))

            record_source = record["morphology_source"]
            allowed_sources = (
                ("subject", "virtual")
                if morphology_source == "virtual"
                else ("subject",)
            )
            if record_source not in allowed_sources:
                raise ValueError(f"Unexpected morphology source: {index_path}")

            expected = {"embodiment": embodiment}
            for name, value in expected.items():
                if record.get(name) != value:
                    raise ValueError(
                        f"{index_path}: expected {name}={value!r}, "
                        f"got {record.get(name)!r}"
                    )

            subject_id = record["source_subject_id"]
            index_subjects = (
                tuple(subjects.get(sid) for sid in sorted(expected_subjects))
                if embodiment == "baseline"
                else (subjects.get(subject_id),)
            )

            source_directory = Path(record["source_directory"]).expanduser()
            if not source_directory.is_absolute():
                source_directory = index_path.parent / source_directory

            for subject in index_subjects:
                identity = (subject.subject_id, record_source)
                if identity in index_subjects_seen:
                    raise ValueError(f"Duplicate subject index: {subject.subject_id}")
                index_subjects_seen.add(identity)
                if source_directory.resolve() != subject.xml_path.parent:
                    raise ValueError(
                        f"Index source differs from the subject model: {index_path}"
                    )

            current_baseline = Path(record["baseline_path"]).expanduser()
            if not current_baseline.is_absolute():
                current_baseline = index_path.parent / current_baseline
            current_baseline = current_baseline.resolve()

            if not current_baseline.is_file():
                raise FileNotFoundError(current_baseline)

            current_simulation = record["simulation_config"]
            current_signature = record["model_signature"]

            if (
                not isinstance(current_simulation, dict)
                or not isinstance(current_signature, str)
                or not current_signature.strip()
            ):
                raise ValueError(f"Invalid simulation metadata: {index_path}")

            if baseline_path is None:
                baseline_path = current_baseline
                simulation_config = current_simulation
                model_signature = current_signature
            else:
                if current_baseline != baseline_path:
                    raise ValueError(f"Incompatible baseline: {index_path}")
                if current_simulation != simulation_config:
                    raise ValueError(
                        f"Incompatible simulation configuration: {index_path}"
                    )
                if current_signature != model_signature:
                    raise ValueError(f"Incompatible model signature: {index_path}")

            samples = record["samples"]
            if (
                not samples
                or record["count"] != len(samples)
                or (record_source == "subject" and len(samples) != 1)
            ):
                raise ValueError(f"Invalid sample count: {index_path}")

            variants = []
            variant_ids = set()

            for sample in samples:
                variant_id = sample["variant_id"]
                if (
                    not isinstance(variant_id, str)
                    or not variant_id.strip()
                    or variant_id in variant_ids
                ):
                    raise ValueError(f"Invalid or duplicate variant ID: {index_path}")
                variant_ids.add(variant_id)

                bundle_path = (index_path.parent / sample["bundle"]).resolve()
                if not bundle_path.is_file():
                    raise FileNotFoundError(bundle_path)
                if bundle_path in bundle_records:
                    raise ValueError(f"Duplicate bundle path: {bundle_path}")

                features = MorphologyFeatures.from_dict(sample["features"])
                height = float(sample["subject_height_m"])
                if not np.isfinite(height) or height <= 0.0:
                    raise ValueError(f"Invalid subject height: {index_path}")

                bundle_records[bundle_path] = (
                    subject_id,
                    variant_id,
                    features,
                    height,
                    record_source,
                )
                variants.append((variant_id, bundle_path))

            for subject in index_subjects:
                if subject.subject_id in entries:
                    variants_for_subject = (
                        list(entries[subject.subject_id].variants) + variants
                    )
                else:
                    variants_for_subject = variants
                entries[subject.subject_id] = TrainingSubject(
                    subject=subject,
                    variants=tuple(sorted(variants_for_subject)),
                )

        self.baseline_path = baseline_path
        self.simulation_config = simulation_config
        self.model_signature = model_signature
        self.subjects = tuple(entries[subject_id] for subject_id in sorted(entries))
        self._bundle_records = bundle_records
        self._subject_entries = entries
        self.rng = np.random.default_rng(seed)

    def sample(self) -> RolloutSelection:
        """Sample a subject and model; commands sample its motions per env."""
        entry = self.subjects[int(self.rng.integers(len(self.subjects)))]
        variant_id, bundle_path = entry.variants[
            int(self.rng.integers(len(entry.variants)))
        ]

        return RolloutSelection(
            subject=entry.subject,
            variant_id=variant_id,
            bundle_path=bundle_path,
        )

    def load_bundle(
        self,
        selection: RolloutSelection,
    ) -> EmbodimentBundle:
        """Check the selected bundle against its index and configuration."""
        path = Path(selection.bundle_path).expanduser().resolve()

        if path not in self._bundle_records:
            raise ValueError(f"Bundle is not registered in this training bank: {path}")

        subject_id, variant_id, features, height, morphology_source = (
            self._bundle_records[path]
        )

        entry = self._subject_entries.get(selection.subject.subject_id)
        if (
            entry is None
            or selection.subject != entry.subject
            or (selection.variant_id, path) not in entry.variants
        ):
            raise ValueError(f"Selection differs from the registered bundle: {path}")

        bundle = EmbodimentBundle.load(path)

        expected = {
            "source_subject_id": subject_id,
            "variant_id": variant_id,
            "embodiment": self.embodiment,
            "morphology_source": morphology_source,
            "model_signature": self.model_signature,
        }
        for name, value in expected.items():
            if getattr(bundle, name) != value:
                raise ValueError(
                    f"{path}: bundle {name} differs from training metadata"
                )

        # Both records originate from the same offline measurements.
        if bundle.features != features:
            raise ValueError(f"Bundle features differ from the index: {path}")

        if bundle.subject_height_m != height:
            raise ValueError(f"Bundle height differs from the index: {path}")

        return bundle
