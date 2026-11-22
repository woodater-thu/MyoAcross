"""Discovery and selection of subject-specific MyoFullBody XML files."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from types import MappingProxyType

from myoacross.utils.datalists import get_target_subject_ids, get_motion_specs
from myoacross.utils.paths import (
    DATASET_ROOT,
    EMBODIMENTS_DIR,
    RETARGETS_DIR,
)


@dataclass(frozen=True, slots=True)
class MotionRecord:
    """One reference motion and its source metadata."""

    motion_id: str
    source_path: Path
    retarget_path: Path
    source_subject_id: str | None = None
    category: str | None = None
    report_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.motion_id.strip():
            raise ValueError("motion_id must not be empty")

        for name in (
            "source_path",
            "retarget_path",
        ):
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"{name} must not be None")

            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name}: {path}")

            object.__setattr__(self, name, path)


@dataclass(frozen=True, slots=True)
class SubjectRecord:
    """One subject model and its registered reference motions."""

    subject_id: str
    split: str
    xml_path: Path
    motions: tuple[MotionRecord, ...]

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must not be empty")
        if self.split not in ("train", "test"):
            raise ValueError(f"Unknown split: {self.split!r}")

        path = Path(self.xml_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Subject XML does not exist: {path}")

        motions = tuple(self.motions)
        if not motions:
            raise ValueError(f"{self.subject_id}: no motions")

        motion_ids = [motion.motion_id for motion in motions]
        if len(set(motion_ids)) != len(motion_ids):
            raise ValueError(f"{self.subject_id}: duplicate motion IDs")

        object.__setattr__(self, "xml_path", path)
        object.__setattr__(self, "motions", motions)


def _load_target_model(embodiments_root, embodiment, subject_id, expected_subject):
    """Check target identity and acceptance before registering its XML."""
    model_dir = embodiments_root / embodiment
    if expected_subject is not None:
        model_dir /= subject_id
    report_path = model_dir / "report.json"
    report = json.loads(report_path.read_text())
    if (
        report.get("strategy") != embodiment
        or "subject_id" not in report
        or report["subject_id"] != expected_subject
    ):
        raise ValueError(f"Embodiment identity mismatch: {report_path}")
    if report.get("status") != "passed":
        raise ValueError(f"Embodiment not accepted: {report_path}")
    model_path = model_dir / "model.xml"
    return model_path


def _validate_motion_report(
    motion_report,
    motion_report_path,
    spec,
    *,
    expected_subject,
    model_path,
    dataset_root,
    motion_set,
    embodiment,
    split,
    require_quality_pass,
):
    """Check motion identity, paths, and recorded screening status."""
    metadata = motion_report.get("metadata", {})
    expected = dict(
        target_subject_id=expected_subject,
        source_subject_id=spec.source_subject_id,
        motion_id=spec.motion_id,
        category=spec.category,
        motion_set=motion_set,
        embodiment=embodiment,
        target_split=split if expected_subject is not None else None,
    )
    if metadata.get("identity") != expected:
        raise ValueError(f"Retarget identity mismatch: {motion_report_path}")
    if (
        Path(metadata.get("robot_xml_path", "")).resolve() != model_path.resolve()
        or Path(metadata.get("motion_path", "")).resolve()
        != (dataset_root / spec.source_path).resolve()
    ):
        raise ValueError(
            f"Retarget model/source identity mismatch: {motion_report_path}"
        )
    if (
        require_quality_pass
        and motion_report.get("quality", {}).get("status") != "passed"
    ):
        raise ValueError(f"Retarget quality not accepted: {motion_report_path}")


class DatasetRecord:
    """A deterministically ordered collection of subjects from one split."""

    def __init__(self, subjects: Iterable[SubjectRecord]) -> None:
        records = {}

        for subject in subjects:
            if subject.subject_id in records:
                raise ValueError(f"Duplicate subject_id: {subject.subject_id!r}")
            records[subject.subject_id] = subject

        if not records:
            raise ValueError("Dataset contains no subjects")

        splits = {subject.split for subject in records.values()}
        if len(splits) != 1:
            raise ValueError("One DatasetRecord must contain only one split")

        self._split = next(iter(splits))
        self._records = MappingProxyType(dict(sorted(records.items())))

    @classmethod
    def from_split(
        cls,
        *,
        split: str,
        embodiment: str,
        motion_set: str = "kit_flat_locomotion_72",
        subject_ids: Iterable[str] | None = None,
        motion_ids: Iterable[str] | None = None,
        require_quality_pass: bool = True,
        dataset_root: str | Path = DATASET_ROOT,
        embodiments_root: str | Path = EMBODIMENTS_DIR,
        retargets_root: str | Path = RETARGETS_DIR,
    ) -> DatasetRecord:
        """Load explicit target × motion pairs; reject partial or mismatched data.

        Virtual variants use their source target body's reference group. The
        source actor recorded on a motion has no role in selecting the body.
        """
        if embodiment not in ("baseline", "height_scale", "segment_scale"):
            raise ValueError(f"Unknown embodiment: {embodiment!r}")
        if type(require_quality_pass) is not bool:
            raise ValueError("require_quality_pass must be a boolean")
        if (
            not motion_set
            or Path(motion_set).name != motion_set
            or motion_set in (".", "..")
        ):
            raise ValueError("Invalid motion_set")
        targets = select_target_ids(split, embodiment, subject_ids)
        specs = get_motion_specs(motion_set=motion_set)
        if motion_ids is not None:
            requested = set(motion_ids)
            if not requested or requested - {m.motion_id for m in specs}:
                raise ValueError("Unknown or empty motion selection")
            specs = tuple(m for m in specs if m.motion_id in requested)
        dataset_root = Path(dataset_root).expanduser().resolve()
        embodiments_root = Path(embodiments_root).expanduser().resolve()
        retargets_root = Path(retargets_root).expanduser().resolve()
        subjects = []
        digest = hashlib.sha256()
        for subject_id in targets:
            expected_subject = None if embodiment == "baseline" else subject_id
            model_path = _load_target_model(
                embodiments_root, embodiment, subject_id, expected_subject
            )
            motions = []
            for spec in specs:
                model_output = retargets_root / embodiment
                if embodiment != "baseline":
                    model_output /= subject_id
                folder = model_output / spec.motion_id
                motion_report_path = folder / "report.json"
                raw_report = motion_report_path.read_bytes()
                motion_report = json.loads(raw_report)
                _validate_motion_report(
                    motion_report,
                    motion_report_path,
                    spec,
                    expected_subject=expected_subject,
                    model_path=model_path,
                    dataset_root=dataset_root,
                    motion_set=motion_set,
                    embodiment=embodiment,
                    split=split,
                    require_quality_pass=require_quality_pass,
                )
                digest.update(subject_id.encode())
                digest.update(spec.motion_id.encode())
                digest.update(raw_report)
                motions.append(
                    MotionRecord(
                        motion_id=spec.motion_id,
                        source_path=dataset_root / spec.source_path,
                        retarget_path=folder / "motion.npz",
                        source_subject_id=spec.source_subject_id,
                        category=spec.category,
                        report_path=motion_report_path,
                    )
                )
            digest.update(model_path.read_bytes())
            subjects.append(
                SubjectRecord(subject_id, split, model_path, tuple(motions))
            )
        catalog = cls(subjects)
        catalog.fingerprint = digest.hexdigest()
        catalog.motion_set = motion_set
        return catalog

    @property
    def split(self) -> str:
        return self._split

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    @property
    def num_motions(self) -> int:
        return sum(len(subject.motions) for subject in self._records.values())

    def get(self, subject_id: str) -> SubjectRecord:
        try:
            return self._records[subject_id]
        except KeyError as error:
            raise KeyError(
                f"Subject {subject_id!r} is not in this {self.split} dataset"
            ) from error

    def select(self, subject_ids: Iterable[str]) -> DatasetRecord:
        """Create a subset while preserving records and split."""
        return type(self)(self.get(subject_id) for subject_id in subject_ids)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[SubjectRecord]:
        return iter(self._records.values())

    def __getitem__(self, subject_id: str) -> SubjectRecord:
        return self.get(subject_id)


def select_target_ids(split: str, embodiment: str, subject_ids=None) -> tuple[str, ...]:
    """Validate explicit target selection before touching model/motion files."""
    registered = get_target_subject_ids(split=split, embodiment=embodiment)
    allowed = ("shared",) if embodiment == "baseline" else registered
    if subject_ids is None:
        return tuple(allowed)
    if isinstance(subject_ids, str):
        raise ValueError("subject_ids must be a list, not a string")
    selected = tuple(subject_ids)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(allowed)
    ):
        raise ValueError(f"Invalid target subjects for {split}: {selected}")
    return tuple(sorted(selected))
