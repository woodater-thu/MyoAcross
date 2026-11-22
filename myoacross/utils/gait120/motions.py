"""Select registered KIT walking references and build evaluation catalogs."""

import json
from pathlib import Path
from myoacross.utils.datalists import get_motion_specs

MOTION_SET = "kit_flat_locomotion_72"


def select_kit_motions():
    """Select the eight registered medium walking clips without duplicating IDs."""
    motions = tuple(
        m
        for m in get_motion_specs(motion_set=MOTION_SET)
        if m.category == "walk_medium"
    )
    if len(motions) != 8:
        raise ValueError(
            "The Gait120 protocol requires eight registered walk_medium clips"
        )
    return motions


def resolved_motions(config):
    root = Path(config["kit_dataset_root"])
    motions = [
        dict(
            motion_id=m.motion_id,
            path=str(root / m.source_path),
            source_subject_id=m.source_subject_id,
            category=m.category,
        )
        for m in select_kit_motions()
    ]
    for motion in motions:
        if not Path(motion["path"]).is_file():
            raise FileNotFoundError(motion["path"])
    return motions


def build_catalog(config, records):
    from myoacross.rl.embodiments.subjects import (
        DatasetRecord,
        SubjectRecord,
        MotionRecord,
    )

    root = Path(config["output_root"])
    sources = resolved_motions(config)
    subjects = []
    for record in records:
        sid = record["subject_id"]
        model = root / "embodiments/segment_scale" / sid / "model.xml"
        if json.loads(model.with_name("report.json").read_text())["status"] != "passed":
            raise ValueError(f"Subject model did not pass checks: {sid}")
        motions = []
        for source in sources:
            folder = root / "retargets/segment_scale" / sid / source["motion_id"]
            report = json.loads((folder / "report.json").read_text())
            if report["quality"]["status"] != "passed":
                raise ValueError(f"Retarget did not pass checks: {folder}")
            if "gait120_initialization" in report["metadata"]:
                raise ValueError(
                    f"Expected original KIT retarget, found extra initialization: {folder}"
                )
            motions.append(
                MotionRecord(
                    source["motion_id"],
                    Path(source["path"]),
                    folder / "motion.npz",
                    source["source_subject_id"],
                    source["category"],
                    folder / "report.json",
                )
            )
        subjects.append(SubjectRecord(sid, "test", model, tuple(motions)))
    return DatasetRecord(subjects)
