"""Inventory retarget outputs and summarize recorded numerical screening."""

import json
from pathlib import Path


def build_dataset_manifest(
    *, embodiment, motion_set="kit_flat_locomotion_72", retargets_root=None
):
    """Inventory the full requested product without silently dropping failed pairs."""
    from myoacross.utils.datalists import get_target_subject_ids, get_motion_specs
    from myoacross.utils.paths import RETARGETS_DIR

    root = (
        Path(retargets_root)
        if retargets_root is not None
        else RETARGETS_DIR
    )
    records = []
    ready = {"train": [], "test": []}
    motions = get_motion_specs(motion_set=motion_set)
    for split in ("train", "test"):
        targets = (
            ("shared",)
            if embodiment == "baseline"
            else get_target_subject_ids(split=split, embodiment=embodiment)
        )
        if embodiment == "baseline" and split == "test":
            continue
        for subject in targets:
            passed = 0
            for motion in motions:
                model_output = root / embodiment
                if embodiment != "baseline":
                    model_output /= subject
                folder = model_output / motion.motion_id
                status = "missing"
                reasons = []
                if (folder / "report.json").is_file() and (
                    folder / "motion.npz"
                ).is_file():
                    report = json.loads((folder / "report.json").read_text())
                    quality = report.get("quality", {})
                    status = quality.get("status", "unscreened")
                    reasons = quality.get("reasons", [])
                passed += status == "passed"
                records.append(
                    dict(
                        target_split=split,
                        target_subject_id=subject,
                        motion_id=motion.motion_id,
                        category=motion.category,
                        status=status,
                        reasons=reasons,
                    )
                )
            if passed == len(motions):
                ready[split].append(subject)
    return dict(
        embodiment=embodiment,
        motion_set=motion_set,
        ready_subject_ids=ready,
        records=records,
        note="Readiness is the numerical screen only; RL loading also validates identities and paths.",
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embodiment",
        required=True,
        choices=["baseline", "height_scale", "segment_scale"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_dataset_manifest(embodiment=args.embodiment), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
