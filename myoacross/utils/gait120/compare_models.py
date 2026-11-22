"""Construct, retarget and evaluate the three Gait120 model conditions."""

import argparse
import json
from pathlib import Path
from myoacross.utils.paths import PROJECT_ROOT


def prepare(root, gait_root, checkpoint):
    from musclemimic_models import get_xml_path
    from myoacross.embodiment.generate import generate
    from myoacross.embodiment.export import save
    from myoacross.embodiment.config import load_config
    from .motions import resolved_motions

    root.mkdir(parents=True, exist_ok=False)
    subjects = json.loads((gait_root / "manifest.json").read_text())["subjects"]
    models = []
    for condition, sid in [("template", None)] + [
        ("height_scale", r["subject_id"]) for r in subjects
    ]:
        options = load_config(
            PROJECT_ROOT / "myoacross/configs/embodiment/segment_scale.yaml"
        )["optimization"].copy()
        for key in (
            "sigma_symmetry",
            "sigma_adjacency",
            "max_normalized_measurement_error",
        ):
            options.pop(key, None)
        spec, model, report = generate(
            "baseline" if condition == "template" else condition,
            sid,
            baseline_path=Path(get_xml_path("myofullbody")),
            measurements_dir=gait_root / "embodiments/measurements",
            optimization_options=None if condition == "template" else options,
        )
        destination = save(spec, report, output_root=root / "embodiments")
        models.append(
            dict(
                subject=sid or "template",
                condition=condition,
                model=str(destination),
                mass_actual=float(model.body_mass.sum()),
            )
        )
    manifest = dict(
        models=models,
        subjects=subjects,
        source_motions=resolved_motions(
            dict(kit_dataset_root=str(PROJECT_ROOT / "data/dataset"))
        ),
        checkpoint=str(checkpoint),
        existing_regional=str(gait_root),
        seed=42,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def retarget(root, condition):
    from .retarget import fitted_shape_cache
    from .motions import MOTION_SET
    from myoacross.retarget.run import run_motion

    manifest = json.loads((root / "manifest.json").read_text())
    with fitted_shape_cache(root / "cache/fitted_shapes"):
        for entry in manifest["models"]:
            if entry["condition"] != condition:
                continue
            for motion in manifest["source_motions"]:
                run_motion(
                    motion["path"],
                    entry["model"],
                    PROJECT_ROOT / "myoacross/configs/retarget/gmr_fit.yaml",
                    root
                    / "retargets"
                    / condition
                    / entry["subject"]
                    / motion["motion_id"],
                    smplh_model_dir=PROJECT_ROOT / "data/models/smplh",
                    identity=dict(
                        target_subject_id=entry["subject"],
                        source_subject_id=motion["source_subject_id"],
                        motion_id=motion["motion_id"],
                        category=motion["category"],
                        motion_set=MOTION_SET,
                        embodiment="baseline"
                        if condition == "template"
                        else "height_scale",
                        target_split="test",
                    ),
                )


def evaluate(
    root,
    gait_root,
    condition,
    checkpoint=None,
    output=None,
    subject_ids=None,
    motion_ids=None,
):
    from myoacross.rl.runners.evaluate import evaluate_policy
    from myoacross.rl.embodiments.subjects import (
        DatasetRecord,
        SubjectRecord,
        MotionRecord,
    )
    from .evaluation import Gait120SignalCollector
    from .motions import build_catalog

    manifest = json.loads((root / "manifest.json").read_text())
    if condition == "regional_scale":
        records = manifest["subjects"]
        if subject_ids:
            records = [r for r in records if r["subject_id"] in subject_ids]
        catalog = build_catalog(
            dict(
                output_root=str(gait_root),
                kit_dataset_root=str(PROJECT_ROOT / "data/dataset"),
            ),
            records,
        )
    else:
        subjects = []
        for entry in manifest["models"]:
            if entry["condition"] != condition or (
                subject_ids and entry["subject"] not in subject_ids
            ):
                continue
            motions = []
            for source in manifest["source_motions"]:
                if motion_ids and source["motion_id"] not in motion_ids:
                    continue
                folder = (
                    root
                    / "retargets"
                    / condition
                    / entry["subject"]
                    / source["motion_id"]
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
            subjects.append(
                SubjectRecord(
                    "shared" if condition == "template" else entry["subject"],
                    "test",
                    Path(entry["model"]),
                    tuple(motions),
                )
            )
        catalog = DatasetRecord(subjects)
    if motion_ids and condition == "regional_scale":
        catalog = DatasetRecord(
            [
                SubjectRecord(
                    s.subject_id,
                    s.split,
                    s.xml_path,
                    tuple(m for m in s.motions if m.motion_id in motion_ids),
                )
                for s in catalog
            ]
        )
    destination = output or root / "evaluations" / condition
    return evaluate_policy(
        checkpoint or manifest["checkpoint"],
        destination,
        catalog=catalog,
        target_embodiment={
            "template": "baseline",
            "height_scale": "height_scale",
            "regional_scale": "segment_scale",
        }[condition],
        allow_external_motions=True,
        signal_collector=Gait120SignalCollector(destination / "profiles"),
        envs_per_subject=8,
        device="cuda:0",
        seed=42,
        bundle_root=None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "retarget", "evaluate"])
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data/experiments/gait120/model_comparison",
    )
    parser.add_argument(
        "--gait-root", type=Path, default=PROJECT_ROOT / "data/experiments/gait120"
    )
    parser.add_argument(
        "--condition",
        choices=["template", "height_scale", "regional_scale"],
        default="height_scale",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--motions", nargs="+")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(
            args.root.resolve(),
            args.gait_root.resolve(),
            (
                args.checkpoint
                or PROJECT_ROOT / "data/myoacross/rl/checkpoint/final.pt"
            ).resolve(),
        )
    elif args.stage == "retarget":
        retarget(args.root.resolve(), args.condition)
    else:
        import sys
        import importlib.util
        from myoacross.rl.runners.multi_embodiment.backend import (
            MARKER,
            run_with_backend,
        )

        spec = importlib.util.find_spec("mujoco_warp")
        if MARKER not in (Path(spec.origin).parent / "_src/sensor.py").read_text():
            raise SystemExit(
                run_with_backend(
                    [
                        sys.executable,
                        "-m",
                        "myoacross.utils.gait120.compare_models",
                        *sys.argv[1:],
                    ],
                    args.root / "cache/backend",
                )
            )
        evaluate(
            args.root.resolve(),
            args.gait_root.resolve(),
            args.condition,
            args.checkpoint,
            args.output,
            args.subjects,
            args.motions,
        )


if __name__ == "__main__":
    main()
