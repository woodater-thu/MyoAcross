"""Dispatch image reconstruction, body construction, retargeting, and evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from myoacross.utils.config import load_config
from myoacross.utils.paths import PROJECT_ROOT, DATASET_ROOT, resolve_path


def run_image2msksim(config: dict) -> Path:
    """Run each stage in its own environment and preserve one experiment output."""
    subject = config["subject_id"]
    if (
        not isinstance(subject, str)
        or subject in ("", ".", "..")
        or "/" in subject
        or "\\" in subject
    ):
        raise ValueError("subject_id must be one directory name")
    mass = float(config["body_mass_kg"])
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("body_mass_kg must be positive")
    height = float(config["height_m"])
    if not math.isfinite(height) or height <= 0:
        raise ValueError("height_m must be positive")
    root = resolve_path(config["output_root"], PROJECT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf

    resolved = root / "config.yaml"
    with resolved.open("x") as stream:
        OmegaConf.save(OmegaConf.create(config), stream)
    environment = os.environ.copy()
    cache = resolve_path(config["cache_root"], PROJECT_ROOT)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        WARP_CACHE_PATH=str(cache / "warp"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
        MPLCONFIGDIR=str(cache / "matplotlib"),
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT)]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    commands = [
        [
            str(PROJECT_ROOT / Path(config["hsmr"]["python"]).expanduser()),
            "-m",
            "myoacross.utils.image2msksim.reconstruct",
            "--config",
            str(resolved),
        ],
        [
            sys.executable,
            "-m",
            "myoacross.utils.image2msksim.run",
            "--config",
            str(resolved),
            "--stage",
            "embodiment",
        ],
        [
            str(PROJECT_ROOT / Path(config["retarget_python"]).expanduser()),
            "-m",
            "myoacross.utils.image2msksim.run",
            "--config",
            str(resolved),
            "--stage",
            "retarget",
        ],
        [
            sys.executable,
            "-m",
            "myoacross.utils.image2msksim.run",
            "--config",
            str(resolved),
            "--stage",
            "evaluate",
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    return root


def run_dataset(config: dict) -> Path:
    """Run the subject manifest and report successes and failed stages."""
    import torch
    from omegaconf import OmegaConf

    manifest = resolve_path(config["subjects_path"], PROJECT_ROOT)
    inputs = json.loads(manifest.read_text())
    subjects = inputs["subjects"]
    identifiers = [subject["subject_id"] for subject in subjects]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate subject IDs in the experiment manifest")
    checkpoint = torch.load(
        resolve_path(config["checkpoint_path"], PROJECT_ROOT),
        map_location="cpu", weights_only=True,
    )
    iteration = checkpoint["iteration"]
    del checkpoint
    root = resolve_path(config["output_root"], PROJECT_ROOT)
    destination = root / "evaluations" / config["policy_scheme"] / f"test_iter{iteration}"
    destination.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config), destination / "config.yaml")
    report = dict(status="running", checkpoint_iteration=iteration,
                  input_metadata={key: value for key, value in inputs.items() if key != "subjects"},
                  num_requested_subjects=len(subjects), subjects=[])
    for subject in subjects:
        settings = deepcopy(config)
        settings.pop("subjects_path")
        settings.update(subject)
        settings.update(subject_id=subject["subject_id"],
                        image_path=subject["image_path"], video_path=None,
                        video_frame_index=None, body_mass_kg=subject["body_mass_kg"],
                        output_root=str(Path(config["output_root"]) / "subjects" / subject["subject_id"]))
        record = dict(subject_id=subject["subject_id"],
                      source_subject_id=subject["source_subject_id"],
                      source_split=subject["source_split"])
        try:
            output = run_image2msksim(settings)
            summary = output / "evaluations" / config["policy_scheme"] / f"test_iter{iteration}" / "summary.json"
            record.update(status="finished", metrics=json.loads(summary.read_text()))
        except (subprocess.CalledProcessError, ValueError, OSError) as error:
            stage = "preparation"
            if isinstance(error, subprocess.CalledProcessError):
                command = error.cmd
                stage = command[command.index("--stage") + 1] if "--stage" in command else "reconstruction"
            record.update(status="failed", failed_stage=stage, error=str(error))
            print(f"{subject['subject_id']}: {stage} failed: {error}", file=sys.stderr)
        report["subjects"].append(record)
        (destination / "dataset_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    completed = [record for record in report["subjects"] if record["status"] == "finished"]
    report.update(num_completed_subjects=len(completed),
                  num_failed_subjects=len(subjects) - len(completed),
                  status="finished" if len(completed) == len(subjects) else "finished_with_failures",
                  by_source_split={})
    for split in [None] + sorted({subject["source_split"] for subject in subjects}):
        selected = [record["metrics"] for record in completed
                    if split is None or record["source_split"] == split]
        group = dict(num_completed_subjects=len(selected),
                     num_requested_subjects=sum(split is None or subject["source_split"] == split
                                                for subject in subjects))
        if selected:
            group.update({key: sum(metrics[key] for metrics in selected) / len(selected)
                          for key in selected[0] if key not in ("num_subjects", "num_trials")})
            group["num_trials"] = sum(metrics["num_trials"] for metrics in selected)
        if split is None:
            report["overall"] = group
        else:
            report["by_source_split"][split] = group
    (destination / "dataset_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return destination


def build_embodiment(config: dict) -> Path:
    """Construct the image-derived body with the existing segment-scale model."""
    from musclemimic_models import get_xml_path
    from myoacross.embodiment.config import load_config as load_embodiment_config
    from myoacross.embodiment.generate import generate
    from myoacross.embodiment.export import save

    root = resolve_path(config["output_root"], PROJECT_ROOT)
    options = load_embodiment_config(
        resolve_path(config["embodiment_config"], PROJECT_ROOT)
    )
    spec, _, report = generate(
        "segment_scale",
        config["subject_id"],
        baseline_path=Path(get_xml_path("myofullbody")),
        measurements_dir=root / "embodiments/measurements",
        optimization_options=options.get("optimization"),
    )
    return save(spec, report, output_root=root / "embodiments")


def _motion_specs(config):
    """Use the checkpoint's motion selection for every pipeline stage."""
    import torch
    from myoacross.utils.datalists import get_motion_specs

    checkpoint = torch.load(
        resolve_path(config["checkpoint_path"], PROJECT_ROOT),
        map_location="cpu",
        weights_only=True,
    )
    data = checkpoint["config"]["data"]
    specs = get_motion_specs(
        motion_set=data.get("motion_set", "kit_flat_locomotion_72")
    )
    allowed = data.get("motion_ids")
    requested = config.get("motion_ids")
    if requested is not None and (
        not requested or (allowed is not None and not set(requested) <= set(allowed))
    ):
        raise ValueError("Selected motions are outside the checkpoint motion selection")
    ids = requested if requested is not None else allowed
    if ids is not None:
        if set(ids) - {spec.motion_id for spec in specs}:
            raise ValueError("Unknown motion IDs")
        specs = tuple(spec for spec in specs if spec.motion_id in ids)
    return specs, data.get("motion_set", "kit_flat_locomotion_72")


def retarget_subject(config: dict) -> None:
    """Fit the standard source motions to the reconstructed body."""
    from myoacross.retarget.run import run_motion

    root = resolve_path(config["output_root"], PROJECT_ROOT)
    subject = config["subject_id"]
    model = root / "embodiments/segment_scale" / subject / "model.xml"
    report = json.loads(model.with_name("report.json").read_text())
    if report["status"] != "passed":
        raise ValueError("The reconstructed embodiment did not pass export checks")
    specs, motion_set = _motion_specs(config)
    for spec in specs:
        run_motion(
            DATASET_ROOT / spec.source_path,
            model,
            resolve_path(config["retarget_config"], PROJECT_ROOT),
            root / "retargets/segment_scale" / subject / spec.motion_id,
            identity=dict(
                target_subject_id=subject,
                source_subject_id=spec.source_subject_id,
                motion_id=spec.motion_id,
                category=spec.category,
                motion_set=motion_set,
                embodiment="segment_scale",
                target_split="test",
            ),
        )


def evaluate_subject(config: dict) -> Path:
    """Pass an explicit image-derived catalog to the standard parallel evaluator."""
    import torch
    from myoacross.rl.embodiments.subjects import (
        DatasetRecord,
        SubjectRecord,
        MotionRecord,
    )
    from myoacross.rl.runners.evaluate import evaluate_policy

    root = resolve_path(config["output_root"], PROJECT_ROOT)
    subject = config["subject_id"]
    specs, _ = _motion_specs(config)
    motions = []
    for spec in specs:
        folder = root / "retargets/segment_scale" / subject / spec.motion_id
        report = json.loads((folder / "report.json").read_text())
        if report["quality"]["status"] != "passed":
            raise ValueError(f"Retarget quality not accepted: {folder}")
        motions.append(
            MotionRecord(
                spec.motion_id,
                DATASET_ROOT / spec.source_path,
                folder / "motion.npz",
                spec.source_subject_id,
                spec.category,
                folder / "report.json",
            )
        )
    catalog = DatasetRecord(
        [
            SubjectRecord(
                subject,
                "test",
                root / "embodiments/segment_scale" / subject / "model.xml",
                tuple(motions),
            )
        ]
    )
    checkpoint_path = resolve_path(config["checkpoint_path"], PROJECT_ROOT)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    iteration = checkpoint["iteration"]
    del checkpoint
    scheme = config["policy_scheme"]
    if (
        not isinstance(scheme, str)
        or scheme in ("", ".", "..")
        or "/" in scheme
        or "\\" in scheme
    ):
        raise ValueError("policy_scheme must be one directory name")
    return evaluate_policy(
        checkpoint_path,
        root / "evaluations" / scheme / f"test_iter{iteration}",
        target_embodiment="segment_scale",
        catalog=catalog,
        motion_ids=[spec.motion_id for spec in specs],
        envs_per_subject=config["envs_per_subject"],
        device=config["device"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "embodiment", "retarget", "evaluate"), default="all"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.stage == "evaluate":
        # Import the evaluator only in the child with the adapted backend.
        import importlib.util
        from myoacross.rl.runners.multi_embodiment.backend import (
            MARKER,
            run_with_backend,
        )

        spec = importlib.util.find_spec("mujoco_warp")
        if spec is None or spec.origin is None:
            raise RuntimeError("mujoco_warp is not installed")
        if MARKER not in (Path(spec.origin).parent / "_src/sensor.py").read_text():
            command = [
                sys.executable,
                "-m",
                "myoacross.utils.image2msksim.run",
                "--config",
                str(args.config.resolve()),
                "--stage",
                "evaluate",
            ]
            cache = resolve_path(config["cache_root"], PROJECT_ROOT)
            raise SystemExit(run_with_backend(command, cache / "backend"))
    functions = dict(
        all=run_dataset if config.get("subjects_path") else run_image2msksim,
        embodiment=build_embodiment,
        retarget=retarget_subject,
        evaluate=evaluate_subject,
    )
    output = functions[args.stage](config)
    print(output)
    if args.stage == "all" and config.get("subjects_path"):
        if json.loads((output / "dataset_summary.json").read_text())["num_failed_subjects"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
