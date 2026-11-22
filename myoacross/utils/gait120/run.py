"""Dispatch the individual-model, KIT walking, and Gait120 reference protocol."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from myoacross.utils.paths import PROJECT_ROOT, resolve_path

STAGES = (
    "prepare",
    "fit",
    "measure",
    "embodiment",
    "retarget",
    "references",
    "evaluate",
    "analysis",
)


def load_pipeline_config(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    from myoacross.utils.config import load_config

    config = load_config(path)
    for field in (
        "dataset_root",
        "kit_dataset_root",
        "smplh_model_dir",
        "output_root",
        "embodiment_config",
        "retarget_config",
    ):
        config[field] = str(resolve_path(config[field], PROJECT_ROOT))
    for name, checkpoint in config["policies"].items():
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("Policy names must be single directory names")
        config["policies"][name] = str(resolve_path(checkpoint, PROJECT_ROOT))
    if not config["policies"]:
        raise ValueError("Select at least one policy")
    for field in (
        "support_dir",
        "model_path",
        "pose_body_prior",
        "pose_hand_prior",
        "head_marker_corr",
    ):
        if config["moshpp"].get(field) is not None:
            config["moshpp"][field] = str(
                resolve_path(config["moshpp"][field], PROJECT_ROOT)
            )
    for group, key in [
        (config["moshpp"], "python"),
        *((config["environments"], name) for name in config["environments"]),
    ]:
        executable = Path(group[key]).expanduser()
        # Do not resolve the Python symlink: that bypasses its virtual environment.
        group[key] = str(
            executable if executable.is_absolute() else PROJECT_ROOT / executable
        )
    return config


def load_records(config):
    return json.loads((Path(config["output_root"]) / "manifest.json").read_text())[
        "subjects"
    ]


def run_prepare(config):
    from .convert import collect_subject_inputs
    from .motions import MOTION_SET, resolved_motions

    root = Path(config["output_root"])
    records = collect_subject_inputs(config["dataset_root"], config["subject_ids"])
    motions = resolved_motions(config)
    root.mkdir(parents=True, exist_ok=True)
    for filename in ("manifest.json", "subjects.json"):
        if (root / filename).exists():
            raise FileExistsError(root / filename)
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                dataset_root=config["dataset_root"],
                subjects=records,
                motion_set=MOTION_SET,
                reference_motions=motions,
            ),
            indent=2,
        )
    )
    (root / "subjects.json").write_text(
        json.dumps(
            dict(subjects={r["subject_id"]: r["metadata"] for r in records}), indent=2
        )
    )


def run_fit(config):
    from .fitting import fit_subject_shape

    for record in load_records(config):
        fit_subject_shape(record, config)


def run_measure(config):
    from .modeling import measure_subjects

    return measure_subjects(config, load_records(config))


def run_embodiment(config):
    from .modeling import build_subject_models

    return build_subject_models(config, load_records(config))


def run_retarget(config):
    from .retarget import retarget_subject_motions

    return retarget_subject_motions(config, load_records(config))


def run_references(config):
    from .references import prepare_reference_profiles

    return prepare_reference_profiles(
        load_records(config), Path(config["output_root"]) / "references"
    )


def evaluation_output(config, scheme, checkpoint):
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    iteration = int(saved["iteration"])
    if iteration != config["checkpoint_iteration"]:
        raise ValueError(
            f"{scheme}: expected iteration {config['checkpoint_iteration']}, found {iteration}"
        )
    return (
        Path(config["output_root"]) / "evaluations" / scheme / f"test_iter{iteration}"
    )


def run_evaluate(config):
    from myoacross.rl.runners.evaluate import evaluate_policy
    from .evaluation import Gait120SignalCollector
    from .motions import build_catalog, select_kit_motions

    catalog = build_catalog(config, load_records(config))
    for scheme, checkpoint in config["policies"].items():
        destination = evaluation_output(config, scheme, checkpoint)
        evaluate_policy(
            checkpoint,
            destination,
            catalog=catalog,
            target_embodiment="segment_scale",
            allow_external_motions=True,
            signal_collector=Gait120SignalCollector(destination / "profiles"),
            envs_per_subject=len(select_kit_motions()),
            device=config["device"],
            seed=config["seed"],
            bundle_root=None,
        )
        (destination / "pipeline_config.json").write_text(json.dumps(config, indent=2))


def run_analysis(config):
    from .analysis import summarize_metrics

    for scheme, checkpoint in config["policies"].items():
        destination = evaluation_output(config, scheme, checkpoint)
        summarize_metrics(
            destination / "profiles",
            Path(config["output_root"]) / "references",
            destination / "analysis",
        )


def dispatch_stage(config, stage, path):
    root = Path(config["output_root"])
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(
            [str(PROJECT_ROOT), environment.get("PYTHONPATH", "")]
        ),
        WARP_CACHE_PATH=str(root / "cache/warp"),
        CUDA_CACHE_PATH=str(root / "cache/cuda"),
        MPLCONFIGDIR=str(root / "cache/matplotlib"),
    )
    python = (
        config["moshpp"]["python"] if stage == "fit" else config["environments"][stage]
    )
    subprocess.run(
        [
            python,
            "-m",
            "myoacross.utils.gait120.run",
            "--config",
            str(path),
            "--stage",
            stage,
        ],
        env=environment,
        check=True,
    )


def run_pipeline(config):
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pipeline_config.json"
    with path.open("x") as file:
        json.dump(config, file, indent=2)
    for stage in STAGES:
        dispatch_stage(config, stage, path)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)
    if args.stage != "all" and Path(args.config).suffix != ".json":
        with TemporaryDirectory(prefix="gait120-stage-") as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config))
            dispatch_stage(config, args.stage, path)
        return
    if args.stage == "evaluate":
        import importlib.util
        from myoacross.rl.runners.multi_embodiment.backend import (
            MARKER,
            run_with_backend,
        )

        spec = importlib.util.find_spec("mujoco_warp")
        if spec is None or spec.origin is None:
            raise RuntimeError(
                "mujoco_warp is not installed in the evaluation environment"
            )
        if MARKER not in (Path(spec.origin).parent / "_src/sensor.py").read_text():
            command = [
                sys.executable,
                "-m",
                "myoacross.utils.gait120.run",
                "--config",
                str(Path(args.config).resolve()),
                "--stage",
                "evaluate",
            ]
            raise SystemExit(
                run_with_backend(command, Path(config["output_root"]) / "cache/backend")
            )
    function = run_pipeline if args.stage == "all" else globals()[f"run_{args.stage}"]
    result = function(config)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
