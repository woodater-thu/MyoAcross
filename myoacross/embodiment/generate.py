"""Generate and validate MyoFullBody embodiments."""

from pathlib import Path
import mujoco
import argparse


from musclemimic_models import get_xml_path

from myoacross.embodiment.baseline.model import BaselineModel
from myoacross.embodiment.height_scale.model import HeightScaleModel
from myoacross.embodiment.segment_scale.model import SegmentScaleModel
from myoacross.embodiment.model import OptimizationFailure
from myoacross.embodiment.config import load_config
from myoacross.embodiment.export import save, save_failure_report

from myoacross.utils.paths import MEASUREMENTS_DIR, EMBODIMENTS_DIR
from myoacross.utils.datalists import get_target_subject_ids


def generate(
    strategy: str,
    subject_id: str | None = None,
    *,
    baseline_path: Path,
    measurements_dir: str | Path = MEASUREMENTS_DIR,
    optimization_options: dict | None = None,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, dict]:
    """Return the generated spec, compiled model and validation report."""

    # Validate strategy
    if strategy not in {"baseline", "height_scale", "segment_scale"}:
        raise ValueError(f"Unknown embodiment strategy: {strategy!r}")

    # Validate selected subjects
    if strategy == "baseline":
        if subject_id is not None:
            raise ValueError("Baseline does not use a subject ID")
    else:
        if subject_id is None or not subject_id.strip():
            raise ValueError(f"{strategy} requires a nonempty subject ID")
        subject_id = subject_id.strip()

    if optimization_options is not None and strategy == "baseline":
        raise ValueError(
            "Optimization options are only supported by height_scale and segment_scale"
        )

    # Load baseline myofullbody
    baseline_spec = mujoco.MjSpec.from_file(str(baseline_path))

    # Select the scaling strategy
    if strategy == "baseline":
        embodiment = BaselineModel(baseline_spec)
    elif strategy == "height_scale":
        embodiment = HeightScaleModel(
            baseline_spec, subject_id, measurements_dir=measurements_dir
        )
    elif strategy == "segment_scale":
        embodiment = SegmentScaleModel(
            baseline_spec, subject_id, measurements_dir=measurements_dir
        )

    # Solve body scales
    if strategy in {"height_scale", "segment_scale"}:
        body_scales = embodiment.solve(**(optimization_options or {}))
    else:
        body_scales = embodiment.solve()

    # Determine the explicit target mass.
    if strategy == "baseline":
        target_mass_kg = float(embodiment.baseline_model.body_mass[1:].sum())
    else:
        target_mass_kg = float(embodiment.measurements["body_mass"])

    # Build and compile the scaled model.
    spec = embodiment.build(body_scales, target_mass_kg)
    model = spec.copy().compile()

    # Validate using the same target mass.
    validation = embodiment.validate(
        model,
        body_scales,
        target_mass_kg,
    )

    # Generate report
    report = {
        "strategy": strategy,
        "subject_id": subject_id,
        "baseline_path": str(baseline_path),
        "body_scales": body_scales,
        "validation": validation,
    }

    if strategy in {"height_scale", "segment_scale"}:
        report["optimization"] = embodiment.report

    return spec, model, report


def main(argv: list[str] | None = None) -> None:
    """Generate, validate and save embodiments using a YAML configuration."""

    # Read cli parameters
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Embodiment YAML configuration."
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Target subject split (KIT/BMLmovi); default: train.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Process one subject; otherwise process all target subjects in the selected split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model.xml and report.json.",
    )
    args = parser.parse_args(argv)

    # Load config file
    config = load_config(args.config)
    strategy = config["name"]

    # Baseline myofullbody path
    baseline_path = Path(get_xml_path("myofullbody")).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Installed MyoFullBody model not found: {baseline_path}"
        )

    # Selected subjects
    if strategy == "baseline":
        if args.subject is not None:
            parser.error("Baseline does not use --subject.")
        subject_ids = [None]
    else:
        # Default all subjects
        subjects = get_target_subject_ids(
            split=args.split,
            embodiment=strategy,
        )

        # Overwrite selected subjects by cli
        if args.subject is not None:
            subject_id = args.subject.strip()
            if not subject_id:
                parser.error("--subject must not be empty.")
            if subject_id not in subjects:
                parser.error(
                    f"Subject {subject_id!r} is not registered in target split {args.split!r}."
                )
            subject_ids = [subject_id]

        else:
            subject_ids = list(subjects)

        if not subject_ids:
            parser.error("No target subjects were found.")

    # Check all output directories before starting optimization
    for subject_id in subject_ids:
        output_dir = EMBODIMENTS_DIR / strategy

        if subject_id is not None:
            output_dir /= subject_id

        if output_dir.exists():
            if not output_dir.is_dir():
                raise NotADirectoryError(
                    f"Output path is not a directory: {output_dir}"
                )
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}; use --overwrite to replace outputs"
                )

    # Generate and save each embodiment independently
    succeeded = 0
    failures = []
    for index, subject_id in enumerate(subject_ids, start=1):
        print(
            f"[{index}/{len(subject_ids)}] Generating {strategy}: {subject_id or 'baseline'}",
            flush=True,
        )

        # Generate.
        # Save failure report if failed.
        try:
            spec, _, report = generate(
                strategy,
                subject_id,
                baseline_path=baseline_path,
                optimization_options=config.get("optimization"),
            )
        except OptimizationFailure as error:
            failure_path = save_failure_report(
                error.report,
                config=config,
                config_path=args.config,
                overwrite=args.overwrite,
            )
            failures.append(
                {
                    "subject_id": subject_id,
                    "error": str(error),
                    "report_path": str(failure_path.resolve()),
                }
            )
            print(f"Skipped {subject_id}: {error}", flush=True)
            print(f"Failure report: {failure_path}", flush=True)
            continue

        # Save model.xml and the report.json
        report["config_path"] = str(args.config.expanduser().resolve())
        report["config"] = config
        model_path = save(spec, report, overwrite=args.overwrite)
        succeeded += 1

        # Print optimization message
        validation = report["validation"]
        print(f"Model:  {model_path}")
        print(f"Report: {model_path.parent / 'report.json'}")
        print(f"Height: {validation['measurements']['height']['predicted']:.6f} m")
        print(f"Mass:   {validation['body_mass_kg']:.6f} kg")
        if validation["target_body_mass_kg"] is not None:
            print(f"Mass error: {validation['body_mass_error_kg']:.3e} kg")

    # Finish
    print(f"Done: {succeeded} succeeded, {len(failures)} failed.")
    if failures:
        print(
            f"Failed subjects: {', '.join(entry['subject_id'] for entry in failures)}"
        )


if __name__ == "__main__":
    main()
