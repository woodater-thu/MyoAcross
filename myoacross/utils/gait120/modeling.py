"""Build subject models through the existing measurement and embodiment APIs."""

from pathlib import Path


def measure_subjects(config, records):
    from myoacross.embodiment.smplh_to_measurements import generate_measurements

    root = Path(config["output_root"])
    shapes = {
        r["subject_id"]: [root / "shapes" / r["subject_id"] / "shape_identity.npz"]
        for r in records
    }
    return generate_measurements(
        shapes,
        metadata_path=root / "subjects.json",
        output_dir=root / "embodiments/measurements",
    )


def build_subject_models(config, records):
    from musclemimic_models import get_xml_path
    from myoacross.embodiment.config import load_config
    from myoacross.embodiment.generate import generate
    from myoacross.embodiment.export import save

    root = Path(config["output_root"])
    options = load_config(config["embodiment_config"])
    for record in records:
        spec, _, report = generate(
            "segment_scale",
            record["subject_id"],
            baseline_path=Path(get_xml_path("myofullbody")),
            measurements_dir=root / "embodiments/measurements",
            optimization_options=options.get("optimization"),
        )
        save(spec, report, output_root=root / "embodiments")
