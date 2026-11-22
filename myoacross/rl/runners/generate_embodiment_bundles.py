"""Export subject or generated embodiment bundles offline."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import hashlib
import json

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from mjlab.sim.sim import MujocoCfg, SimulationCfg

from myoacross.embodiment.measurements import (
    compute_myofullbody_height,
)
from myoacross.rl.embodiments.subjects import select_target_ids
from myoacross.utils.paths import EMBODIMENTS_DIR, PROJECT_ROOT, resolve_path
from myoacross.rl.embodiments.myofullbody_cfg import SubjectSource
from myoacross.rl.embodiments.bundle import (
    EmbodimentBundle,
    WarpBundleSwitcher,
)
from myoacross.rl.embodiments.features import MorphologyFeatures
from myoacross.rl.embodiments.virtual.parameters import (
    GenerationParameters,
    HEIGHT_PARAMETER_NAMES,
    SEGMENT_PARAMETER_NAMES,
)
from myoacross.rl.embodiments.virtual.sampling import (
    MorphologySampler,
    build_height_covariance,
    build_segment_covariance,
)
from myoacross.rl.embodiments.virtual.generator import (
    VirtualEmbodimentGenerator,
)
from myoacross.rl.tasks.motion_tracking.myofullbody_env_cfg import (
    make_myofullbody_scene_cfg,
)


def _build_body_conditions(settings):
    """Expand fixed one-sided shortening levels; include the original once."""
    if not isinstance(settings, dict) or set(settings) != {
        "model_root", "relative_shortening", "sides"
    }:
        raise ValueError("body_perturbation requires model_root, relative_shortening, and sides")
    if not isinstance(settings["model_root"], (str, Path)) or not str(settings["model_root"]).strip():
        raise ValueError("body_perturbation.model_root must be a nonempty path")
    changes, sides = settings["relative_shortening"], settings["sides"]
    if (not isinstance(changes, list) or not changes
            or any(isinstance(d, bool) or not isinstance(d, (int, float))
                   or not np.isfinite(d) or not 0 <= d < 1 for d in changes)
            or len(set(changes)) != len(changes)):
        raise ValueError("relative_shortening must contain unique finite values in [0, 1)")
    if (not isinstance(sides, list) or not sides
            or any(side not in ("left", "right") for side in sides)
            or len(set(sides)) != len(sides)):
        raise ValueError("sides must contain unique left or right values")
    return [("original", None, 0.0)] + [
        (f"{side}_{100 * change:g}pct".replace(".", "p"), side, float(change))
        for change in sorted(changes) if change != 0 for side in sides
    ]


def run_generation(config: dict) -> list[Path]:
    """Export training banks or fixed test conditions from a configuration."""
    required = {"data"}
    allowed = required | {
        "count_per_subject",
        "seed",
        "sampling",
        "simulation",
        "workers",
        "body_perturbation",
    }

    missing = required - set(config)
    extra = set(config) - allowed
    if missing or extra:
        raise ValueError(
            f"Invalid generation configuration: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    workers = config.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")

    data = config["data"]
    if (
        not isinstance(data, dict)
        or {"embodiment", "bundle_root"} - set(data)
        or set(data) - {"embodiment", "subject_ids", "bundle_root", "split"}
    ):
        raise ValueError(
            "Generation data requires embodiment and bundle_root; optional subject_ids and split"
        )

    embodiment = data["embodiment"]
    if embodiment not in (
        "baseline",
        "height_scale",
        "segment_scale",
    ):
        raise ValueError(f"Unknown embodiment: {embodiment!r}")

    split = data.get("split", "train")
    if split not in {"train", "test"}:
        raise ValueError("Generation split must be train or test")
    body_settings = config.get("body_perturbation")
    body_conditions = _build_body_conditions(body_settings) if body_settings is not None else None
    if body_conditions is not None and (embodiment != "segment_scale" or split != "test"
                                        or config.get("sampling") is not None):
        raise ValueError("Fixed body conditions require segment_scale test subjects without sampling")

    sampling = config.get("sampling")
    generate_virtual = sampling is not None

    count = 1
    seed = 42

    if generate_virtual:
        if embodiment == "baseline":
            raise ValueError("Baseline does not support virtual morphology generation")

        if not isinstance(sampling, dict):
            raise ValueError("sampling must be a mapping")

        missing = {"count_per_subject", "seed"} - set(config)
        if missing:
            raise ValueError(f"Missing virtual generation settings: {sorted(missing)}")

        count = config["count_per_subject"]
        seed = config["seed"]

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count_per_subject must be a positive integer")

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        required_sampling = {"covariance", "max_relative_change"}
        allowed_sampling = required_sampling | {"max_attempts"}

        missing = required_sampling - set(sampling)
        extra = set(sampling) - allowed_sampling
        if missing or extra:
            raise ValueError(
                f"Invalid sampling configuration: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        factors = sampling["covariance"]
        if not isinstance(factors, dict):
            raise ValueError("sampling.covariance must be a mapping")

        build_covariance = (
            build_height_covariance
            if embodiment == "height_scale"
            else build_segment_covariance
        )

        # Validate virtual sampling before exporting subject bundles.
        MorphologySampler(
            covariance=build_covariance(**factors),
            max_relative_change=np.asarray(
                sampling["max_relative_change"],
                dtype=np.float64,
            ),
            max_attempts=sampling.get("max_attempts", 1000),
        )
    else:
        unused = {
            key for key in ("count_per_subject", "seed") if config.get(key) is not None
        }
        if unused:
            raise ValueError(
                f"{sorted(unused)} require sampling; omit or set them to null for subject-only export"
            )

    simulation = config.get("simulation", {})
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be a mapping")

    simulation = dict(simulation)
    mujoco_options = simulation.pop("mujoco", {})
    if not isinstance(mujoco_options, dict):
        raise ValueError("simulation.mujoco must be a mapping")

    simulation_config = SimulationCfg(
        mujoco=MujocoCfg(**mujoco_options),
        **simulation,
    )

    if embodiment == "baseline":
        subject_dirs = {None: EMBODIMENTS_DIR / "baseline"}
    else:
        subject_ids = select_target_ids(split, embodiment, data.get("subject_ids"))
        if not subject_ids:
            raise ValueError("The generation subject allowlist is empty")

        subject_dirs = {
            subject_id: EMBODIMENTS_DIR / embodiment / subject_id
            for subject_id in subject_ids
        }

    root = (
        Path(data["bundle_root"])
        .expanduser()
        .resolve()
    )

    if body_conditions is not None:
        model_root = resolve_path(str(body_settings["model_root"]), PROJECT_ROOT)
        if root.exists():
            raise FileExistsError(f"Bundle directory already exists: {root}")
        for name, side, _ in body_conditions:
            if side is not None and (model_root / name).exists():
                raise FileExistsError(f"Model directory already exists: {model_root / name}")
        root.mkdir(parents=True)
        index_paths, records = [], []
        for name, side, shortening in body_conditions:
            condition = dict(name=name, shortened_side=side, relative_shortening=shortening)
            index_paths.extend(generate_embodiment_bundles(
                subject_dirs, root / name, embodiment=embodiment,
                morphology_source="subject", simulation_config=simulation_config,
                workers=workers, body_condition=condition, model_root=model_root,
            ))
            records.append(condition)
        manifest = dict(split=split, embodiment=embodiment,
                        subject_ids=list(subject_dirs), conditions=records, config=config)
        with (root / "conditions.json").open("x") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
        return index_paths

    sources = ["subject"]
    if generate_virtual:
        sources.append("virtual")

    # Check all destinations before starting either export.
    for source in sources:
        destination = root / source
        if destination.exists():
            raise FileExistsError(f"Bundle directory already exists: {destination}")

    index_paths = []

    for source in sources:
        is_virtual = source == "virtual"

        paths = generate_embodiment_bundles(
            subject_dirs=subject_dirs,
            output_dir=root / source,
            embodiment=embodiment,
            morphology_source=source,
            simulation_config=simulation_config,
            count_per_subject=count if is_virtual else 1,
            seed=seed,
            sampling=sampling if is_virtual else None,
            workers=workers,
        )
        index_paths.extend(paths)

    return index_paths


def compile_myofullbody_scene(
    subject: SubjectSource,
    *,
    simulation_config: SimulationCfg,
    init_state: EntityCfg.InitialStateCfg | None = None,
) -> mujoco.MjModel:
    """Compile the same scene layout used by online training."""
    scene_cfg = make_myofullbody_scene_cfg(
        subject,
        num_envs=1,
        init_state=init_state,
    )
    scene = Scene(scene_cfg, device="cpu")
    model = scene.compile()
    simulation_config.mujoco.apply(model)
    return model


def _iter_models(
    subject_dir: Path,
    *,
    sampler: MorphologySampler | None,
    count: int,
    seed: int,
    body_condition: dict | None = None,
    model_root: Path | None = None,
) -> Iterator[
    tuple[
        GenerationParameters | None,
        mujoco.MjSpec,
        mujoco.MjModel,
    ]
]:
    """Yield original or generated models without retaining the bank."""
    if body_condition is not None and body_condition["shortened_side"] is not None:
        from myoacross.embodiment.export import save
        from myoacross.rl.embodiments.virtual.generator import VirtualEmbodiment
        from myoacross.rl.embodiments.virtual.parameters import REGION_NAMES, SegmentScaleParameters

        source = json.loads((subject_dir / "report.json").read_text())
        scales = dict(source["scales"]["regions"])
        side = body_condition["shortened_side"]
        suffix = "l" if side == "left" else "r"
        for region in ("thigh", "shank"):
            scales[f"{region}_{suffix}"] *= 1 - body_condition["relative_shortening"]
        baseline = resolve_path(source["baseline_path"], subject_dir)
        parameters = SegmentScaleParameters(tuple(scales[name] for name in REGION_NAMES),
                                            source["mass"]["actual_kg"])
        body = VirtualEmbodiment(mujoco.MjSpec.from_file(str(baseline)), parameters,
                                 source["subject_id"])
        spec, model, _, validation = body.generate()
        save(spec, dict(strategy="segment_scale", subject_id=source["subject_id"],
                        baseline_path=str(baseline), body_scales=body.solve(),
                        validation=validation,
                        config=dict(condition=body_condition, source_model=str(subject_dir / "model.xml"),
                                    source_scales=source["scales"]["regions"],
                                    source_mass_kg=source["mass"]["actual_kg"])),
             output_root=model_root / body_condition["name"])
        yield None, spec, model
        return

    if sampler is None:
        spec = mujoco.MjSpec.from_file(str(subject_dir / "model.xml"))
        yield None, spec, spec.copy().compile()
        return

    generator = VirtualEmbodimentGenerator(
        subject_dir=subject_dir,
        sampler=sampler,
    )
    rng = np.random.default_rng(seed)

    for parameters, spec, model, _, _ in generator.generate(count, rng):
        yield parameters, spec, model


def generate_embodiment_bundles(
    subject_dirs: Mapping[str | None, str | Path],
    output_dir: str | Path,
    *,
    embodiment: str,
    morphology_source: str,
    simulation_config: SimulationCfg,
    count_per_subject: int = 1,
    seed: int = 42,
    sampling: dict | None = None,
    workers: int = 1,
    body_condition: dict | None = None,
    model_root: Path | None = None,
) -> list[Path]:
    """
    Export one bank with a shared embodiment and morphology source.

    Baseline exports one shared model using a single None source key.
    Height and segment subject mode export one existing model per subject.
    Virtual mode samples around each subject's accepted model.
    Fixed test conditions scale selected regions and save viewer XML files.

    Existing banks are never overwritten. An interrupted export may
    leave an incomplete directory; choose a fresh output directory
    after inspecting such a failure.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")

    if embodiment not in (
        "baseline",
        "height_scale",
        "segment_scale",
    ):
        raise ValueError(f"Unknown embodiment: {embodiment!r}")

    if morphology_source not in ("subject", "virtual"):
        raise ValueError(f"Unknown morphology_source: {morphology_source!r}")

    if (
        isinstance(count_per_subject, bool)
        or not isinstance(count_per_subject, int)
        or count_per_subject <= 0
    ):
        raise ValueError("count_per_subject must be a positive integer")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    sampler = None
    sampling_record = None

    if morphology_source == "subject":
        if count_per_subject != 1:
            raise ValueError("Subject mode requires count_per_subject=1")
        if sampling is not None:
            raise ValueError("Subject mode does not use sampling parameters")
    else:
        if embodiment == "baseline":
            raise ValueError("Baseline does not support virtual morphology generation")
        if not isinstance(sampling, dict):
            raise ValueError("Virtual mode requires a sampling mapping")

        required = {"covariance", "max_relative_change"}
        allowed = required | {"max_attempts"}
        missing = required - set(sampling)
        extra = set(sampling) - allowed
        if missing or extra:
            raise ValueError(
                f"Invalid sampling configuration: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        factors = sampling["covariance"]
        if not isinstance(factors, dict):
            raise ValueError("sampling.covariance must be a mapping")

        if embodiment == "height_scale":
            covariance = build_height_covariance(**factors)
            parameter_names = HEIGHT_PARAMETER_NAMES
        else:
            covariance = build_segment_covariance(**factors)
            parameter_names = SEGMENT_PARAMETER_NAMES

        bounds = np.asarray(
            sampling["max_relative_change"],
            dtype=np.float64,
        )
        max_attempts = sampling.get("max_attempts", 1000)

        sampler = MorphologySampler(
            covariance=covariance,
            max_relative_change=bounds,
            max_attempts=max_attempts,
        )

        sampling_record = {
            "parameter_names": list(parameter_names),
            "covariance_parameters": factors,
            "covariance": covariance.tolist(),
            "max_relative_change": bounds.tolist(),
            "max_attempts": max_attempts,
            "distribution": ("relative Gaussian perturbations before rejection"),
        }

    if body_condition is not None and (embodiment != "segment_scale"
                                      or morphology_source != "subject" or model_root is None):
        raise ValueError("Fixed scaling requires segment_scale subject export and model_root")

    if not subject_dirs:
        raise ValueError("No subject directories were provided")

    if embodiment == "baseline" and set(subject_dirs) != {None}:
        raise ValueError("Baseline requires exactly one shared source keyed by None")

    # Validate all source identities before writing output.
    sources = []
    common_baseline = None

    for subject_id, directory in subject_dirs.items():
        if embodiment != "baseline" and (
            not isinstance(subject_id, str)
            or not subject_id.strip()
            or subject_id in (".", "..")
            or "/" in subject_id
            or "\\" in subject_id
        ):
            raise ValueError(f"Invalid subject ID: {subject_id!r}")

        directory = Path(directory).expanduser().resolve()

        for filename in ("model.xml", "report.json"):
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(path)

        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        expected_subject = None if embodiment == "baseline" else subject_id

        if (
            report.get("status") != "passed"
            or report.get("strategy") != embodiment
            or "subject_id" not in report
            or report["subject_id"] != expected_subject
        ):
            raise ValueError(f"Source identity or validation mismatch: {directory}")

        baseline = Path(report["baseline_path"]).expanduser()
        if not baseline.is_absolute():
            baseline = directory / baseline
        baseline = baseline.resolve()

        if not baseline.is_file():
            raise FileNotFoundError(baseline)

        if common_baseline is None:
            common_baseline = baseline
        elif baseline != common_baseline:
            raise ValueError(f"Subjects use different baseline templates: {directory}")

        sources.append((subject_id, directory))

    sources.sort()

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    bank_record = {
        "embodiment": embodiment,
        "morphology_source": morphology_source,
        "baseline_path": str(common_baseline),
        "seed": seed if sampler is not None else None,
        "count_per_subject": count_per_subject,
        "sampling": sampling_record,
        "simulation_config": asdict(simulation_config),
        "subject_ids": [
            subject_id for subject_id, _ in sources if subject_id is not None
        ],
    }

    if body_condition is not None:
        bank_record["body_condition"] = body_condition

    # Ensure provenance is JSON-serializable before starting export.
    bank_json = json.dumps(
        bank_record,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    expected_signature = None
    index_paths = []

    generate_subject = partial(
        _generate_subject_bundles,
        output_dir=output_dir,
        embodiment=embodiment,
        morphology_source=morphology_source,
        simulation_config=simulation_config,
        count_per_subject=count_per_subject,
        seed=seed,
        sampler=sampler,
        sampling_record=sampling_record,
        common_baseline=common_baseline,
        body_condition=body_condition,
        model_root=model_root,
    )
    subject_ids, directories = zip(*sources)
    with ExitStack() as stack:
        if workers == 1 or len(sources) == 1:
            results = map(generate_subject, subject_ids, directories)
        else:
            executor = stack.enter_context(ProcessPoolExecutor(
                max_workers=min(workers, len(sources)),
                mp_context=get_context("spawn"),
            ))
            results = executor.map(generate_subject, subject_ids, directories)

        for position, (subject_id, (index_path, signature)) in enumerate(
            zip(subject_ids, results), start=1,
        ):
            if expected_signature is None:
                expected_signature = signature
            elif signature != expected_signature:
                raise ValueError(f"Incompatible model signature: {subject_id}")
            index_paths.append(index_path)
            print(
                f"[{position}/{len(sources)}] "
                f"{embodiment}/{morphology_source}: {subject_id or 'shared'} completed",
                flush=True,
            )

    # Written last: its presence marks successful completion of the bank.
    with (output_dir / "bank.json").open("x", encoding="utf-8") as file:
        file.write(bank_json)

    return index_paths


def _generate_subject_bundles(
    subject_id: str | None,
    directory: Path,
    *,
    output_dir: Path,
    embodiment: str,
    morphology_source: str,
    simulation_config: SimulationCfg,
    count_per_subject: int,
    seed: int,
    sampler: MorphologySampler | None,
    sampling_record: dict | None,
    common_baseline: Path,
    body_condition: dict | None = None,
    model_root: Path | None = None,
) -> tuple[Path, str]:
    """Generate one subject's variants and index in an independent process."""
    expected_signature = None
    destination = output_dir
    if subject_id is not None:
        destination = output_dir / subject_id
        destination.mkdir()

    # Subject sampling remains stable when subjects are reordered.
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).digest()
    subject_seed = int.from_bytes(digest[:8], byteorder="little")

    records = []

    models = _iter_models(
        directory,
        sampler=sampler,
        count=count_per_subject,
        seed=subject_seed,
        body_condition=body_condition,
        model_root=model_root,
    )

    for index, (parameters, spec, model) in enumerate(models):
        # Always measure the actual standalone physical model.
        features = MorphologyFeatures.from_model(model)
        subject_height_m = compute_myofullbody_height(model)

        scene_model = compile_myofullbody_scene(
            spec,
            simulation_config=simulation_config,
        )

        with wp.ScopedDevice("cpu"):
            warp_model = mjwarp.put_model(scene_model)
            simulation_config.apply_wp_opt(warp_model.opt)

            switcher = WarpBundleSwitcher(warp_model)
            signature = switcher.compute_model_signature(scene_model)

            if expected_signature is None:
                expected_signature = signature
            elif signature != expected_signature:
                raise ValueError(
                    f"Incompatible model signature: "
                    f"{subject_id}, variant {index}. "
                    "Model layout or non-injectable values differ."
                )

            variant_id = "subject" if sampler is None else f"virtual_{index:06d}"
            filename = f"{variant_id}.npz"

            bundle = EmbodimentBundle(
                source_subject_id=subject_id,
                variant_id=variant_id,
                embodiment=embodiment,
                morphology_source=morphology_source,
                parameters=parameters,
                features=features,
                subject_height_m=subject_height_m,
                model_signature=signature,
                arrays=switcher.extract_arrays(),
            )
            bundle.save(destination / filename)

        records.append(
            {
                "variant_id": variant_id,
                "bundle": filename,
                "features": features.to_dict(),
                "subject_height_m": subject_height_m,
            }
        )

        del bundle, switcher, warp_model, scene_model

    model_directory = directory
    if body_condition is not None and body_condition["shortened_side"] is not None:
        model_directory = model_root / body_condition["name"] / embodiment / subject_id
    metadata = {
        "source_subject_id": subject_id,
        "source_directory": str(model_directory),
        "embodiment": embodiment,
        "morphology_source": morphology_source,
        "baseline_path": str(common_baseline),
        "seed": subject_seed if sampler is not None else None,
        "count": len(records),
        "sampling": sampling_record,
        "simulation_config": asdict(simulation_config),
        "model_signature": expected_signature,
        "samples": records,
    }

    if body_condition is not None:
        metadata["body_condition"] = body_condition
        metadata["original_source_directory"] = str(directory)

    index_path = destination / "index.json"
    with index_path.open("x", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    return index_path, expected_signature
