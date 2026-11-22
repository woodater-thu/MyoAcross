"""Fetch licensed AMASS motions and run the prebuilt single-image demonstration."""

from __future__ import annotations

import argparse
from getpass import getpass
import json
import os
import shutil
from pathlib import Path
import subprocess
import tarfile
import xml.etree.ElementTree as ET

import requests
from omegaconf import OmegaConf

from myoacross.utils.datalists import get_motion_specs
from myoacross.utils.paths import DATASET_ROOT, PROJECT_ROOT


AMASS_KIT_URL = (
    "https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&"
    "sfile=amass_per_dataset/smplh/gender_specific/mosh_results/KIT.tar.bz2"
)
SUBJECT_ID = "demo_self_67kg"
WALK_MOTION_ID = "KIT__11__WalkingStraightForwards09_poses"


def selected_motions(evaluate: bool):
    """Select one walking clip, or all 64 clips for full evaluation."""
    specs = get_motion_specs(motion_set="kit_flat_locomotion_72")
    if evaluate:
        if len(specs) != 64:
            raise ValueError(f"Expected 64 selected KIT motions; found {len(specs)}")
        return specs
    return (next(spec for spec in specs if spec.motion_id == WALK_MOTION_ID),)



def download_kit(destination: Path) -> None:
    """Download the official KIT archive with the user's own AMASS account."""
    username = input("AMASS account email: ").strip()
    password = getpass("AMASS password: ")
    if not username or not password:
        raise ValueError("AMASS account email and password are required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        with requests.post(
            AMASS_KIT_URL,
            data={"username": username, "password": password},
            stream=True,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
        with partial.open("rb") as stream:
            if stream.read(3) != b"BZh":
                raise ValueError(
                    "AMASS returned a page instead of KIT.tar.bz2. "
                    "Check account access, or download the official archive in a browser "
                    "and pass it with --archive."
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_motions(archive_path: Path, source_paths: tuple[Path, ...]) -> None:
    """Extract only the requested KIT motions from the licensed archive."""
    expected = {path.as_posix(): path for path in source_paths}
    found = {}
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            for index, part in enumerate(parts):
                if part == "KIT":
                    key = Path(*parts[index:]).as_posix()
                    if key in expected:
                        if key in found:
                            raise ValueError(f"Duplicate KIT motion in archive: {key}")
                        found[key] = member
                    break
        missing = expected.keys() - found.keys()
        if missing:
            raise ValueError(
                f"Official KIT archive is missing {len(missing)} required motions; "
                f"first: {sorted(missing)[0]}"
            )
        for key, member in found.items():
            destination = DATASET_ROOT / expected[key]
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            temporary = destination.with_name(destination.name + ".part")
            try:
                with source, temporary.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)


def run_pipeline(
    demo_root: Path, checkpoint: Path, *, evaluate: bool = False,
    headless: bool = False, device: str = "cuda:0", output_root: Path | None = None,
) -> None:
    """Prepare references, optionally evaluate, and display policy-driven walking."""
    from musclemimic_models import get_xml_path

    source_dir = demo_root / "embodiments/segment_scale" / SUBJECT_ID
    output_root = output_root or PROJECT_ROOT / "data/experiments/single_image_demo"
    specs = selected_motions(evaluate)
    model_dir = output_root / "embodiments/segment_scale" / SUBJECT_ID
    shutil.copytree(source_dir, model_dir)
    if not (model_dir / "model.xml").is_file():
        raise FileNotFoundError(model_dir / "model.xml")
    report_path = model_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "passed":
        raise ValueError("The demo model did not pass export checks")
    baseline_path = Path(get_xml_path("myofullbody")).resolve()
    # Resolve shared resources only in the generated XML; keep the packaged
    # individual model portable and leave its fitted parameters unchanged.
    model_path = model_dir / "model.xml"
    tree = ET.parse(model_path)
    compiler = tree.getroot().find("compiler")
    compiler.set("meshdir", str(baseline_path.parent.parent))
    compiler.set("texturedir", str((PROJECT_ROOT / "data/myoacross/assets").resolve()))
    tree.write(model_path, encoding="unicode")
    baseline = str(baseline_path)
    if report.get("baseline_path") != baseline:
        report["baseline_path"] = baseline
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    config = {
        "subject_id": SUBJECT_ID,
        "output_root": str(output_root),
        "cache_root": str(output_root / "cache"),
        "retarget_config": "myoacross/configs/retarget/gmr_fit.yaml",
        "checkpoint_path": str(checkpoint),
        "policy_scheme": "segment_film",
        "motion_ids": [spec.motion_id for spec in specs],
        "device": device,
        "envs_per_subject": len(specs),
    }
    config_path = output_root / "pipeline.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    stages = [(PROJECT_ROOT / "tools/gmr_fit/.venv/bin/python", "retarget")]
    if evaluate or headless:
        stages.append((PROJECT_ROOT / ".venv/bin/python", "evaluate"))
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["WARP_CACHE_PATH"] = str(output_root / "cache/warp")
    environment["CUDA_CACHE_PATH"] = str(output_root / "cache/cuda")
    environment["MPLCONFIGDIR"] = str(output_root / "cache/matplotlib")
    for python, stage in stages:
        if not python.is_file():
            raise FileNotFoundError(python)
        subprocess.run(
            [str(python), "-m", "myoacross.utils.image2msksim.run",
             "--config", str(config_path), "--stage", stage],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )

    if not headless:
        python = PROJECT_ROOT / ".venv/bin/python"
        if not python.is_file():
            raise FileNotFoundError(python)
        motion = output_root / "retargets/segment_scale" / SUBJECT_ID / WALK_MOTION_ID / "motion.npz"
        print("Opening policy-controlled walking. Close the viewer to finish.", flush=True)
        subprocess.run(
            [str(python), "-m", "myoacross.play",
             "--model", str(model_dir / "model.xml"), "--motion", str(motion),
             "--checkpoint", str(checkpoint), "--device", device,
             "--camera", "front-side"],
            cwd=PROJECT_ROOT, env=environment, check=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path,
        help="Official KIT.tar.bz2 already downloaded with your AMASS account",
    )
    parser.add_argument(
        "--demo-root", type=Path,
        default=PROJECT_ROOT / "data/demo/single_image",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "data/myoacross/rl/checkpoint/final.pt",
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Extract selected source motions only; do not retarget, evaluate, or open a viewer",
    )
    parser.add_argument("--evaluate", action="store_true",
                        help="Retarget and evaluate all 64 reference motions before playback")
    parser.add_argument("--headless", action="store_true",
                        help="Skip the viewer; evaluate the walking clip, or all clips with --evaluate")
    parser.add_argument("--device", default="cuda:0", help="GPU for playback and evaluation")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "data/experiments/single_image_demo",
                        help="New output directory; existing generated models are not overwritten")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not args.prepare_only:
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
        if not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            parser.error("No display detected. Use --headless on a server, or run from a graphical desktop.")
    specs = selected_motions(args.evaluate)
    source_paths = tuple(spec.source_path for spec in specs)
    missing = [path for path in source_paths if not (DATASET_ROOT / path).is_file()]
    if missing:
        archive = args.archive or PROJECT_ROOT / "data/downloads/amass/KIT.tar.bz2"
        if not archive.is_file():
            if args.archive:
                raise FileNotFoundError(archive)
            download_kit(archive)
        extract_motions(archive, source_paths)
    print(f"Prepared {len(source_paths)} KIT source motions in {DATASET_ROOT / 'KIT'}", flush=True)
    if not args.prepare_only:
        run_pipeline(
            args.demo_root.expanduser().resolve(), checkpoint,
            evaluate=args.evaluate, headless=args.headless, device=args.device,
            output_root=args.output_root.expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
