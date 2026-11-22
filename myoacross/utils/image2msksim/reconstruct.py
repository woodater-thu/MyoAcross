"""Run HSMR in its separate environment and export target measurements."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from importlib.metadata import distribution
import io
import urllib.request
import zipfile

from myoacross.utils.paths import PROJECT_ROOT, resolve_path


def select_person(count: int, person_index: int | None) -> int:
    """Require an explicit detection index when more than one person is present."""
    if count == 0:
        raise ValueError("No person detected in the image")
    if person_index is None:
        if count != 1:
            raise ValueError(f"{count} people detected; set person_index explicitly")
        return 0
    if type(person_index) is not int or not 0 <= person_index < count:
        raise ValueError(f"person_index must be an integer in [0, {count - 1}]")
    return person_index


def compute_reconstructed_height(model, betas) -> float:
    """Measure the shaped skin template along SKEL's vertical axis."""
    import torch

    shape = torch.as_tensor(
        betas, dtype=model.shapedirs.dtype, device=model.shapedirs.device
    )
    vertices = model.skin_template_v + torch.einsum("vci,i->vc", model.shapedirs, shape)
    height = float((vertices[:, 1].max() - vertices[:, 1].min()).item())
    if not math.isfinite(height) or height <= 0:
        raise ValueError("The reconstructed standard-pose height must be positive")
    return height


def export_measurements(
    subject_id, gender, betas, body_mass_kg, output_dir, *,
    height_m, reconstructed_height_m, model_dir, device
):
    """Calibrate standard-pose lengths and preserve measured body mass."""
    from myoacross.embodiment.smplh_to_measurements import (
        compute_tpose_geometry,
        compute_measurements,
        save_subject_measurements,
    )

    if not math.isfinite(body_mass_kg) or body_mass_kg <= 0:
        raise ValueError("body_mass_kg must be positive")
    path = Path(output_dir) / f"{subject_id}.json"
    if path.exists():
        raise FileExistsError(path)
    joints, names = compute_tpose_geometry(
        gender, betas, model_dir=model_dir, device=device
    )
    measurements = compute_measurements(joints, names)
    scale = height_m / reconstructed_height_m
    measurements = {name: value * scale for name, value in measurements.items()}
    measurements["height"] = height_m
    measurements["body_mass"] = body_mass_kg
    return save_subject_measurements(
        subject_id, gender, betas, measurements, output_dir=output_dir
    )


def prepare_hsmr_package() -> Path:
    """Install inference files omitted by the upstream wheel, without a clone."""
    package = distribution("lib")
    root = Path(package.locate_file(""))
    if not (root / "lib/kits/hsmr_demo.py").is_file():
        source = json.loads(package.read_text("direct_url.json"))
        revision = source["vcs_info"]["commit_id"]
        url = f"https://codeload.github.com/IsshikiHugh/HSMR/zip/{revision}"
        with urllib.request.urlopen(url) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        prefix = f"HSMR-{revision}/"
        with archive:
            for entry in archive.infolist():
                relative = Path(entry.filename.removeprefix(prefix))
                if (
                    not entry.filename.startswith(prefix)
                    or entry.is_dir()
                    or relative.parts[0] not in ("lib", "configs")
                ):
                    continue
                if ".." in relative.parts:
                    raise ValueError("Invalid HSMR archive path")
                destination = root / relative
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(entry))
    # Upstream ProjManager checks these paths at import time.
    for name in ("data_inputs", "data_outputs"):
        (root / name).mkdir(exist_ok=True)
    return root


def _prepare_input_image(config: dict, output_dir: Path) -> tuple[Path, dict]:
    """Use a photograph or export one explicitly selected video frame."""
    image = config.get("image_path")
    video = config.get("video_path")
    if bool(image) == bool(video):
        raise ValueError("Set exactly one of image_path and video_path")
    if image:
        return resolve_path(image, PROJECT_ROOT), dict(image_path=image)

    frame_index = config.get("video_frame_index")
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("video_frame_index must be a nonnegative integer")
    import cv2

    path = output_dir / "input.png"
    if path.exists():
        raise FileExistsError(path)
    capture = cv2.VideoCapture(str(resolve_path(video, PROJECT_ROOT)))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError(f"Cannot seek video frame {frame_index}: {video}")
        success, frame = capture.read()
        if not success:
            raise ValueError(f"Cannot read video frame {frame_index}: {video}")
        actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if actual_frame != frame_index:
            raise ValueError(f"Requested video frame {frame_index}, decoded {actual_frame}")
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Cannot save video frame: {path}")
    return path, dict(
        image_path=path.name,
        video_path=video,
        video_frame_index=actual_frame,
        video_fps=fps,
    )


def reconstruct_image(config: dict) -> Path:
    """Infer one selected person with the upstream detector and image pipeline."""
    height_m = float(config["height_m"])
    if not math.isfinite(height_m) or height_m <= 0:
        raise ValueError("height_m must be positive")
    output_root = resolve_path(config["output_root"], PROJECT_ROOT)
    cache = resolve_path(config["cache_root"], PROJECT_ROOT)
    os.environ["FVCORE_CACHE"] = str(cache / "detector")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    import numpy as np
    import torch

    hsmr = config["hsmr"]
    prepare_hsmr_package()
    from lib.platform import PM

    PM.inputs = resolve_path(hsmr["model_root"], PROJECT_ROOT).parent
    PM.outputs = resolve_path(config["output_root"], PROJECT_ROOT)
    from lib.kits.hsmr_demo import (
        build_detector,
        imgs_det2patches,
        build_inference_pipeline,
        IMG_MEAN_255,
        IMG_STD_255,
    )
    from lib.utils.media import load_img

    model_root = resolve_path(hsmr["model_root"], PROJECT_ROOT)
    output_root = resolve_path(config["output_root"], PROJECT_ROOT)
    destination = output_root / "reconstructions" / config["subject_id"]
    if destination.exists():
        raise FileExistsError(destination)
    image_path, input_source = _prepare_input_image(config, destination)
    image, _ = load_img(image_path)
    detector = build_detector(batch_size=1, max_img_size=512, device=hsmr["device"])
    detections, ratios = detector([image])
    # Keep all detected people so selection never silently truncates the image.
    count = (
        int(
            (
                (detections[0]["pred_classes"] == 0) & (detections[0]["scores"] > 0.5)
            ).sum()
        )
        if detections[0] is not None
        else 0
    )
    if count == 0:
        raise ValueError("No person detected in the image")
    patches, metadata = imgs_det2patches([image], detections, ratios, count)
    selected = select_person(len(patches), config.get("person_index"))
    del detector
    pipeline = build_inference_pipeline(model_root=model_root, device=hsmr["device"])
    normalized = (patches[selected : selected + 1] - IMG_MEAN_255) / IMG_STD_255
    with torch.inference_mode():
        result = pipeline(normalized.transpose(0, 3, 1, 2))
    parameters = {
        name: value[0].detach().cpu().numpy()
        for name, value in result["pd_params"].items()
    }
    # Read identity from the checkpoint's resolved SKEL configuration.
    gender = str(pipeline.cfg.SKEL.gender)
    model_dir = Path(str(pipeline.cfg.SKEL.model_path)).resolve()
    reconstructed_height_m = compute_reconstructed_height(
        pipeline.skel_model, parameters["betas"]
    )
    measurement_path = export_measurements(
        config["subject_id"],
        gender,
        parameters["betas"],
        float(config["body_mass_kg"]),
        output_root / "embodiments/measurements",
        height_m=height_m,
        reconstructed_height_m=reconstructed_height_m,
        model_dir=model_dir,
        device=hsmr["device"],
    )
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination / "parameters.npz",
        **parameters,
        patch_cam_t=result["pd_cam_t"][0].detach().cpu().numpy(),
        bounding_boxes=np.asarray(metadata["bbx_cs"]),
    )
    source = json.loads(distribution("lib").read_text("direct_url.json"))
    revision = source["vcs_info"]["commit_id"]
    (destination / "source.json").write_text(
        json.dumps(
            dict(
                **input_source,
                body_mass_kg=config["body_mass_kg"],
                person_index=selected,
                detected_people=len(patches),
                gender=gender,
                hsmr_revision=revision,
                model_root=hsmr["model_root"],
                scale_source="measured_height",
                height_calibration=True,
                height_m=height_m,
                reconstructed_height_m=reconstructed_height_m,
                height_scale=height_m / reconstructed_height_m,
            ),
            indent=2,
        )
        + "\n"
    )
    return measurement_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepare-environment", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_environment:
        print(prepare_hsmr_package())
        return
    if args.config is None:
        parser.error("--config is required for image reconstruction")
    from myoacross.utils.config import load_config

    print(reconstruct_image(load_config(args.config)))


if __name__ == "__main__":
    main()
