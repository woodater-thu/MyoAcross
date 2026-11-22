"""Fit individual SMPL-H shape with MoSh++ Stage I only."""

from pathlib import Path
import json
import pickle
import numpy as np
from .convert import BODY_MARKERS, prepare_marker_inputs


def fit_subject_shape(record, config):
    from moshpp.chmosh import mosh_stagei
    from moshpp.mosh_head import MoSh
    from moshpp.marker_layout.create_marker_layout_for_mocaps import (
        marker_labels_to_marker_layout,
    )

    root = Path(config["output_root"])
    subject = record["subject_id"]
    markers = prepare_marker_inputs(record, root / "markers" / subject)
    work = root / "moshpp" / subject
    work.mkdir(parents=True, exist_ok=False)
    layout = work / "marker_layout.json"
    marker_labels_to_marker_layout(
        list(BODY_MARKERS),
        str(layout),
        "smplh",
        wrist_markers_on_stick=False,
        separate_types=["body"],
    )
    source_config = config["moshpp"]
    settings = dict(
        mocap=dict(unit="mm", rotate=None),
        surface_model=dict(
            type="smplh",
            gender=record["metadata"]["gender"],
            fname=source_config["model_path"],
            num_betas=16,
        ),
        moshpp=dict(
            optimize_fingers=False,
            optimize_face=False,
            optimize_dynamics=False,
            optimize_betas=True,
            pose_body_prior_fname=source_config["pose_body_prior"],
            pose_hand_prior_fname=source_config["pose_hand_prior"],
            head_marker_corr_fname=source_config["head_marker_corr"],
            stagei_frame_picker=dict(
                stagei_mocap_fnames=[str(p) for p in markers], seed=config["seed"]
            ),
        ),
        dirs=dict(
            write_optimized_marker_layout=False,
            support_base_dir=source_config["support_dir"],
            work_base_dir=str(work),
            stagei_fname=str(work / "stagei.pkl"),
            marker_layout=dict(fname=str(layout)),
        ),
    )
    name = record["motions"][0]["motion_id"]
    settings["mocap"]["fname"] = str(root / "markers" / subject / f"{name}.npz")
    settings["mocap"]["basename"] = name
    settings["dirs"]["stageii_fname"] = str(work / f"{name}.pkl")
    settings["dirs"]["log_fname"] = str(work / "stagei.log")
    solver = MoSh(dict_cfg=settings)
    solver.mosh_stagei(mosh_stagei)
    with (work / "stagei.pkl").open("rb") as handle:
        shape = pickle.load(handle)
    destination = root / "shapes" / subject
    destination.mkdir(parents=True, exist_ok=False)
    np.savez(
        destination / "shape_identity.npz",
        betas=shape["betas"],
        gender=record["metadata"]["gender"],
    )
    (work / "source.json").write_text(json.dumps(record, indent=2) + "\n")
