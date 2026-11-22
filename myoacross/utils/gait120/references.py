"""Prepare original OpenSim joint, MCOS EMG, and corrected GRF references."""

from pathlib import Path
import json

import numpy as np

from .signals import JOINT_NAMES, EMG_CHANNELS, EMG_MAP, stance_profiles
from .mcos import resolve_tables


def assign_force_plates(path, markers, body_mass_kg):
    """Locate each active plate COP relative to synchronized heel-to-toe segments."""
    lines = Path(path).read_text().splitlines()
    header = lines.index("endheader") + 1
    columns = lines[header].split()
    values = np.loadtxt(lines[header + 1 :], ndmin=2)
    source_time = values[:, columns.index("time")]
    target = markers["time"]
    if not np.allclose(np.diff(target), 0.01, atol=1e-6):
        raise ValueError("GRF references require 100-Hz TRC samples")
    # The force acquisition ends one force sample before the final camera frame.
    if (
        source_time[0] > target[0] + 1e-6
        or target[-1] - source_time[-1] > np.median(np.diff(source_time)) + 1e-6
    ):
        raise ValueError(f"Force and marker timelines do not overlap: {path}")
    from mm_evalkit.datasets.gait120_kinetics import _normalize_grf

    normalized = _normalize_grf(
        values[:, [columns.index(f"ground_force{plate}_vy") for plate in (1, 2)]],
        "N",
        {"subject_weight_kg": body_mass_kg},
        Path(path),
    )
    feet = []
    for side in ("L", "R"):
        feet.append(
            [
                np.stack(
                    [
                        np.interp(
                            source_time,
                            target,
                            markers["markers"][
                                :, list(markers["labels"]).index(side + name), axis
                            ]
                            / 1000.0,
                        )
                        for axis in (0, 2)
                    ],
                    axis=1,
                )
                for name in ("HEE", "TOE")
            ]
        )
    forces = np.zeros((len(source_time), 2))
    sides = np.full((len(source_time), 2), -1, dtype=int)
    segments = []
    valid = np.ones(len(source_time), dtype=bool)
    for plate in (1, 2):
        force = values[:, columns.index(f"ground_force{plate}_vy")]
        cop = values[
            :,
            [
                columns.index(f"ground_force{plate}_px"),
                columns.index(f"ground_force{plate}_pz"),
            ],
        ]
        distances = []
        for heel, toe in feet:
            direction = toe - heel
            fraction = np.clip(
                np.sum((cop - heel) * direction, axis=1) / np.sum(direction**2, axis=1),
                0,
                1,
            )
            distances.append(
                np.linalg.norm(cop - heel - fraction[:, None] * direction, axis=1)
            )
        distances = np.array(distances).T
        # Use the same 0.05 BW contact threshold as MM-EvalKit kinetics.
        active = normalized[:, plate - 1] > 0.05
        if not np.isfinite(distances[active]).all():
            raise ValueError(f"Missing foot markers during force contact: {path}")
        nearest = np.argmin(distances, axis=1)
        # Extend an established plate identity to the measured contact tails.
        contact_sides = np.unique(nearest[active])
        assigned = (
            np.full(len(force), contact_sides[0])
            if len(contact_sides) == 1
            else nearest
        )
        sides[:, plate - 1] = assigned
        forces[np.arange(len(force)), assigned] += force
        changes = np.diff(np.r_[False, active, False].astype(int))
        for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        ):
            if len(np.unique(nearest[start:end])) != 1:
                valid[start:end] = False
            segments.append(
                dict(
                    plate=plate,
                    start_sec=float(source_time[start]),
                    end_sec=float(source_time[end - 1]),
                    sides=np.unique(nearest[start:end]).tolist(),
                    median_distance_m=np.median(distances[start:end], axis=0).tolist(),
                )
            )
    nearest_time = np.minimum(
        np.searchsorted(source_time, target), len(source_time) - 1
    )
    before = np.maximum(np.searchsorted(source_time, target, side="right") - 1, 0)
    return dict(
        time_sec=target,
        computed_grf_valid=valid[before] & valid[nearest_time],
        computed_grf=np.stack(
            [np.interp(target, source_time, forces[:, side]) for side in (0, 1)], axis=1
        ),
        plate_sides=sides[nearest_time],
        segments=segments,
    )


def prepare_grf_reference(record, trial, destination):
    """Join recorded steps and export side-labelled GRF with contact validity."""
    from .convert import read_trc

    destination = Path(destination)
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError(destination)
    source = Path(record["source_dir"]) / "MotionCapture/LevelWalking"
    identity = dict(subject_id=record["subject_id"], trial=trial)
    trcs = sorted((source / "TRC" / f"Trial{trial:02d}").glob("Step*.trc"))
    parts, provenance, notes = [], [], []
    for trc in trcs:
        mot = source / "MOT" / trc.parent.name / trc.with_suffix(".mot").name
        if not mot.is_file():
            notes.append(
                dict(**identity, step=trc.stem, reason="missing_force_plate_data")
            )
            continue
        result = assign_force_plates(
            mot, read_trc(trc), record["metadata"]["body_mass_kg"]
        )
        parts.append(result)
        provenance.append(
            dict(markers=str(trc), force_plate=str(mot), segments=result["segments"])
        )
        for segment in result["segments"]:
            if len(segment["sides"]) != 1:
                notes.append(
                    dict(
                        **identity,
                        step=trc.stem,
                        reason="ambiguous_plate_contact",
                        **segment,
                    )
                )
    if not parts:
        return notes or [dict(**identity, reason="missing_force_plate_data")]
    time = np.concatenate([p["time_sec"] for p in parts])
    force = np.concatenate([p["computed_grf"] for p in parts])
    valid = np.concatenate([p["computed_grf_valid"] for p in parts])
    # Step exports can share an endpoint. Keep one sample on the original clock.
    time, first, inverse = np.unique(time, return_index=True, return_inverse=True)
    combined_valid = np.ones(len(time), dtype=bool)
    np.logical_and.at(combined_valid, inverse, valid)
    if not np.allclose(np.diff(time), 0.01, atol=1e-6):
        return notes + [dict(**identity, reason="noncontiguous_grf_steps")]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as file:
        np.savez(
            file,
            time_sec=time,
            computed_grf=force[first],
            computed_grf_valid=combined_valid,
            computed_grf_columns=["ground_force_left_vy", "ground_force_right_vy"],
            computed_grf_units="N",
            subject_weight_kg=record["metadata"]["body_mass_kg"],
        )
    destination.with_suffix(".json").write_text(
        json.dumps(
            dict(
                **identity,
                method="synchronized_cop_to_heel_to_toe_segment",
                contact_threshold_bw=0.05,
                ambiguous_contact_policy="invalidate_contact_segment",
                sources=provenance,
            ),
            indent=2,
        )
        + "\n"
    )
    return notes


def load_grf_reference(path):
    """Read the GRF portion of the MM-EvalKit kinetics reference contract."""
    from mm_evalkit.datasets.gait120_kinetics import (
        _matrix,
        _columns,
        _grf_index,
        _normalize_grf,
        _scalar_text,
        _valid_rows,
        _LEFT_GRF_COLUMNS,
        _RIGHT_GRF_COLUMNS,
    )

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        columns = _columns(data, "computed_grf_columns", path)
        values = _matrix(data, "computed_grf", path)
        ids = [
            _grf_index(columns, _LEFT_GRF_COLUMNS, "left", path),
            _grf_index(columns, _RIGHT_GRF_COLUMNS, "right", path),
        ]
        grf = _normalize_grf(
            values[:, ids], _scalar_text(data, "computed_grf_units", path), data, path
        )
        valid = _valid_rows(data, "computed_grf_valid", len(grf), path)
        if "time_sec" not in data or not np.allclose(
            np.diff(data["time_sec"]), 0.01, atol=1e-6
        ):
            raise ValueError("Corrected GRF must retain the reference 100-Hz timeline")
        return np.array(data["time_sec"]), grf, valid


def prepare_reference_profiles(records, destination):
    """Average available human steps within each of the five walking trials."""
    from mm_evalkit.datasets.gait120 import _load_mot_step, INVERTED_JOINTS

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    audit = []
    for record in records:
        sid = record["subject_id"]
        source = Path(record["source_dir"])
        emg = resolve_tables(source / "EMG/ProcessedData.mat")["LevelWalking"]
        joint_trials, emg_trials, grf_trials = [], [], [[], []]
        for trial in range(1, 6):
            joints, muscles = [], []
            for step in (1, 2):
                loaded = _load_mot_step(
                    source
                    / f"JointAngle/LevelWalking/Trial{trial:02d}/step{step:02d}.mot"
                )
                if loaded is not None:
                    values, columns = loaded
                    angles = values[[columns.index(n) for n in JOINT_NAMES]].copy()
                    for i, name in enumerate(JOINT_NAMES):
                        if name in INVERTED_JOINTS:
                            angles[i] *= -1
                    joints.append(angles)
                table = (
                    emg.get(f"Trial{trial:02d}", {})
                    .get(f"Step{step:02d}", {})
                    .get("EMGs_interpolated")
                )
                if table is not None:
                    columns = list(table["variable_names"])
                    values = table["values"]
                    muscles.append(
                        np.stack(
                            [
                                np.interp(
                                    np.linspace(0, 1, 100),
                                    np.linspace(0, 1, len(values)),
                                    values[:, columns.index(EMG_MAP[name])],
                                )
                                for name in EMG_CHANNELS
                            ]
                        )
                    )
            if joints:
                joint_trials.append(np.mean(joints, axis=0))
            if muscles:
                emg_trials.append(np.mean(muscles, axis=0))
            path = destination / "grf" / f"{sid}_trial{trial}.npz"
            notes = prepare_grf_reference(record, trial, path)
            counts = [0, 0]
            if path.exists():
                time, grf, valid = load_grf_reference(path)
                for side in (0, 1):
                    profiles = stance_profiles(time, grf[:, side], valid)
                    counts[side] = len(profiles)
                    if profiles:
                        grf_trials[side].append(np.mean(profiles, axis=0))
            audit.append(
                dict(
                    subject=sid,
                    trial=trial,
                    joint_steps=len(joints),
                    emg_steps=len(muscles),
                    complete_stances=counts,
                    grf_notes=notes,
                )
            )

        def mean_or_missing(values, shape):
            return np.mean(values, axis=0) if values else np.full(shape, np.nan)

        np.savez_compressed(
            destination / f"{sid}.npz",
            joint_angles_deg=mean_or_missing(joint_trials, (len(JOINT_NAMES), 100)),
            emg=mean_or_missing(emg_trials, (len(EMG_CHANNELS), 100)),
            grf_bw=np.stack([mean_or_missing(values, (100,)) for values in grf_trials]),
            joint_names=JOINT_NAMES,
            emg_names=EMG_CHANNELS,
            joint_trial_count=len(joint_trials),
            emg_trial_count=len(emg_trials),
            grf_trial_counts=[len(values) for values in grf_trials],
        )
    (destination / "reference_audit.json").write_text(json.dumps(audit, indent=2))
    return destination
