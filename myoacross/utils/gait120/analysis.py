"""Compare each individual's completed KIT motions with their human references."""

import csv
import json
from pathlib import Path
import numpy as np
from .signals import (
    EMG_CHANNELS,
    JOINT_NAMES,
    SAGITTAL_JOINTS,
    pearson,
    stance_profiles,
)


def aggregate_subject(folder, records):
    """Weight valid motions equally; retain failed motions in completion counts."""
    folder = Path(folder)
    activation, angles, moments, grf = [], [], [], [[], []]
    valid_ids = []
    for record in records:
        if not record["completed"]:
            continue
        motion = record["motion"]
        with np.load(folder / f"{motion}_cycles.npz") as data:
            if len(data["activation"]) == 0:
                continue
            names = list(data["names"])
            indices = [
                names.index("soleus_r" if n == "soleus_lateral" else n)
                for n in EMG_CHANNELS
            ]
            activation.append(data["activation"].mean(0)[indices])
            angles.append(data["joint_angles_deg"].mean(0))
            moments.append(data["actuator_torque_nm"].mean(0))
        valid_ids.append(motion)
        with np.load(folder / "signals" / f"{motion}.npz") as data:
            for side in (0, 1):
                profiles = stance_profiles(
                    data["force_interval_end_sim_sec"], data["grf_bw"][:, side]
                )
                if profiles:
                    grf[side].append(np.mean(profiles, axis=0))

    def mean_or_missing(values, shape):
        return np.mean(values, axis=0) if values else np.full(shape, np.nan)

    return dict(
        activation=mean_or_missing(activation, (12, 100)),
        joint_angles_deg=mean_or_missing(angles, (len(JOINT_NAMES), 100)),
        actuator_torque_nm=mean_or_missing(moments, (len(JOINT_NAMES), 100)),
        grf_bw=np.stack([mean_or_missing(v, (100,)) for v in grf]),
        valid_motion_ids=valid_ids,
        valid_motions=len(valid_ids),
        grf_valid_motions=[len(v) for v in grf],
        completed_motions=sum(r["completed"] for r in records),
        total_motions=len(records),
    )


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(
        n=len(values),
        mean=float(values.mean()),
        sd=float(values.std()),
        median=float(np.median(values)),
        q25=float(np.quantile(values, 0.25)),
        q75=float(np.quantile(values, 0.75)),
        min=float(values.min()),
        max=float(values.max()),
    )


def write_rows(path, rows, fields):
    with Path(path).open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(profile_root, reference_root, destination):
    """Report all subjects, effective motion counts, and 12-channel correlations."""
    profile_root, reference_root, destination = map(
        Path, (profile_root, reference_root, destination)
    )
    destination.mkdir(parents=True, exist_ok=False)
    audit = json.loads((profile_root / "signal_audit.json").read_text())
    subject_rows, channel_rows = [], []
    for sid in sorted({r["subject"] for r in audit}):
        records = [r for r in audit if r["subject"] == sid]
        model = aggregate_subject(profile_root / sid, records)
        with np.load(reference_root / f"{sid}.npz") as data:
            human = {key: data[key] for key in ("emg", "joint_angles_deg", "grf_bw")}
        metrics = {}
        for domain, names, simulated, measured in (
            ("emg", EMG_CHANNELS, model["activation"], human["emg"]),
            (
                "joint_angle",
                JOINT_NAMES,
                model["joint_angles_deg"],
                human["joint_angles_deg"],
            ),
            ("grf_stance", ("left", "right"), model["grf_bw"], human["grf_bw"]),
        ):
            scores = [pearson(a, b) for a, b in zip(simulated, measured)]
            metrics[domain] = scores
            for i, name in enumerate(names):
                rmse = (
                    float(np.sqrt(np.mean((simulated[i] - measured[i]) ** 2)))
                    if domain == "grf_stance"
                    else ""
                )
                channel_rows.append(
                    dict(
                        subject=sid,
                        domain=domain,
                        channel=name,
                        pearson=scores[i],
                        valid_motions=model["grf_valid_motions"][i]
                        if domain == "grf_stance"
                        else model["valid_motions"],
                        rmse_bw=rmse,
                    )
                )
        subject_rows.append(
            dict(
                subject=sid,
                total_motions=model["total_motions"],
                completed_motions=model["completed_motions"],
                valid_motions=model["valid_motions"],
                completion_rate=model["completed_motions"] / model["total_motions"],
                grf_left_valid_motions=model["grf_valid_motions"][0],
                grf_right_valid_motions=model["grf_valid_motions"][1],
                emg12=float(np.mean(metrics["emg"])),
                joint_sagittal=float(
                    np.mean(
                        [
                            metrics["joint_angle"][JOINT_NAMES.index(n)]
                            for n in SAGITTAL_JOINTS
                        ]
                    )
                ),
                hip_rotation=float(
                    np.mean(
                        [
                            metrics["joint_angle"][JOINT_NAMES.index(n)]
                            for n in ("hip_rotation_r", "hip_rotation_l")
                        ]
                    )
                ),
                grf_stance=float(np.mean(metrics["grf_stance"])),
                grf_rmse_bw=float(
                    np.sqrt(
                        np.mean((model["grf_bw"] - human["grf_bw"]) ** 2, axis=1)
                    ).mean()
                ),
            )
        )
        np.savez_compressed(
            destination / f"{sid}_profiles.npz",
            **model,
            emg_names=EMG_CHANNELS,
            joint_names=JOINT_NAMES,
        )
    write_rows(destination / "per_subject.csv", subject_rows, list(subject_rows[0]))
    write_rows(destination / "per_channel.csv", channel_rows, list(channel_rows[0]))
    summary = dict(
        subjects=len(subject_rows),
        completed_motions=sum(r["completed"] for r in audit),
        total_motions=len(audit),
        completion_rate=sum(r["completed"] for r in audit) / len(audit),
        metrics={
            key: distribution([r[key] for r in subject_rows])
            for key in (
                "emg12",
                "joint_sagittal",
                "hip_rotation",
                "grf_stance",
                "grf_rmse_bw",
            )
        },
        torque_correlation="unavailable_without_comparable_human_joint_moment_reference",
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    (destination / "protocol.json").write_text(
        json.dumps(
            dict(
                averaging="cycles within motion; equal available completed motions; equal subjects",
                emg="12 processed EMG channels versus activation; two soleus channels share model soleus",
                cycles="complete right heel-strike cycles, no optimized phase shifts",
                grf="complete observed stance, 0.05 BW contact threshold, duration >=0.2 s; vertical BW",
                human="available steps within trial, then equal available trials",
                missing="retain subjects and valid-motion counts; undefined correlations remain missing",
                joint_sign="MM-EvalKit INVERTED_JOINTS",
            ),
            indent=2,
        )
    )
    return destination
