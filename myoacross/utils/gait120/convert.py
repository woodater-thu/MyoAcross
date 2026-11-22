"""Adapt labelled Gait120 TRC recordings to the official MoSh++ solver."""

from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

BODY_MARKERS = tuple(
    "LFHD RFHD LBHD RBHD C7 T10 CLAV STRN RBAK LSHO LUPA LELB LFRM LWRA LWRB LFIN RSHO RUPA RELB RFRM RWRA RWRB RFIN LASI RASI LPSI RPSI LTHI LKNE LTIB LANK LHEE LTOE RTHI RKNE RTIB RANK RHEE RTOE".split()
)


def read_trc(path):
    """Read positions without changing their declared physical units."""
    lines = Path(path).read_text().splitlines()
    header = dict(zip(lines[1].split(), lines[2].split()))
    labels = [part.strip() for part in lines[3].split("\t")[2:] if part.strip()]
    samples = [line.split() for line in lines[5:] if line.strip()]
    corrections = []
    # S068 has a truncated final static row (RFHD coordinates are absent).
    if "Static" in Path(path).parts and len(samples[-1]) != len(samples[0]):
        if len(samples) != int(header["NumFrames"]) or any(
            len(row) != len(samples[0]) for row in samples[:-1]
        ):
            raise ValueError(f"Unexpected truncated TRC rows: {path}")
        samples = samples[:-1]
        corrections.append("removed_incomplete_final_static_frame")
    values = np.asarray(samples, dtype=float)
    # Eight released static headers name four markers whose columns are absent.
    extras = ("LTHIA", "LTIBA", "RTHIA", "RTIBA")
    if "Static" in Path(path).parts and len(labels) == 43 and values.shape[1] == 119:
        if tuple(n for n in labels if n not in extras) != BODY_MARKERS:
            raise ValueError(f"Unknown static marker-column mismatch: {path}")
        labels = [n for n in labels if n not in extras]
        corrections.append("removed_four_header_only_leg_markers")
    if header["Units"] != "mm":
        raise ValueError(f"Expected Gait120 TRC units mm: {path}")
    if values.shape != (
        int(header["NumFrames"])
        - int("removed_incomplete_final_static_frame" in corrections),
        2 + 3 * len(labels),
    ):
        raise ValueError(f"TRC header and samples disagree: {path}")
    time = values[:, 1]
    rate = float(header["DataRate"])
    if len(time) < 2 or not np.allclose(np.diff(time), 1 / rate, atol=1e-6):
        raise ValueError(f"TRC requires uniform time samples: {path}")
    markers = values[:, 2:].reshape(len(time), len(labels), 3)
    markers[(markers == 0).all(axis=2)] = np.nan
    return dict(
        markers=markers,
        labels=np.array(labels),
        frame_rate=rate,
        time=time,
        unit=header["Units"],
        corrections=corrections,
    )


def merge_trial_steps(steps):
    """Join Step01 and Step02 on their shared original timestamp."""
    steps = sorted(steps, key=lambda step: step["step"])
    if len(steps) == 1 and steps[0]["step"] in (1, 2):
        data = read_trc(steps[0]["path"])
        return dict(
            data,
            source_step=np.full(len(data["time"]), steps[0]["step"], dtype=int),
            source_frame_index=np.arange(len(data["time"])),
        )
    if [step["step"] for step in steps] != [1, 2]:
        raise ValueError("A continuous trial requires Step01 and Step02")
    first, second = [read_trc(step["path"]) for step in steps]
    if first["frame_rate"] != second["frame_rate"] or not np.array_equal(
        first["labels"], second["labels"]
    ):
        raise ValueError("Step marker labels and frame rates must match")
    if not np.isclose(first["time"][-1], second["time"][0], atol=1e-8, rtol=0):
        raise ValueError("Trial steps must have one shared endpoint")
    if not np.allclose(
        first["markers"][-1], second["markers"][0], atol=1e-5, rtol=0, equal_nan=True
    ):
        raise ValueError("Marker positions differ at the shared endpoint")
    n, m = len(first["time"]), len(second["time"])
    return dict(
        time=np.r_[first["time"], second["time"][1:]],
        markers=np.concatenate([first["markers"], second["markers"][1:]]),
        labels=first["labels"],
        frame_rate=first["frame_rate"],
        unit=first["unit"],
        source_step=np.r_[np.ones(n, dtype=int), np.full(m - 1, 2, dtype=int)],
        source_frame_index=np.r_[np.arange(n), np.arange(1, m)],
    )


def read_demographics(path):
    """Read the released workbook without adding an Excel dependency."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = [
            "".join(n.itertext())
            for n in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(
                "m:si", ns
            )
        ]
        rows = []
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall(
            ".//m:sheetData/m:row", ns
        ):
            cells = {}
            for cell in row:
                value = cell.find("m:v", ns)
                if value is not None:
                    text = value.text
                    cells[re.sub(r"\d", "", cell.get("r"))] = (
                        strings[int(text)] if cell.get("t") == "s" else text
                    )
            rows.append(cells)
    subjects = {}
    for row in rows[1:]:
        subjects[row["A"]] = dict(
            age_years=int(float(row["B"])),
            gender={"M": "male", "F": "female"}[row["C"]],
            height_m=float(row["D"]),
            body_mass_kg=float(row["E"]),
        )
    return dict(subjects=subjects)


def collect_subject_inputs(dataset_root, subject_ids=None):
    root = Path(dataset_root)
    metadata = read_demographics(root / "Subject_Demographics.xlsx")
    selected = list(metadata["subjects"]) if subject_ids is None else subject_ids
    records = []
    for subject in selected:
        if subject not in metadata["subjects"]:
            raise ValueError(f"Unknown Gait120 subject: {subject}")
        sources = list(root.glob(f"Gait120_*/{subject}"))
        if len(sources) != 1:
            raise ValueError(f"Expected one source directory for {subject}")
        folder = sources[0]
        if metadata["subjects"][subject]["gender"] != "male":
            raise ValueError("This adapter requires the all-male Gait120 release")
        motions = []
        for path in sorted(folder.glob("MotionCapture/LevelWalking/TRC/*/*.trc")):
            trial = re.fullmatch(r"Trial(\d+)", path.parent.name, re.I)
            step = re.fullmatch(r"Step(\d+)", path.stem, re.I)
            if trial is None or step is None:
                raise ValueError(f"Unexpected Gait120 motion identity: {path}")
            trial_id, step_id = int(trial[1]), int(step[1])
            motions.append(
                dict(
                    path=str(path.resolve()),
                    trial=trial_id,
                    step=step_id,
                    motion_id=f"LevelWalking_Trial{trial_id:02d}_Step{step_id:02d}",
                )
            )
        if not motions:
            raise ValueError(f"No level-walking motions for {subject}")
        trials = []
        for trial in sorted({motion["trial"] for motion in motions}):
            steps = [motion for motion in motions if motion["trial"] == trial]
            if sorted(step["step"] for step in steps) not in ([1], [2], [1, 2]):
                raise ValueError(f"Unexpected steps in Trial {trial}")
            trials.append(
                dict(
                    trial=trial,
                    motion_id=f"LevelWalking_Trial{trial:02d}_AllSteps",
                    source_steps=steps,
                )
            )
        records.append(
            dict(
                subject_id=subject,
                source_dir=str(folder.resolve()),
                metadata=metadata["subjects"][subject],
                motions=trials,
            )
        )
    return records


def prepare_marker_inputs(record, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    paths = []
    audit = []
    inputs = [
        (m["motion_id"], merge_trial_steps(m["source_steps"]))
        for m in record["motions"]
    ]
    inputs += [
        (f"Static_{i:02d}", read_trc(p))
        for i, p in enumerate(
            sorted(Path(record["source_dir"]).glob("MotionCapture/Static/TRC/*.trc"))
        )
    ]
    for name, data in inputs:
        ids = [i for i, label in enumerate(data["labels"]) if label in BODY_MARKERS]
        if len(ids) != 39:
            raise ValueError(f"Expected the 39 Gait120 body markers: {name}")
        path = destination / f"{name}.npz"
        mapping = {
            key: data[key]
            for key in ("source_step", "source_frame_index")
            if key in data
        }
        np.savez(
            path,
            markers=data["markers"][:, ids],
            labels=data["labels"][ids],
            frame_rate=data["frame_rate"],
            source_time_sec=data["time"],
            **mapping,
        )
        audit.append(
            dict(
                recording=name,
                frames=len(data["time"]),
                corrections=data.get("corrections", []),
            )
        )
        paths.append(path)
    (destination / "input_audit.json").write_text(json.dumps(audit, indent=2))
    return paths
