"""Gait120 input boundaries and physiological metrics."""

from pathlib import Path
import numpy as np
import pytest


def test_trc_units_labels_and_missing_markers(tmp_path):
    from myoacross.utils.gait120.convert import read_trc

    path = tmp_path / "Step01.trc"
    path.write_text(
        "PathFileType\t4\t(X/Y/Z)\tStep01.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n100\t100\t2\t2\tmm\nFrame#\tTime\tLASI\t\t\tRASI\t\t\n\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n1\t0.0\t1\t2\t3\t0\t0\t0\n2\t0.01\t4\t5\t6\t7\t8\t9\n"
    )
    data = read_trc(path)
    assert data["unit"] == "mm"
    assert data["labels"].tolist() == ["LASI", "RASI"]
    assert data["markers"].shape == (2, 2, 3)
    assert np.isnan(data["markers"][0, 1]).all()
    np.testing.assert_array_equal(data["markers"][1, 0], [4, 5, 6])


def test_mcos_plain_mat_reader(tmp_path):
    from scipy.io import savemat
    from myoacross.utils.gait120.mcos import resolve_tables

    path = tmp_path / "plain.mat"
    savemat(path, {"LevelWalking": {"AvailableTrialIdx": np.arange(1, 6)}})
    result = resolve_tables(path)
    np.testing.assert_array_equal(
        result["LevelWalking"]["AvailableTrialIdx"].ravel(), np.arange(1, 6)
    )


def test_grf_warp_reduction_preserves_sides_worlds_and_sign(tmp_path, monkeypatch):
    import warp as wp
    from myoacross.utils.gait120.grf import accumulate_foot_forces

    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp"))
    wp.init()
    geoms = wp.array(
        np.array([[0, 1], [2, 0], [0, 3]], np.int32), dtype=wp.vec2i, device="cpu"
    )
    force = wp.array(
        np.array(
            [[1, 2, 100, 0, 0, 0], [-1, -2, -200, 0, 0, 0], [0, 0, 999, 0, 0, 0]],
            np.float32,
        ),
        dtype=wp.spatial_vector,
        device="cpu",
    )
    output = wp.zeros((2, 2), dtype=wp.vec3, device="cpu")
    wp.launch(
        accumulate_foot_forces,
        dim=3,
        inputs=[
            wp.array([3], dtype=int, device="cpu"),
            geoms,
            wp.array([0, 1, 0], dtype=int, device="cpu"),
            force,
            wp.array([-1, 1, 0, -1], dtype=int, device="cpu"),
            wp.array([1, 0, 0, 0], dtype=int, device="cpu"),
        ],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy()[0, 1], [1, 2, 100])
    np.testing.assert_array_equal(output.numpy()[1, 0], [1, 2, 200])


def test_named_stage_dispatches_configured_interpreter(tmp_path, monkeypatch):
    from myoacross.utils.gait120 import run
    import json

    config = dict(
        output_root=str(tmp_path),
        moshpp={"python": "mosh-python"},
        environments={"references": "analysis-python"},
    )
    monkeypatch.setattr(run, "load_pipeline_config", lambda _: config)
    monkeypatch.setattr(
        run, "run_references", lambda _: pytest.fail("Called the root interpreter")
    )
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        worker_config = json.loads(
            Path(command[command.index("--config") + 1]).read_text()
        )
        assert worker_config == config

    monkeypatch.setattr(run.subprocess, "run", execute)
    run.main(["--config", str(tmp_path / "config.yaml"), "--stage", "references"])
    assert commands[0][0] == "analysis-python"


def test_raw_grf_maps_cop_in_meters_to_synchronized_feet(tmp_path):
    from myoacross.utils.gait120.references import assign_force_plates
    from myoacross.utils.gait120.convert import read_trc

    trc = tmp_path / "Step01.trc"
    header = "PathFileType\t4\t(X/Y/Z)\tStep01.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n100\t100\t3\t4\tmm\nFrame#\tTime\tLHEE\t\t\tLTOE\t\t\tRHEE\t\t\tRTOE\t\t\n\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\tX4\tY4\tZ4\n"
    rows = [
        f"{i + 1}\t{i * 0.01}\t100\t20\t100\t100\t20\t300\t300\t20\t100\t300\t20\t300\n"
        for i in range(3)
    ]
    trc.write_text(header + "".join(rows))
    mot = tmp_path / "Step01.mot"
    columns = ["time"] + [
        f"ground_force{plate}_{axis}" for plate in (1, 2) for axis in ("vy", "px", "pz")
    ]
    mot.write_text(
        "endheader\n"
        + "\t".join(columns)
        + "\n"
        + "".join(
            (
                f"{i * 0.005}\t{(20 if i == 4 else 100)}\t.3\t.2\t200\t.1\t.2\n"
                for i in range(5)
            )
        )
    )
    result = assign_force_plates(mot, read_trc(trc), 60.0)
    np.testing.assert_array_equal(result["plate_sides"], np.tile([1, 0], (3, 1)))
    np.testing.assert_allclose(
        result["computed_grf"], [[200, 100], [200, 100], [200, 20]]
    )
    np.testing.assert_allclose(result["time_sec"], [0, 0.01, 0.02])


def _write_grf_step(source, step, time, *, switch=False):
    trc = source / f"MotionCapture/LevelWalking/TRC/Trial01/Step{step:02d}.trc"
    mot = source / f"MotionCapture/LevelWalking/MOT/Trial01/Step{step:02d}.mot"
    trc.parent.mkdir(parents=True, exist_ok=True)
    mot.parent.mkdir(parents=True, exist_ok=True)
    trc.write_text(
        f"PathFileType\t4\t(X/Y/Z)\tStep.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n100\t100\t{len(time)}\t4\tmm\nFrame#\tTime\tLHEE\t\t\tLTOE\t\t\tRHEE\t\t\tRTOE\t\t\n\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\tX4\tY4\tZ4\n"
        + "".join(
            (
                f"{i + 1}\t{t:.3f}\t100\t20\t100\t100\t20\t300\t300\t20\t100\t300\t20\t300\n"
                for (i, t) in enumerate(time)
            )
        )
    )
    mot.write_text(
        "endheader\ntime ground_force1_vy ground_force1_px ground_force1_pz ground_force2_vy ground_force2_px ground_force2_pz\n"
        + "".join(
            (
                f"{t:.3f} 200 {(0.1 if switch and i > 0 else 0.3)} .2 {(100 if t < 0.5 else 0)} .1 .2\n"
                for (i, t) in enumerate(time)
            )
        )
    )
    return (trc, mot)


def test_grf_reference_reports_missing_plate_data_without_zero_reference(tmp_path):
    from myoacross.utils.gait120.references import prepare_grf_reference

    record = dict(
        subject_id="S001",
        source_dir=str(tmp_path / "source"),
        metadata={"body_mass_kg": 60.0},
    )
    target = tmp_path / "reference.npz"
    notes = prepare_grf_reference(record, 1, target)
    assert notes[0]["reason"] == "missing_force_plate_data"
    assert not target.exists()


def test_grf_ambiguous_contact_is_invalid_not_silently_reassigned(tmp_path):
    from myoacross.utils.gait120.references import assign_force_plates
    from myoacross.utils.gait120.convert import read_trc

    (trc, mot) = _write_grf_step(tmp_path, 1, np.arange(3) * 0.01, switch=True)
    data = assign_force_plates(mot, read_trc(trc), 60.0)
    assert not data["computed_grf_valid"].any()
    assert data["segments"][0]["sides"] == [0, 1]


def test_pipeline_dispatch_preserves_virtual_environment_interpreter():
    import subprocess
    from myoacross.utils.gait120.run import load_pipeline_config
    from myoacross.utils.paths import PROJECT_ROOT

    config = load_pipeline_config(
        PROJECT_ROOT / "myoacross/configs/experiments/gait120/pipeline.yaml"
    )
    prefix = subprocess.check_output(
        [
            config["environments"]["references"],
            "-B",
            "-c",
            "import sys; print(sys.prefix)",
        ],
        text=True,
    ).strip()
    assert Path(prefix).resolve() == (PROJECT_ROOT / ".venv").resolve()


def test_merge_trial_steps_deduplicates_endpoint_and_keeps_source_clock(tmp_path):
    from myoacross.utils.gait120.convert import merge_trial_steps

    (p1, _) = _write_grf_step(tmp_path, 1, np.array([5.0, 5.01, 5.02]))
    (p2, _) = _write_grf_step(tmp_path, 2, np.array([5.02, 5.03, 5.04]))
    data = merge_trial_steps([dict(path=str(p1), step=1), dict(path=str(p2), step=2)])
    np.testing.assert_allclose(data["time"], [5.0, 5.01, 5.02, 5.03, 5.04])
    np.testing.assert_array_equal(data["source_step"], [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(data["source_frame_index"], [0, 1, 2, 1, 2])
    assert data["markers"].shape == (5, 4, 3)


def test_merge_trial_steps_rejects_time_gap(tmp_path):
    from myoacross.utils.gait120.convert import merge_trial_steps

    (p1, _) = _write_grf_step(tmp_path, 1, np.array([5.0, 5.01, 5.02]))
    (p2, _) = _write_grf_step(tmp_path, 2, np.array([5.04, 5.05, 5.06]))
    with pytest.raises(ValueError, match="shared endpoint"):
        merge_trial_steps([dict(path=str(p1), step=1), dict(path=str(p2), step=2)])
