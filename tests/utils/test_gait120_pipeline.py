"""Final Gait120 protocol boundaries, without running fitting or simulation."""

import numpy as np
import pytest


def test_kit_selection_is_registered_medium_set():
    from myoacross.utils.gait120.motions import select_kit_motions

    motions = select_kit_motions()
    assert len(motions) == 8
    assert {m.category for m in motions} == {"walk_medium"}
    assert "KIT__3__walking_medium04_poses" in {m.motion_id for m in motions}


def test_complete_stance_excludes_clipped_and_invalid_contacts():
    from myoacross.utils.gait120.signals import stance_profiles

    time = np.arange(120) * 0.01
    force = np.zeros(120)
    force[:25] = 1
    force[35:75] = np.sin(np.linspace(0, np.pi, 40))
    force[90:] = 1
    valid = np.ones(120, dtype=bool)
    assert len(stance_profiles(time, force, valid)) == 1
    valid[50] = False
    assert len(stance_profiles(time, force, valid)) == 0


def test_merge_single_available_step_keeps_source_indices(monkeypatch):
    from myoacross.utils.gait120 import convert

    data = dict(
        time=np.arange(3) * 0.01,
        markers=np.ones((3, 1, 3)),
        labels=np.array(["LASI"]),
        frame_rate=100,
        unit="mm",
    )
    monkeypatch.setattr(convert, "read_trc", lambda _: data)
    merged = convert.merge_trial_steps([dict(path="unused", step=1)])
    np.testing.assert_array_equal(merged["source_step"], [1, 1, 1])
    np.testing.assert_array_equal(merged["source_frame_index"], [0, 1, 2])


def test_aggregation_retains_seven_completed_motions(tmp_path):
    from myoacross.utils.gait120.analysis import aggregate_subject
    from myoacross.utils.gait120.signals import MUSCLE_NAMES, JOINT_NAMES

    (tmp_path / "signals").mkdir()
    records = []
    x = np.linspace(0, 1, 100)
    for i in range(8):
        motion = f"motion{i}"
        records.append(dict(motion=motion, completed=i != 7))
        if i == 7:
            continue  # A failed episode must not be read or counted as a zero waveform.
        np.savez(
            tmp_path / f"{motion}_cycles.npz",
            activation=np.tile((i + x)[None, None, :], (1, len(MUSCLE_NAMES), 1)),
            joint_angles_deg=np.zeros((1, len(JOINT_NAMES), 100)),
            actuator_torque_nm=np.zeros((1, len(JOINT_NAMES), 100)),
            names=MUSCLE_NAMES,
        )
        np.savez(
            tmp_path / "signals" / f"{motion}.npz",
            force_interval_end_sim_sec=np.arange(100) * 0.01,
            grf_bw=np.zeros((100, 2)),
        )
    result = aggregate_subject(tmp_path, records)
    assert result["valid_motions"] == result["completed_motions"] == 7
    assert result["total_motions"] == 8
    np.testing.assert_allclose(result["activation"][0], 3 + x)
    assert result["activation"].shape == (12, 100)
    result = aggregate_subject(tmp_path, [dict(motion="missing", completed=False)])
    assert result["valid_motions"] == 0
    assert np.isnan(result["activation"]).all()


def test_shape_cache_routing_restores_on_exception(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from myoacross.utils.gait120.retarget import fitted_shape_cache
    import myoacross.retarget

    calls = []

    def shape(*args, **kwargs):
        calls.append(kwargs)
        return "cache"

    module = SimpleNamespace(ensure_fitted_shape=shape)
    monkeypatch.setitem(sys.modules, "myoacross.retarget.retargeter", module)
    monkeypatch.setattr(myoacross.retarget, "retargeter", module, raising=False)
    with pytest.raises(RuntimeError):
        with fitted_shape_cache(tmp_path):
            assert module.ensure_fitted_shape("model") == "cache"
            assert calls == [dict(cache_root=tmp_path)]
            raise RuntimeError("solver failure")
    assert module.ensure_fitted_shape is shape


def test_truncated_static_final_row_does_not_shift_marker_columns(tmp_path):
    from myoacross.utils.gait120.convert import read_trc

    path = tmp_path / "Static" / "TRC" / "Trial01.trc"
    path.parent.mkdir(parents=True)
    path.write_text(
        "PathFileType\t4\t(X/Y/Z)\tTrial01.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n100\t100\t3\t2\tmm\nFrame#\tTime\tLFHD\t\t\tRFHD\t\t\n\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n1\t0\t1\t2\t3\t4\t5\t6\n2\t0.01\t1\t2\t3\t4\t5\t6\n3\t0.02\t1\t2\t3\t\t\t\n"
    )
    original = path.read_bytes()
    data = read_trc(path)
    assert data["markers"].shape == (2, 2, 3)
    assert data["corrections"] == ["removed_incomplete_final_static_frame"]
    assert path.read_bytes() == original


def test_header_only_static_leg_markers_are_removed(tmp_path):
    from myoacross.utils.gait120.convert import read_trc, BODY_MARKERS

    path = tmp_path / "Static" / "TRC" / "Trial01.trc"
    path.parent.mkdir(parents=True)
    labels = (*BODY_MARKERS, "LTHIA", "LTIBA", "RTHIA", "RTIBA")
    header = "Frame#\tTime\t" + "\t\t\t".join(labels)
    rows = [
        f"{i + 1}\t{i * 0.01}\t" + "\t".join(str(v + 1) for v in range(117))
        for i in range(2)
    ]
    path.write_text(
        "PathFileType\t4\t(X/Y/Z)\tTrial01.trc\nDataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n100\t100\t2\t43\tmm\n"
        + header
        + "\ncoordinates\n"
        + "\n".join(rows)
    )
    data = read_trc(path)
    assert tuple(data["labels"]) == BODY_MARKERS
    np.testing.assert_array_equal(data["markers"][0, 0], [1, 2, 3])


def test_checkpoint_iteration_is_checked(tmp_path):
    import torch
    from myoacross.utils.gait120.run import evaluation_output

    checkpoint = tmp_path / "final.pt"
    torch.save(dict(iteration=20000), checkpoint)
    with pytest.raises(ValueError, match="15000"):
        evaluation_output(
            dict(output_root=str(tmp_path), checkpoint_iteration=15000),
            "film",
            checkpoint,
        )


def test_evaluation_uses_one_catalog_and_eight_envs_per_subject(monkeypatch, tmp_path):
    from myoacross.utils.gait120 import run, motions, evaluation
    from myoacross.rl.runners import evaluate

    calls = []
    catalog = object()
    monkeypatch.setattr(
        run,
        "load_records",
        lambda _: [dict(subject_id="S011"), dict(subject_id="S065")],
    )
    monkeypatch.setattr(motions, "build_catalog", lambda *args: catalog)
    monkeypatch.setattr(
        run, "evaluation_output", lambda c, scheme, ckpt: tmp_path / scheme
    )
    monkeypatch.setattr(evaluation, "Gait120SignalCollector", lambda path: path)

    def fake_evaluate(checkpoint, destination, **kwargs):
        destination.mkdir()
        calls.append((checkpoint, kwargs))

    monkeypatch.setattr(evaluate, "evaluate_policy", fake_evaluate)
    run.run_evaluate(
        dict(
            output_root=str(tmp_path),
            policies=dict(film="film.pt", v0="v0.pt"),
            device="cuda:0",
            seed=42,
        )
    )
    assert len(calls) == 2
    for _, kwargs in calls:
        assert kwargs["catalog"] is catalog
        assert kwargs["envs_per_subject"] == 8
        assert kwargs["allow_external_motions"] is True
        assert kwargs["bundle_root"] is None


def test_fitting_runs_stage_i_only(monkeypatch, tmp_path):
    import pickle, sys
    from types import ModuleType
    from myoacross.utils.gait120 import fitting

    modules = {}
    for name in (
        "moshpp",
        "moshpp.chmosh",
        "moshpp.mosh_head",
        "moshpp.marker_layout",
        "moshpp.marker_layout.create_marker_layout_for_mocaps",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    calls = []

    class MoSh:
        def __init__(self, dict_cfg):
            self.settings = dict_cfg

        def mosh_stagei(self, method):
            calls.append(self.settings)
            with open(self.settings["dirs"]["stagei_fname"], "wb") as handle:
                pickle.dump(dict(betas=np.arange(16)), handle)

        def mosh_stageii(self, method):
            pytest.fail("The final protocol must not fit Stage II")

    modules["moshpp.mosh_head"].MoSh = MoSh
    modules["moshpp.chmosh"].mosh_stagei = object()
    modules[
        "moshpp.marker_layout.create_marker_layout_for_mocaps"
    ].marker_labels_to_marker_layout = lambda *args, **kwargs: None
    monkeypatch.setattr(
        fitting, "prepare_marker_inputs", lambda *args: [tmp_path / "markers.npz"]
    )
    config = dict(
        output_root=str(tmp_path),
        seed=42,
        moshpp={
            n: "unused"
            for n in (
                "model_path",
                "pose_body_prior",
                "pose_hand_prior",
                "head_marker_corr",
                "support_dir",
            )
        },
    )
    fitting.fit_subject_shape(
        dict(
            subject_id="S011",
            metadata=dict(gender="male"),
            motions=[dict(motion_id="trial1"), dict(motion_id="trial2")],
        ),
        config,
    )
    assert len(calls) == 1
    assert not calls[0]["moshpp"]["optimize_dynamics"]
    with np.load(tmp_path / "shapes/S011/shape_identity.npz") as data:
        np.testing.assert_array_equal(data["betas"], np.arange(16))
