"""Explicit external motion selection does not alter the default protocol."""

import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from myoacross.rl.runners import evaluate


class Catalog:
    split = "test"
    subject_ids = ("S001",)

    def __iter__(self):
        return iter([SimpleNamespace(motions=[SimpleNamespace(motion_id="new_walk")])])


def test_explicit_catalog_accepts_new_motion_and_records_actual_selection(tmp_path):
    saved = {"config": {"data": {"dataset_version": "embodiment_motion_v1",
                                 "embodiment": "segment_scale", "motion_ids": ["kit_walk"]}}}
    collector = SimpleNamespace(finalize=lambda: None)
    with patch.object(evaluate.torch, "load", return_value=saved), \
         patch.object(evaluate, "_run_evaluation_batch", return_value=[]) as run, \
         patch.object(evaluate, "_summarize_trials", return_value=([], [], {})):
        evaluate.evaluate_policy(tmp_path / "unused.pt", tmp_path / "out",
                                 catalog=Catalog(), signal_collector=collector,
                                 allow_external_motions=True)
    assert run.call_args.kwargs["signal_collector"] is collector
    report = json.loads((tmp_path / "out/config.json").read_text())
    assert report["training_motion_ids"] == ["kit_walk"]
    assert report["motion_ids"] == ["new_walk"]


def test_default_selection_still_rejects_new_motion(tmp_path):
    saved = {"config": {"data": {"dataset_version": "embodiment_motion_v1",
                                 "embodiment": "segment_scale", "motion_ids": ["kit_walk"]}}}
    with patch.object(evaluate.torch, "load", return_value=saved):
        with pytest.raises(ValueError, match="checkpoint motion selection"):
            evaluate.evaluate_policy(tmp_path / "unused.pt", tmp_path / "out", motion_ids=["new_walk"])
