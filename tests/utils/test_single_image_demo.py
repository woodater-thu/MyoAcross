"""Keep the distributed demo model unchanged during pipeline preparation."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from myoacross.utils.image2msksim import prepare_single_image_demo as demo


@pytest.mark.parametrize("evaluate,headless,count", [(False, False, 1), (False, True, 1), (True, False, 64), (True, True, 64)])
def test_pipeline_uses_generated_output_and_preserves_demo(tmp_path, monkeypatch, evaluate, headless, count):
    source = tmp_path / "source"
    model = source / "embodiments/segment_scale" / demo.SUBJECT_ID
    model.mkdir(parents=True)
    source_xml = '<mujoco><compiler meshdir="." texturedir="."/><asset><mesh name="bone" file="meshes/sacrum.stl" scale="1.1 1.2 1.3"/></asset></mujoco>'
    (model / "model.xml").write_text(source_xml)
    original = json.dumps({"status": "passed", "baseline_path": "portable"})
    (model / "report.json").write_text(original)
    project = tmp_path / "project"
    for relative in ("tools/gmr_fit/.venv/bin/python", ".venv/bin/python"):
        python = project / relative
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
    monkeypatch.setattr(demo, "PROJECT_ROOT", project)
    commands = []
    monkeypatch.setattr(demo.subprocess, "run", lambda command, **kw: commands.append((command, kw)))
    demo.run_pipeline(source, tmp_path / "final.pt", evaluate=evaluate, headless=headless)
    output = project / "data/experiments/single_image_demo"
    settings = OmegaConf.load(output / "pipeline.yaml")
    assert settings.output_root == str(output)
    assert (model / "report.json").read_text() == original
    copied = output / "embodiments/segment_scale" / demo.SUBJECT_ID
    from musclemimic_models import get_xml_path
    tree = ET.parse(copied / "model.xml")
    compiler = tree.find("compiler")
    assert Path(compiler.get("meshdir")) == Path(get_xml_path("myofullbody")).resolve().parent.parent
    assert Path(compiler.get("texturedir")) == project / "data/myoacross/assets"
    assert (model / "model.xml").read_text() == source_xml
    assert tree.find("asset/mesh").get("scale") == "1.1 1.2 1.3"
    assert not (copied / "assets").exists()
    assert len(settings.motion_ids) == count
    assert settings.envs_per_subject == count
    assert "KIT__11__WalkingStraightForwards09_poses" in settings.motion_ids
    stages = [command[-1] for command, _ in commands if "--stage" in command]
    assert stages == (["retarget", "evaluate"] if evaluate or headless else ["retarget"])
    playback = [command for command, _ in commands if "myoacross.play" in command]
    assert len(playback) == (0 if headless else 1)
    if playback:
        command = playback[0]
        assert command[command.index("--camera") + 1] == "front-side"
        assert command[command.index("--checkpoint") + 1] == str(tmp_path / "final.pt")
        assert command[command.index("--model") + 1] == str(copied / "model.xml")
        assert command[command.index("--motion") + 1].endswith("KIT__11__WalkingStraightForwards09_poses/motion.npz")
    assert all(options["cwd"] == project and options["check"] for _, options in commands)
    with pytest.raises(FileExistsError):
        demo.run_pipeline(source, tmp_path / "final.pt", evaluate=evaluate, headless=headless)


@pytest.mark.parametrize("flags,count", [([], 1), (["--headless"], 1), (["--evaluate", "--headless"], 64)])
def test_cli_prepares_selected_motions(tmp_path, monkeypatch, flags, count):
    checkpoint = tmp_path / "final.pt"
    checkpoint.touch()
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(demo, "DATASET_ROOT", tmp_path / "dataset")
    archive = tmp_path / "KIT.tar.bz2"
    archive.touch()
    extracted, calls = [], []
    monkeypatch.setattr(demo, "extract_motions", lambda archive, paths: extracted.extend(paths))
    monkeypatch.setattr(demo, "run_pipeline", lambda *args, **kw: calls.append(kw))
    demo.main(["--checkpoint", str(checkpoint), "--archive", str(archive), *flags])
    assert len(extracted) == count
    assert calls[0]["evaluate"] == ("--evaluate" in flags)
    assert calls[0]["headless"] == ("--headless" in flags)


def test_no_display_fails_before_download(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "final.pt"
    checkpoint.touch()
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(demo, "download_kit", lambda *args: pytest.fail("must not download"))
    with pytest.raises(SystemExit):
        demo.main(["--checkpoint", str(checkpoint)])
    assert "--headless" in capsys.readouterr().err


def test_prepare_only_needs_neither_checkpoint_nor_display(tmp_path, monkeypatch):
    archive = tmp_path / "KIT.tar.bz2"
    archive.touch()
    monkeypatch.setattr(demo, "DATASET_ROOT", tmp_path / "dataset")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    extracted = []
    monkeypatch.setattr(demo, "extract_motions", lambda archive, paths: extracted.extend(paths))
    monkeypatch.setattr(demo, "run_pipeline", lambda *args, **kw: pytest.fail("must not run"))
    demo.main(["--prepare-only", "--evaluate", "--archive", str(archive)])
    assert len(extracted) == 64
