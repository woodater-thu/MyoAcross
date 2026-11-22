"""Select adapted execution before importing task implementations."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from myoacross.rl import run
from myoacross.rl.runners.multi_embodiment import backend


class TaskBackendTests(TestCase):
    def test_generate_and_single_body_train_do_not_inspect_backend(self):
        with patch.object(backend.importlib.util, "find_spec") as find:
            run._prepare_task_backend("generate", {}, Path("config.yaml"))
            run._prepare_task_backend(
                "train", {"training": {"embodiments_per_rank": 1}}, Path("config.yaml")
            )
            find.assert_not_called()

    def test_multi_body_train_preserves_resume_and_exit_status(self):
        with TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "_src").mkdir()
            (source / "_src/sensor.py").write_text("original")
            spec = SimpleNamespace(origin=str(source / "__init__.py"))
            with (
                patch.object(backend.importlib.util, "find_spec", return_value=spec),
                patch.object(backend, "run_with_backend", return_value=7) as child,
            ):
                with self.assertRaises(SystemExit) as stopped:
                    run._prepare_task_backend(
                        "train",
                        {"training": {"embodiments_per_rank": 8}},
                        source / "config.yaml",
                        resume=source / "resume.pt",
                    )
            self.assertEqual(stopped.exception.code, 7)
            self.assertEqual(
                child.call_args.args[0],
                [
                    sys.executable,
                    "-m",
                    "myoacross",
                    "rl",
                    "train",
                    "--config",
                    str(source / "config.yaml"),
                    "--resume",
                    str(source / "resume.pt"),
                ],
            )

    def test_evaluate_relaunches_without_resume(self):
        with TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "_src").mkdir()
            (source / "_src/sensor.py").write_text("original")
            with (
                patch.object(
                    backend.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(source / "__init__.py")),
                ),
                patch.object(backend, "run_with_backend", return_value=0) as child,
            ):
                with self.assertRaises(SystemExit):
                    run._prepare_task_backend("evaluate", {}, source / "config.yaml")
            self.assertEqual(child.call_args.args[0][4], "evaluate")
            self.assertNotIn("--resume", child.call_args.args[0])

    def test_adapted_process_continues_without_relaunch(self):
        with TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "_src").mkdir()
            (source / "_src/sensor.py").write_text(backend.MARKER + " = True")
            with (
                patch.object(
                    backend.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(source / "__init__.py")),
                ),
                patch.object(backend, "run_with_backend") as child,
            ):
                run._prepare_task_backend(
                    "train",
                    {"training": {"embodiments_per_rank": 8}},
                    source / "config.yaml",
                )
                child.assert_not_called()
