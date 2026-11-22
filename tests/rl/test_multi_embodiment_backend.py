"""The source adapter must reject drift without publishing a partial package."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from myoacross.rl.runners.multi_embodiment.backend import (
    adapt_sensor_source,
    adapt_physics_source,
    prepare_backend,
)


SENSOR = """
def first(site_size: wp.array[wp.vec3]):
    return site_size[refid]
def second(site_size: wp.array[wp.vec3]):
    return site_size[objid]
def third(site_size: wp.array[wp.vec3]):
    return site_size[objid]
f(m.site_size, x)
g(m.site_size, y)
h(m.site_size, z)
"""


PHYSICS = {
    "derivative": "def f(m, d):\n    wp.launch(_qderiv_tendon_damping, inputs=[m.tendon_damping, m.tendon_dampingpoly], outputs=[d])\n",
    "smooth": "def f(m, d):\n    wp.launch(_tendon_armature, inputs=[m.tendon_armature], outputs=[d])\n    wp.launch(_tendon_bias_coef, inputs=[m.tendon_armature], outputs=[d])\n    wp.launch(_tendon_bias_qfrc, inputs=[m.tendon_armature], outputs=[d])\n",
    "passive": "def f(m, d):\n    wp.launch(_spring_damper_tendon_passive, inputs=[m.tendon_stiffness, m.tendon_stiffnesspoly, m.tendon_damping, m.tendon_dampingpoly], outputs=[d])\n",
}


def write_physics(source):
    for name, text in PHYSICS.items():
        (source / f"_src/{name}.py").write_text(text)


class BackendAdapterTests(TestCase):
    def test_validated_adapter_preserves_original(self):
        with TemporaryDirectory() as temp:
            source, output = Path(temp) / "original", Path(temp) / "adapter"
            (source / "_src").mkdir(parents=True)
            write_physics(source)
            original = source / "_src/sensor.py"
            original.write_text(SENSOR)
            destination = prepare_backend(source, output)
            self.assertEqual(original.read_text(), SENSOR)
            adapted = (destination / "_src/sensor.py").read_text()
            self.assertEqual(adapted.count("site_size[worldid % site_size.shape[0]"), 3)
            self.assertEqual(adapted.count("if m.site_size.ndim == 1"), 3)
            with self.assertRaises(FileExistsError):
                prepare_backend(source, output)
            with self.assertRaisesRegex(ValueError, "already adapted"):
                adapt_sensor_source(adapted)

    def test_every_replacement_is_checked_not_only_declarations(self):
        for token in (
            "site_size: wp.array[wp.vec3]",
            "site_size[objid]",
            "site_size[refid]",
            "m.site_size,",
        ):
            with (
                self.subTest(token=token),
                self.assertRaisesRegex(ValueError, "Unsupported"),
            ):
                adapt_sensor_source(SENSOR.replace(token, "changed", 1))

    def test_copy_failure_does_not_publish_or_leave_partial_package(self):
        with TemporaryDirectory() as temp:
            source, output = Path(temp) / "original", Path(temp) / "adapter"
            (source / "_src").mkdir(parents=True)
            write_physics(source)
            (source / "_src/sensor.py").write_text(SENSOR)
            with patch(
                "myoacross.rl.runners.multi_embodiment.backend.shutil.copytree",
                side_effect=OSError("disk"),
            ):
                with self.assertRaises(OSError):
                    prepare_backend(source, output)
            self.assertEqual(list(output.iterdir()), [])

    def test_unsupported_source_does_not_create_output(self):
        with TemporaryDirectory() as temp:
            source, output = Path(temp) / "original", Path(temp) / "adapter"
            (source / "_src").mkdir(parents=True)
            write_physics(source)
            (source / "_src/sensor.py").write_text("changed")
            with self.assertRaises(ValueError):
                prepare_backend(source, output)
            self.assertFalse(output.exists())

    def test_backend_child_process_uses_adapter_without_changing_parent(self):
        import os
        import sys
        from types import SimpleNamespace
        from myoacross.rl.runners.multi_embodiment import backend

        with TemporaryDirectory() as temp:
            source = Path(temp) / "original"
            (source / "_src").mkdir(parents=True)
            write_physics(source)
            (source / "__init__.py").touch()
            (source / "_src/sensor.py").write_text(SENSOR)
            output = Path(temp) / "cache"
            marker = Path(temp) / "child-path.txt"
            previous = os.environ.get("PYTHONPATH")
            with patch.object(
                backend.importlib.util,
                "find_spec",
                return_value=SimpleNamespace(origin=str(source / "__init__.py")),
            ):
                result = backend.run_with_backend(
                    [
                        sys.executable,
                        "-c",
                        "import os; from pathlib import Path; Path(__import__('sys').argv[1]).write_text(os.environ['PYTHONPATH'])",
                        str(marker),
                    ],
                    output,
                )
            self.assertEqual(result, 0)
            self.assertEqual(os.environ.get("PYTHONPATH"), previous)
            package = Path(marker.read_text().split(os.pathsep)[0]) / "mujoco_warp"
            self.assertIn(backend.MARKER, (package / "_src/sensor.py").read_text())
            self.assertEqual((source / "_src/sensor.py").read_text(), SENSOR)

    def test_concurrent_ranks_share_one_prepared_cache(self):
        import os
        import sys
        import time
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from myoacross.rl.runners.multi_embodiment import backend

        with TemporaryDirectory() as temp:
            source = Path(temp) / "original"
            (source / "_src").mkdir(parents=True)
            write_physics(source)
            (source / "__init__.py").touch()
            (source / "_src/sensor.py").write_text(SENSOR)
            original_prepare = backend.prepare_backend

            def slow_prepare(*args):
                time.sleep(0.05)
                return original_prepare(*args)

            command = [
                sys.executable,
                "-c",
                "import os; assert os.environ['RANK']=='2'; assert os.environ['LOCAL_RANK']=='0'; assert os.environ['WORLD_SIZE']=='4'",
            ]
            with (
                patch.object(
                    backend.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(source / "__init__.py")),
                ),
                patch.object(
                    backend, "prepare_backend", side_effect=slow_prepare
                ) as prepare,
                patch.dict(
                    os.environ, {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "4"}
                ),
            ):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(
                        pool.map(
                            lambda _: backend.run_with_backend(
                                command, Path(temp) / "cache"
                            ),
                            range(4),
                        )
                    )
            self.assertEqual(results, [0] * 4)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual((source / "_src/sensor.py").read_text(), SENSOR)


class PhysicsAdapterTests(TestCase):
    def test_missing_duplicate_or_changed_kernel_inputs_are_rejected(self):
        for module, text in PHYSICS.items():
            with self.subTest(module=module):
                with self.assertRaises(ValueError):
                    adapt_physics_source("", module)
                with self.assertRaises(ValueError):
                    adapt_physics_source(text + text, module)
                with self.assertRaises(ValueError):
                    adapt_physics_source(text.replace("m.tendon_", "m.changed_"), module)
                adapted = adapt_physics_source(text, module)
                with self.assertRaisesRegex(ValueError, "already adapted"):
                    adapt_physics_source(adapted, module)

    def test_bad_physics_source_is_not_published(self):
        with TemporaryDirectory() as temp:
            source, output = Path(temp) / "original", Path(temp) / "adapter"
            (source / "_src").mkdir(parents=True)
            write_physics(source)
            (source / "_src/sensor.py").write_text(SENSOR)
            (source / "_src/derivative.py").write_text("changed")
            with self.assertRaises(ValueError):
                prepare_backend(source, output)
            self.assertFalse(output.exists())

    def test_cache_changes_when_adapter_changes_and_rejects_physics_tampering(self):
        import os
        import sys
        from types import SimpleNamespace
        from myoacross.rl.runners.multi_embodiment import backend

        with TemporaryDirectory() as temp:
            source = Path(temp) / "original"
            (source / "_src").mkdir(parents=True)
            write_physics(source)
            (source / "__init__.py").touch()
            (source / "_src/sensor.py").write_text(SENSOR)
            marker = Path(temp) / "path.txt"
            command = [sys.executable, "-c", "import os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(os.environ['PYTHONPATH'])", str(marker)]
            with patch.object(backend.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(source / "__init__.py"))):
                backend.run_with_backend(command, Path(temp) / "cache")
                original_path = marker.read_text().split(os.pathsep)[0]
                original_adapt = backend.adapt_physics_source
                with patch.object(backend, "adapt_physics_source", side_effect=lambda text, module: original_adapt(text, module) + "\n# adapter change\n"):
                    backend.run_with_backend(command, Path(temp) / "cache")
                self.assertNotEqual(marker.read_text().split(os.pathsep)[0], original_path)
                (Path(original_path) / "mujoco_warp/_src/derivative.py").write_text("tampered")
                with self.assertRaisesRegex(RuntimeError, "Cached derivative"):
                    backend.run_with_backend(command, Path(temp) / "cache")
