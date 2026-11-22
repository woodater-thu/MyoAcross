"""Import plotting tools without command parsing or artifact writes."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase

from myoacross.utils.paths import PROJECT_ROOT


class PlottingEntrypointTests(TestCase):
    def test_import_has_no_output_side_effects(self):
        for module in (
            "myoacross.utils.plotting.plot_morphology_conditioning",
        ):
            with self.subTest(module=module), TemporaryDirectory() as temp:
                code = """
import importlib
import sys
from unittest.mock import patch
import matplotlib.pyplot
sys.argv = ['import-check']
with patch('pathlib.Path.mkdir', side_effect=AssertionError('import creates directory')), \
     patch('matplotlib.figure.Figure.savefig', side_effect=AssertionError('import saves figure')):
    importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else MODULE)
""".replace("MODULE", repr(module))
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=temp,
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(PROJECT_ROOT),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLCONFIGDIR": str(Path(temp) / "mpl"),
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_does_not_create_artifacts(self):
        for module in (
            "myoacross.utils.plotting.plot_morphology_conditioning",
        ):
            with self.subTest(module=module), TemporaryDirectory() as temp:
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=temp,
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(PROJECT_ROOT),
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLCONFIGDIR": str(Path(temp) / "mpl"),
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertEqual(sorted(p.name for p in Path(temp).iterdir()), ["mpl"])
