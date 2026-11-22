"""Compose project and temporary configurations without changing the runtime."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from myoacross.utils.config import load_config


class ConfigurationTests(TestCase):
    def test_temporary_config_inherits_project_defaults_at_root(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "experiment.yaml"
            path.write_text(
                "# @package _global_\n"
                "defaults:\n  - /rl/train@_global_\n  - _self_\n"
                "name: temporary\ntraining:\n  num_iterations: 3\n"
                "encoder:\n  mode: concat\nmorphology_encoder: null\n"
            )
            config = load_config(path)
            self.assertEqual(config["name"], "temporary")
            self.assertEqual(config["training"]["num_iterations"], 3)
            self.assertEqual(config["training"]["embodiments_per_rank"], 8)
            self.assertEqual(config["encoder"]["mode"], "concat")
            self.assertIsNone(config["morphology_encoder"])
            self.assertNotIn("defaults", config)
            self.assertNotIn("hydra", config)
            self.assertNotIn("rl", config)
            self.assertIsInstance(config["network"], dict)
            self.assertIsInstance(config["network"]["group_hidden_dims"]["root"], list)

    def test_local_inheritance_resolves_interpolation_and_replaces_lists(self):
        current = Path.cwd()
        with TemporaryDirectory(prefix="config with spaces ") as temp:
            root = Path(temp)
            (root / "base.yaml").write_text(
                "settings:\n  values: [1, 2, 3]\n  enabled: true\n  optional: null\n"
                "output: ${settings.values}\n"
            )
            path = root / "experiment.yaml"
            path.write_text(
                "defaults: [base, _self_]\nsettings:\n  values: [4, 5]\n"
            )
            config = load_config(path)
            self.assertEqual(config["settings"]["values"], [4, 5])
            self.assertEqual(config["output"], [4, 5])
            self.assertTrue(config["settings"]["enabled"])
            self.assertIsNone(config["settings"]["optional"])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["base.yaml", "experiment.yaml"])
        self.assertEqual(Path.cwd(), current)

    def test_repeated_loading_does_not_keep_previous_config(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("name: first\n")
            self.assertEqual(load_config(path), {"name": "first"})
            path.write_text("name: second\n")
            self.assertEqual(load_config(path), {"name": "second"})

    def test_missing_config_fails_at_input(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                load_config(Path(temp) / "missing.yaml")

    def test_missing_required_value_is_rejected_before_execution(self):
        from omegaconf.errors import MissingMandatoryValue

        with TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("required: ???\n")
            with self.assertRaises(MissingMandatoryValue):
                load_config(path)
