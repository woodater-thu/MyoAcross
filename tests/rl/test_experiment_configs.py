"""Check paper comparisons, shared banks, and fixed output paths."""

from pathlib import Path
from unittest import TestCase
from myoacross.utils.config import load_config
from myoacross.utils.paths import PROJECT_ROOT
from myoacross.utils.datalists import get_motion_specs

ROOT = PROJECT_ROOT / "myoacross/configs"


def read(path):
    return load_config(ROOT / path)


class ExperimentConfigurationTests(TestCase):
    def test_standard_budget_and_shared_comparison_settings(self):
        standard = read("rl/train.yaml")
        self.assertEqual(standard["training"]["num_iterations"], 15000)
        self.assertEqual(standard["training"]["embodiments_per_rank"], 8)
        self.assertEqual(standard["environment"]["num_envs"], 2048)
        expected = [
            m.motion_id
            for m in get_motion_specs(motion_set="kit_flat_locomotion_72")
        ]
        self.assertEqual(len(expected), 64)
        for p in ROOT.rglob("train*.yaml"):
            self.assertNotIn("motion_ids", load_config(p)["data"])
        for p in (ROOT / "experiments/policy_comparison").glob("train_*.yaml"):
            c = load_config(p)
            for key in ("network", "environment", "algorithm", "runtime"):
                expected_settings = standard[key]
                if key == "network" and c["encoder"]["mode"] == "concat":
                    expected_settings = {
                        **expected_settings,
                        "group_hidden_dims": {
                            **expected_settings["group_hidden_dims"],
                            "morphology": [64, 128, 256],
                        },
                    }
                self.assertEqual(c[key], expected_settings, (p.name, key))
            self.assertNotIn("motion_ids", c["data"])
            self.assertEqual(c["data"]["motion_set"], standard["data"]["motion_set"])
            self.assertEqual(c["training"]["num_iterations"], 15000)

    def test_myofullbody_transfer_is_preserved(self):
        c = read("experiments/policy_comparison/train_myofullbody_mlp.yaml")
        self.assertEqual(c["encoder"]["mode"], "none")
        self.assertEqual(c["data"]["embodiment"], "baseline")
        self.assertEqual(c["data"]["morphology_source"], "subject")
        self.assertEqual(c["training"]["embodiments_per_rank"], 1)
        e = read("experiments/policy_comparison/evaluate_myofullbody_mlp_test.yaml")
        self.assertEqual(e["target_embodiment"], "segment_scale")
        self.assertEqual(e["split"], "test")

    def test_generated_banks_match_consumers(self):
        standard = read("rl/generate.yaml")
        for scheme in ("segment_mlp", "segment_concat"):
            c = read(f"experiments/policy_comparison/train_{scheme}.yaml")
            self.assertEqual(c["data"]["bundle_root"], standard["data"]["bundle_root"])
        for n in (5,):
            g = read(f"experiments/virtual_embodiment_count/generate_virtual{n}_train.yaml")
            c = read(f"experiments/virtual_embodiment_count/train_virtual{n}_film.yaml")
            self.assertEqual(g["count_per_subject"], n)
            self.assertEqual(g["sampling"], standard["sampling"])
            self.assertEqual(c["data"]["bundle_root"], g["data"]["bundle_root"])

    def test_asymmetry_comparison_changes_only_sampling_and_paths(self):
        standard = read("rl/generate.yaml")
        g = read("experiments/asymmetry_generalization/generate_asymmetry_enhanced_train.yaml")
        c = read("experiments/asymmetry_generalization/train_asymmetry_enhanced_film.yaml")
        e = read("experiments/asymmetry_generalization/evaluate_asymmetry_enhanced_film_test.yaml")
        expected = dict(standard["sampling"]["covariance"], asymmetry_std=0.025)
        self.assertEqual(g["sampling"]["covariance"], expected)
        self.assertEqual(g["sampling"]["max_relative_change"], [0.12] * 13 + [0.15])
        for key in ("count_per_subject", "seed", "workers", "simulation"):
            self.assertEqual(g[key], standard[key])
        self.assertEqual(c["data"]["bundle_root"], g["data"]["bundle_root"])
        train = read("rl/train.yaml")
        for key in ("training", "environment", "network", "algorithm", "encoder", "runtime"):
            self.assertEqual(c[key], train[key])
        self.assertEqual(e["checkpoint_path"], c["output"]["checkpoint_dir"] + "/final.pt")
        self.assertTrue(e["output_dir"].endswith("test_iter15000"))
        self.assertEqual(e["split"], "test")
        normal = read("experiments/asymmetry_generalization/evaluate_segment_film_test.yaml")
        self.assertEqual(normal["checkpoint_path"], read("rl/evaluate.yaml")["checkpoint_path"])
        test_bank = read("experiments/asymmetry_generalization/generate_asymmetry_test.yaml")
        self.assertEqual(e["bundle_root"], normal["bundle_root"])
        self.assertEqual(e["bundle_root"], test_bank["data"]["bundle_root"])
        self.assertEqual(test_bank["data"]["split"], "test")
        self.assertIsNone(test_bank["sampling"])
        self.assertEqual(test_bank["body_perturbation"]["relative_shortening"], [0, 0.01, 0.025, 0.05, 0.1])
        self.assertEqual(test_bank["body_perturbation"]["sides"], ["left", "right"])
        for key in ("target_embodiment", "perturbation", "seed", "envs_per_subject"):
            self.assertEqual(e[key], read("rl/evaluate.yaml")[key])

    def test_only_supported_modes_and_standard_configs_remain(self):
        self.assertEqual(
            {p.name for p in (ROOT / "rl").iterdir()},
            {"generate.yaml", "train.yaml", "evaluate.yaml"},
        )
        for p in ROOT.rglob("*.yaml"):
            c = load_config(p)
            if "encoder" in c:
                self.assertIn(c["encoder"]["mode"], ("none", "concat", "rl"))
                self.assertNotEqual(
                    c["output"]["checkpoint_dir"], c["output"]["logs_dir"]
                )
