"""Resume must preserve optimizer state, absolute counters and save cadence."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import json
import tempfile

import torch

from myoacross.rl.runners.resume import validate_resume_checkpoint
from myoacross.rl.runners.train_rl import train_policy
from myoacross.utils.config import load_config


def example_checkpoint():
    config = {
        "network": {},
        "environment": {"num_envs": 2},
        "algorithm": {},
        "training": {
            "num_iterations": 1000,
            "num_steps_per_env": 2,
            "updates_per_embodiment": 10,
            "save_interval": 100,
        },
        "data": {"embodiment": "segment_scale", "morphology_source": "virtual"},
        "runtime": {"seed": 42},
        "encoder": {"mode": "none"},
        "simulation": {},
        "distributed": {"world_size": 1},
    }
    return {
        "iteration": 499,
        "config": config,
        "algorithm": {
            "policy_state_dict": {},
            "optimizer_state_dict": {"sentinel": 123},
            "ppo_config": {"learning_rate": 1e-5},
            "encoder_mode": "none",
        },
    }


@dataclass
class FakeSimConfig:
    timestep: float = 0.002


class ResumeTests(TestCase):
    def test_allow_new_output_and_cadence_but_reject_batch_changes(self):
        checkpoint = example_checkpoint()
        config = deepcopy(checkpoint["config"])
        config["name"] = "continued"
        config["training"]["save_interval"] = 500
        self.assertEqual(validate_resume_checkpoint(checkpoint, config, 1), 499)
        with self.assertRaisesRegex(ValueError, "world_size"):
            validate_resume_checkpoint(checkpoint, config, 4)
        config["environment"]["num_envs"] = 4
        with self.assertRaisesRegex(ValueError, "environment"):
            validate_resume_checkpoint(checkpoint, config, 1)

    def test_resume_accepts_removed_zero_loss_setting(self):
        checkpoint = example_checkpoint()
        config = deepcopy(checkpoint["config"])
        checkpoint["config"]["algorithm"]["reconstruction_coef"] = 0.0
        self.assertEqual(validate_resume_checkpoint(checkpoint, config, 1), 499)
        checkpoint["config"]["algorithm"]["reconstruction_coef"] = 0.1
        with self.assertRaisesRegex(ValueError, "algorithm"):
            validate_resume_checkpoint(checkpoint, config, 1)

    def test_resume_accepts_equivalent_relative_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = example_checkpoint()
            checkpoint["config"]["data"]["bundle_root"] = "data/bank"
            checkpoint["config"]["data"]["retargets_root"] = "data/retargets"
            config = deepcopy(checkpoint["config"])
            config["data"]["bundle_root"] = str(root / "data/bank")
            config["data"]["retargets_root"] = str(root / "data/retargets")
            with patch("myoacross.rl.runners.resume.PROJECT_ROOT", root):
                self.assertEqual(validate_resume_checkpoint(checkpoint, config, 1), 499)
                config["data"]["bundle_root"] = str(root / "data/unrelated_bank")
                with self.assertRaisesRegex(ValueError, "bundle_root"):
                    validate_resume_checkpoint(checkpoint, config, 1)

    def test_resume_rejects_changed_data_paths(self):
        for field in ("bundle_root", "retargets_root"):
            with self.subTest(field=field):
                checkpoint = example_checkpoint()
                checkpoint["config"]["data"][field] = "data/old_location"
                config = deepcopy(checkpoint["config"])
                config["data"][field] = "data/new_location"
                with self.assertRaisesRegex(ValueError, field):
                    validate_resume_checkpoint(checkpoint, config, 1)

    def test_resume_preserves_simultaneous_body_count(self):
        checkpoint = example_checkpoint()
        config = deepcopy(checkpoint["config"])
        config["training"]["embodiments_per_rank"] = 1
        self.assertEqual(validate_resume_checkpoint(checkpoint, config, 1), 499)
        config["training"]["embodiments_per_rank"] = 2
        with self.assertRaisesRegex(ValueError, "embodiments_per_rank"):
            validate_resume_checkpoint(checkpoint, config, 1)

    def test_total_target_and_optimizer_are_required(self):
        checkpoint = example_checkpoint()
        config = deepcopy(checkpoint["config"])
        config["training"]["num_iterations"] = 499
        with self.assertRaisesRegex(ValueError, "total target"):
            validate_resume_checkpoint(checkpoint, config, 1)
        config["training"]["num_iterations"] = 1000
        del checkpoint["algorithm"]["optimizer_state_dict"]
        with self.assertRaisesRegex(ValueError, "optimizer_state_dict"):
            validate_resume_checkpoint(checkpoint, config, 1)

    def test_existing_output_is_rejected_without_changing_saved_config(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "final.pt").write_bytes(b"existing checkpoint")
            saved = output / "config.yaml"
            saved.write_text("original: configuration\n")
            with self.assertRaises(FileExistsError):
                train_policy(
                    SimpleNamespace(),
                    num_iterations=1,
                    num_steps_per_env=1,
                    updates_per_embodiment=1,
                    algorithm_options={},
                    output_dir=output,
                    resolved_config={"new": "configuration"},
                )
            self.assertEqual(saved.read_text(), "original: configuration\n")

    def test_runner_continues_absolute_iterations_and_saves_at_500(self):
        checkpoint = example_checkpoint()
        obs = {
            "actor": {"root": torch.zeros(2, 3)},
            "critic": {"root": torch.zeros(2, 3)},
        }
        selection = SimpleNamespace(
            bundle_path="bundle",
            variant_id="variant",
            subject=SimpleNamespace(subject_id="subject"),
        )
        next_selection = SimpleNamespace(
            bundle_path="bundle",
            variant_id="variant",
            subject=SimpleNamespace(subject_id="other-subject"),
        )
        command = SimpleNamespace(
            motion_paths=(Path("motion-a"), Path("motion-b")),
            motion_ids=torch.tensor([0, 1]),
            motion_library=SimpleNamespace(num_motions=2),
            consume_motion_sample_counts=lambda: torch.tensor([1, 1]),
        )
        policy = SimpleNamespace(
            encoder_mode="none",
            morphology_encoder=None,
            update_observation_statistics=lambda *args: None,
        )
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            cfg=SimpleNamespace(sim=FakeSimConfig()),
            command_manager=SimpleNamespace(get_term=lambda name: command),
            action_manager=SimpleNamespace(total_action_dim=1),
            step=lambda actions: (
                obs,
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            ),
        )
        state = SimpleNamespace(
            env=env,
            policy=policy,
            observations=obs,
            selection=selection,
            training_data=SimpleNamespace(sample=lambda: next_selection),
            switcher=object(),
            prepared_bundle=object(),
        )
        loaded = []
        events = []

        def switch(env, switcher, prepared, selected, *, inject_model):
            self.assertFalse(inject_model)
            self.assertIs(selected, next_selection)
            events.append("switch")
            return obs, {}

        class FakeAlgorithm:
            def __init__(self, **kwargs):
                self.learning_rate = 0.1

            def load(self, saved, *, load_optimizer):
                loaded.append((saved, load_optimizer))
                self.learning_rate = saved["ppo_config"]["learning_rate"]

            def train_mode(self):
                pass

            def act(self, observations):
                return torch.zeros(2, 1)

            def process_env_step(self, **kwargs):
                pass

            def compute_returns(self, observations):
                events.append("bootstrap")

            def update(self):
                events.append("update")
                return {"surrogate": 0.0, "value": 0.0, "kl": 0.01}

            def save(self):
                return checkpoint["algorithm"]

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("myoacross.rl.runners.train_rl.MorphologyPPO", FakeAlgorithm),
                patch(
                    "myoacross.rl.runners.train_rl.switch_training_embodiment", switch
                ),
            ):
                train_policy(
                    state,
                    num_iterations=501,
                    num_steps_per_env=2,
                    updates_per_embodiment=10,
                    algorithm_options={},
                    output_dir=directory,
                    resolved_config=checkpoint["config"],
                    save_interval=500,
                    resume_checkpoint=checkpoint,
                )
            path = Path(directory)
            self.assertEqual(load_config(path / "config.yaml"), checkpoint["config"])
            rows = [
                json.loads(line)
                for line in (path / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["iteration"] for r in rows], [500, 501])
            self.assertEqual([r["environment_steps"] for r in rows], [2000, 2004])
            self.assertEqual(rows[0]["active_motion_env_counts"], [1, 1])
            self.assertEqual(
                events, ["bootstrap", "update", "switch", "bootstrap", "update"]
            )
            self.assertEqual(
                sorted(p.name for p in path.glob("*.pt")),
                ["checkpoint_000500.pt", "final.pt"],
            )
            self.assertEqual(
                torch.load(path / "final.pt", weights_only=True)["iteration"], 501
            )
            for name in ("checkpoint_000500.pt", "final.pt"):
                self.assertNotIn(
                    "motion_sampling_version",
                    torch.load(path / name, weights_only=True),
                )
        self.assertEqual(loaded, [(checkpoint["algorithm"], True)])


if __name__ == "__main__":
    main()
