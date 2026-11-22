"""Encoder-only policies and supported legacy RL checkpoint loading."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from rsl_rl.storage import RolloutStorage

from myoacross.rl.algorithms.morphology_ppo import MorphologyPPO
from myoacross.rl.networks.actor_critic import build_actor_critic, load_policy
from myoacross.rl.networks.morphology_encoder import MorphologyEncoderConfig
from myoacross.rl.networks.rsl_adapter import pack_policy_observations
from tests.rl.test_morphology_concat import OPTIONS, STATS


class MorphologyEncoderTests(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.obs = {"root": torch.randn(4, 3)}
        self.morph = torch.exp(torch.randn(4, 18) * 0.2)
        self.config = MorphologyEncoderConfig(hidden_dims=(8,), latent_dim=8)
        self.policy = build_actor_critic(
            self.obs,
            2,
            encoder_mode="rl",
            morphology_encoder_config=self.config,
            morphology_statistics=STATS,
            **OPTIONS,
        )

    def legacy_state(self):
        state = {
            name.replace("morphology_encoder.", "autoencoder."): value.clone()
            for name, value in self.policy.state_dict().items()
        }
        state["autoencoder.decoder.0.weight"] = torch.randn(8, 8)
        state["autoencoder.decoder.0.bias"] = torch.randn(8)
        return state

    def algorithm(self, policy):
        observations = {"actor": self.obs, "critic": self.obs, "morphology": self.morph}
        storage = RolloutStorage(
            "rl",
            4,
            2,
            pack_policy_observations(observations, encoder_mode="rl"),
            [2],
            "cpu",
        )
        return MorphologyPPO(policy, storage, num_learning_epochs=1, num_mini_batches=2)

    def test_encoder_output_shape(self):
        self.assertEqual(
            self.policy.morphology_encoder.encode(torch.zeros(4, 18)).shape, (4, 8)
        )

    def test_legacy_policy_restores_exact_outputs(self):
        legacy_config = {
            **asdict(self.config),
            "measurement_weight": 17.0,
            "mass_weight": 1.0,
        }
        checkpoint = {
            "config": {
                "encoder": {"mode": "rl"},
                "autoencoder": legacy_config,
                "network": OPTIONS,
            },
            "algorithm": {
                "encoder_mode": "rl",
                "policy_state_dict": self.legacy_state(),
            },
        }
        with TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.pt"
            torch.save(checkpoint, path)
            restored = load_policy(path, self.obs, 2)
        torch.testing.assert_close(
            restored.act_inference(self.obs, self.morph),
            self.policy.act_inference(self.obs, self.morph),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            restored.value(self.obs, self.morph),
            self.policy.value(self.obs, self.morph),
            rtol=0,
            atol=0,
        )
        self.assertFalse(any("decoder" in name for name in restored.state_dict()))

    def test_legacy_optimizer_state_restores_and_updates(self):
        original = self.algorithm(self.policy)
        sum(p.square().sum() for p in original.parameters).backward()
        original.optimizer.step()
        original.optimizer.zero_grad()
        checkpoint = original.save()
        checkpoint["policy_state_dict"] = self.legacy_state()
        other = build_actor_critic(
            self.obs,
            2,
            encoder_mode="rl",
            morphology_encoder_config=self.config,
            morphology_statistics=STATS,
            **OPTIONS,
        )
        restored = self.algorithm(other)
        restored.load(checkpoint)
        for first, second in zip(original.parameters, restored.parameters):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        for algorithm in (original, restored):
            sum(p.square().sum() for p in algorithm.parameters).backward()
            algorithm.optimizer.step()
        for first, second in zip(original.parameters, restored.parameters):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_legacy_loading_still_rejects_missing_and_unknown_weights(self):
        state = self.legacy_state()
        state.pop(
            next(name for name in state if name.startswith("autoencoder.encoder."))
        )
        with self.assertRaises(RuntimeError):
            self.policy.load_state_dict(state)
        state = self.legacy_state()
        state["unexpected.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            self.policy.load_state_dict(state)
