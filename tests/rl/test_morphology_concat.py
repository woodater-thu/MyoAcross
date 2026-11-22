"""Morphology group encoding, PPO gradient flow and checkpoint restoration."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import torch

from myoacross.rl.networks.actor_critic import build_actor_critic, load_policy
from myoacross.rl.networks.commons import FiLMLayer, ObservationEncoder
from myoacross.rl.embodiments.features import (
    MorphologyFeatureNormalizer,
    MorphologyFeatures,
)
from myoacross.rl.networks.morphology_encoder import MorphologyEncoderConfig
from myoacross.rl.networks.rsl_adapter import pack_policy_observations
from myoacross.rl.algorithms.morphology_ppo import MorphologyPPO
from rsl_rl.storage import RolloutStorage


OPTIONS = dict(
    group_hidden_dims={"root": (8,)},
    hidden_dim=16,
    num_blocks=2,
    head_hidden_dims=(8,),
    generator_hidden_dim=8,
)
CONCAT_OPTIONS = {
    **OPTIONS,
    "group_hidden_dims": {"root": (8,), "morphology": (8, 8)},
}
STATS = MorphologyFeatureNormalizer(log_mean=(0.2,) * 18, log_scale=(0.5,) * 18)


class MorphologyConcatTests(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.obs = {"root": torch.randn(4, 3)}
        self.morph = torch.exp(torch.randn(4, 18) * 0.2)

    def build(self, **kwargs):
        return build_actor_critic(
            self.obs, 2, encoder_mode="concat", **CONCAT_OPTIONS, **kwargs
        )

    def test_morphology_group_encoding_and_no_film(self):
        policy = self.build(morphology_statistics=STATS)
        self.assertIsNone(policy.morphology_encoder)
        self.assertFalse(any(isinstance(m, FiLMLayer) for m in policy.modules()))
        actor_encoder = policy.actor.backbone.observation_encoder.encoders["morphology"]
        critic_encoder = policy.critic.backbone.observation_encoder.encoders["morphology"]
        self.assertIsInstance(actor_encoder, ObservationEncoder)
        self.assertIsNot(actor_encoder, critic_encoder)
        self.assertTrue(
            set(map(id, actor_encoder.parameters())).isdisjoint(
                map(id, critic_encoder.parameters())
            )
        )
        self.assertNotIn("morphology", policy.actor_observation_normalizer.normalizers)
        captured = []
        handle = policy.actor.backbone.input_projection[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        policy.act_inference(self.obs, self.morph)
        handle.remove()
        torch.testing.assert_close(
            captured[0][:, -8:], actor_encoder((self.morph.log() - 0.2) / 0.5)
        )
        self.assertEqual(captured[0].shape[-1], 16)
        before = policy.act_inference(self.obs, self.morph)
        after = policy.act_inference(self.obs, self.morph * 1.5)
        self.assertFalse(torch.equal(before, after))
        self.assertFalse(
            torch.equal(
                policy.value(self.obs, self.morph),
                policy.value(self.obs, self.morph * 1.5),
            )
        )
        with self.assertRaisesRegex(ValueError, "requires morphology"):
            policy.act_inference(self.obs)

    def test_actor_and_critic_morphology_gradients_are_independent(self):
        policy = self.build(morphology_statistics=STATS)
        actor_encoder = policy.actor.backbone.observation_encoder.encoders["morphology"]
        critic_encoder = policy.critic.backbone.observation_encoder.encoders["morphology"]
        features = policy.morphology_normalizer(self.morph)
        actor_inputs = policy.actor_observation_normalizer(self.obs)
        critic_inputs = policy.critic_observation_normalizer(self.obs)

        policy.actor(actor_inputs, features).mean.sum().backward()
        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in actor_encoder.parameters()
        ))
        self.assertTrue(all(p.grad is None for p in critic_encoder.parameters()))
        policy.zero_grad(set_to_none=True)
        policy.critic(critic_inputs, features).sum().backward()
        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in critic_encoder.parameters()
        ))
        self.assertTrue(all(p.grad is None for p in actor_encoder.parameters()))

    def test_morphology_group_with_lookahead(self):
        observations = {**self.obs, "lookahead": torch.randn(4, 2, 3)}
        policy = build_actor_critic(
            observations,
            2,
            encoder_mode="concat",
            morphology_statistics=STATS,
            lookahead_frame_hidden_dims=(4,),
            lookahead_fusion_hidden_dims=(4,),
            **CONCAT_OPTIONS,
        )
        self.assertEqual(policy.actor.backbone.input_projection[0].in_features, 20)
        self.assertEqual(policy.act_inference(observations, self.morph).shape, (4, 2))
        self.assertEqual(policy.value(observations, self.morph).shape, (4,))

    def test_ppo_updates_morphology_groups_and_restores_optimizer(self):
        policy = self.build(morphology_statistics=STATS)
        observations = {"actor": self.obs, "critic": self.obs, "morphology": self.morph}

        def algorithm(p):
            storage = RolloutStorage(
                training_type="rl",
                num_envs=4,
                num_transitions_per_env=2,
                obs=pack_policy_observations(observations, encoder_mode="concat"),
                actions_shape=(2,),
                device="cpu",
            )
            return MorphologyPPO(p, storage, num_learning_epochs=1, num_mini_batches=1)

        algo = algorithm(policy)
        weights = [
            p.backbone.observation_encoder.encoders["morphology"].network[0].weight
            for p in (policy.actor, policy.critic)
        ]
        before = [w.detach().clone() for w in weights]
        for _ in range(2):
            algo.act(observations)
            algo.process_env_step(
                torch.randn(4),
                torch.zeros(4, dtype=torch.bool),
                torch.zeros(4, dtype=torch.bool),
            )
        algo.compute_returns(observations)
        metrics = algo.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        for old, w in zip(before, weights):
            self.assertFalse(torch.equal(old, w))
        restored = self.build(policy_state_dict=policy.state_dict())
        other = algorithm(restored)
        other.load(algo.save(), load_optimizer=True)
        self.assertTrue(other.optimizer.state)
        torch.testing.assert_close(
            restored.act_inference(self.obs, self.morph),
            policy.act_inference(self.obs, self.morph),
        )

    def test_full_policy_load_without_ae_config(self):
        policy = self.build(morphology_statistics=STATS)
        with TemporaryDirectory() as d:
            path = Path(d) / "final.pt"
            torch.save(
                {
                    "config": {"encoder": {"mode": "concat"}, "network": CONCAT_OPTIONS},
                    "algorithm": {
                        "encoder_mode": "concat",
                        "policy_state_dict": policy.state_dict(),
                    },
                },
                path,
            )
            restored = load_policy(path, self.obs, 2)
        torch.testing.assert_close(
            restored.act_inference(self.obs, self.morph),
            policy.act_inference(self.obs, self.morph),
        )

    def test_training_initialization_and_resume(self):
        from myoacross.rl.runners.train_rl import initialize_actor_critic

        with (
            TemporaryDirectory() as d,
            patch(
                "myoacross.rl.runners.train_rl.load_morphology_features",
                return_value=(
                    [MorphologyFeatures((1.0,) * 18), MorphologyFeatures((2.0,) * 18)],
                    ("KIT_3",),
                ),
            ),
        ):
            args = dict(
                observations=self.obs,
                action_dim=2,
                encoder_mode="concat",
                train_index_paths=[],
                network_options=CONCAT_OPTIONS,
                device="cpu",
            )
            policy = initialize_actor_critic(output_dir=Path(d) / "first", **args)
            restored = initialize_actor_critic(
                output_dir=Path(d) / "resume",
                resume_checkpoint={
                    "config": {},
                    "algorithm": {"policy_state_dict": policy.state_dict()},
                },
                **args,
            )
            torch.testing.assert_close(
                policy.act_inference(self.obs, self.morph),
                restored.act_inference(self.obs, self.morph),
            )

    def test_reject_ae_options_and_missing_statistics(self):
        for kwargs in (
            {},
            {"morphology_checkpoint": "unused.pt"},
            {"morphology_encoder_config": MorphologyEncoderConfig()},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_supported_modes_state_dict_roundtrip(self):
        ae = MorphologyEncoderConfig(hidden_dims=(8, 8), latent_dim=8)
        for mode in ("none", "rl"):
            with self.subTest(mode=mode):
                extra = (
                    {}
                    if mode == "none"
                    else dict(morphology_encoder_config=ae, morphology_statistics=STATS)
                )
                p = build_actor_critic(
                    self.obs, 2, encoder_mode=mode, **extra, **OPTIONS
                )
                extra = {} if mode == "none" else dict(morphology_encoder_config=ae)
                q = build_actor_critic(
                    self.obs,
                    2,
                    encoder_mode=mode,
                    policy_state_dict=p.state_dict(),
                    **extra,
                    **OPTIONS,
                )
                torch.testing.assert_close(
                    p.act_inference(self.obs, self.morph),
                    q.act_inference(self.obs, self.morph),
                )

    def test_removed_modes_are_rejected_before_model_construction(self):
        for mode in ("frozen", "rl_reconstruction"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "Unknown encoder mode"),
            ):
                build_actor_critic(
                    self.obs,
                    2,
                    encoder_mode=mode,
                    morphology_encoder_config=MorphologyEncoderConfig(),
                    morphology_statistics=STATS,
                    **OPTIONS,
                )
