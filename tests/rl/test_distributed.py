"""Run with torchrun --standalone --nproc-per-node=2 -m tests.rl.test_distributed.

Set TEST_DEVICE=cuda to exercise NCCL. No motion datasets are required.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from rsl_rl.storage import RolloutStorage

from myoacross.rl.runners import distributed as parallel
from myoacross.rl.algorithms.morphology_ppo import MorphologyPPO
from myoacross.rl.networks.morphology_encoder import MorphologyEncoderConfig
from myoacross.rl.embodiments.features import MorphologyFeatureNormalizer
from myoacross.rl.networks.actor_critic import build_actor_critic, load_policy
from myoacross.rl.networks.normalization import ObservationNormalizer
from myoacross.rl.networks.rsl_adapter import pack_policy_observations

DEVICE = "cpu"


def gather(tensor, dim=0):
    buffers = [torch.empty_like(tensor) for _ in range(parallel.world_size())]
    if parallel.world_size() > 1:
        dist.all_gather(buffers, tensor.contiguous())
    else:
        buffers[0].copy_(tensor)
    return torch.cat(buffers, dim=dim)


class DistributedTests(unittest.TestCase):
    def test_global_statistics(self):
        normalizer = ObservationNormalizer(3).to(DEVICE)
        reference = deepcopy(normalizer)
        for step in range(2):
            # Different means and sample counts; one rank has an empty batch.
            count = parallel.rank() + step
            samples = torch.arange(count * 3, device=DEVICE).reshape(count, 3).double()
            samples = samples + 100 * parallel.rank() + step
            normalizer.update(samples)
            pieces = [
                torch.arange((r + step) * 3, device=DEVICE)
                .reshape(r + step, 3)
                .double()
                + 100 * r
                + step
                for r in range(parallel.world_size())
            ]
            with patch.object(parallel, "world_size", return_value=1):
                reference.update(torch.cat(pieces))
            for name in ("mean", "variance", "count"):
                torch.testing.assert_close(
                    getattr(normalizer, name), getattr(reference, name)
                )

    def test_gradient_average_and_unused_parameters(self):
        used = torch.nn.Parameter(torch.tensor([1.0, 2.0], device=DEVICE))
        unused = torch.nn.Parameter(torch.ones(1, device=DEVICE))
        used.grad = torch.full_like(used, parallel.rank() + 1.0)
        parallel.average_gradients([used, unused])
        torch.testing.assert_close(
            used.grad, torch.full_like(used, (parallel.world_size() + 1) / 2)
        )
        self.assertIsNone(unused.grad)
        if parallel.world_size() > 1:
            unused.grad = torch.ones_like(unused) if parallel.rank() == 0 else None
            with self.assertRaisesRegex(RuntimeError, "usage differs"):
                parallel.average_gradients([used, unused])

    def test_primary_result_and_error(self):
        value = parallel.primary_call(lambda: parallel.rank())
        self.assertEqual(value, 0)

        def fail():
            raise ValueError("expected test failure")

        with self.assertRaisesRegex(
            (RuntimeError, ValueError), "expected test failure"
        ):
            parallel.primary_call(fail)

    def test_nonfinite_on_one_rank(self):
        value = torch.tensor(
            float("nan") if parallel.rank() == 0 else 1.0, device=DEVICE
        )
        with self.assertRaises(FloatingPointError):
            parallel.require_finite(value, "test")

    def test_ppo_matches_global_batch_all_encoder_modes(self):
        for mode in ("none", "rl", "concat"):
            with self.subTest(mode=mode):
                self.check_ppo(mode)

    def check_ppo(self, mode):
        torch.manual_seed(42 + parallel.rank())
        count, horizon = 4, 3
        dynamic = {"root": torch.randn(count, 5, device=DEVICE)}
        morphology = torch.rand(count, 18, device=DEVICE) + 0.5
        obs = {"actor": dynamic, "critic": dynamic, "morphology": morphology}
        ae = MorphologyEncoderConfig(hidden_dims=(8, 8), latent_dim=8)
        network = dict(
            group_hidden_dims={"root": (8,)},
            hidden_dim=16,
            num_blocks=1,
            generator_hidden_dim=8,
            head_hidden_dims=(8,),
        )
        if mode == "concat":
            network["group_hidden_dims"]["morphology"] = (8, 8)
        options = (
            {}
            if mode == "none"
            else dict(
                morphology_statistics=MorphologyFeatureNormalizer(
                    log_mean=(0.0,) * 18, log_scale=(1.0,) * 18
                ),
            )
        )
        if mode not in {"none", "concat"}:
            options["morphology_encoder_config"] = ae
        policy = build_actor_critic(
            dynamic, 3, encoder_mode=mode, device=DEVICE, **options, **network
        )
        parallel.broadcast_model(policy)
        reference_policy = deepcopy(policy)
        storage = RolloutStorage(
            "rl",
            count,
            horizon,
            pack_policy_observations(obs, encoder_mode=mode),
            [3],
            DEVICE,
        )
        settings = dict(num_learning_epochs=2, num_mini_batches=1, desired_kl=0.01)
        algorithm = MorphologyPPO(policy, storage, **settings)
        for step in range(horizon):
            algorithm.act(obs)
            rewards = (
                torch.arange(count, device=DEVICE).float() + parallel.rank() + step
            )
            done = torch.zeros(count, device=DEVICE, dtype=torch.bool)
            if step == 1:
                done[0] = True
            algorithm.process_env_step(rewards, done, torch.zeros_like(done))
        algorithm.compute_returns(obs)
        global_adv = gather(storage.advantages, dim=1)
        raw = gather(storage.returns - storage.values, dim=1)
        torch.testing.assert_close(global_adv, (raw - raw.mean()) / (raw.std() + 1e-8))

        global_obs = {
            key: gather(value)
            for key, value in pack_policy_observations(obs, encoder_mode=mode).items()
        }
        from tensordict import TensorDict

        global_storage = RolloutStorage(
            "rl",
            count * parallel.world_size(),
            horizon,
            TensorDict(global_obs, batch_size=[count * parallel.world_size()]),
            [3],
            DEVICE,
        )
        for key in storage.observations.keys():
            global_storage.observations[key].copy_(
                gather(storage.observations[key], dim=1)
            )
        for field in ("actions", "values", "returns", "advantages", "actions_log_prob"):
            getattr(global_storage, field).copy_(gather(getattr(storage, field), dim=1))
        global_storage.distribution_params = tuple(
            gather(p, dim=1) for p in storage.distribution_params
        )
        global_storage.step = horizon
        with patch.object(parallel, "world_size", return_value=1):
            reference = MorphologyPPO(reference_policy, global_storage, **settings)
            reference.returns_ready = True
            reference.update()
        algorithm.update()
        for name, tensor in policy.state_dict().items():
            torch.testing.assert_close(
                tensor,
                reference_policy.state_dict()[name],
                rtol=2e-4,
                atol=2e-5,
                msg=lambda msg: f"{mode}/{name}: {msg}",
            )
            if parallel.world_size() > 1:
                other = tensor.clone()
                dist.broadcast(other, src=0)
                torch.testing.assert_close(tensor, other, rtol=0, atol=0)
        self.assertEqual(algorithm.learning_rate, reference.learning_rate)
        snapshot = algorithm.save()
        algorithm.load(snapshot)
        self.assertFalse(algorithm.normalize_advantage_per_mini_batch)
        # A distributed checkpoint must still load through the ordinary CPU
        # inference loader, without a separate morphology AE checkpoint.
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            torch.save(
                {
                    "algorithm": snapshot,
                    "config": {
                        "encoder": {"mode": mode},
                        "network": network,
                        "morphology_encoder": asdict(ae),
                    },
                },
                checkpoint,
            )
            cpu_dynamic = {key: value.cpu() for key, value in dynamic.items()}
            restored = load_policy(checkpoint, cpu_dynamic, 3, device="cpu")
            cpu_morphology = None if mode == "none" else morphology.cpu()
            expected = policy.act_inference(
                dynamic, None if mode == "none" else morphology
            ).cpu()
            actual = restored.act_inference(cpu_dynamic, cpu_morphology)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    with parallel.training_process(os.environ.get("TEST_DEVICE", "cpu")) as DEVICE:
        unittest.main()
