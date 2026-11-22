"""Exercise adapted kernels and CUDA graph rebuilding with real tendon dynamics."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
import json
import os
import sys


@skipUnless(os.environ.get("MYOACROSS_TEST_CUDA") == "1", "Set MYOACROSS_TEST_CUDA=1 for CUDA regression")
class ZeroTendonCudaTests(TestCase):
    def test_real_dynamics_and_nonzero_graph_rebuild_match_original(self):
        from myoacross.rl.runners.multi_embodiment.backend import run_with_backend

        with TemporaryDirectory() as temp:
            result = Path(temp) / "result.json"
            code = run_with_backend(
                [sys.executable, str(Path(__file__).resolve()), "--run-gpu", str(result)],
                Path(temp) / "backend",
            )
            self.assertEqual(code, 0)
            checks = json.loads(result.read_text())
            self.assertEqual(len(checks), 7)
            self.assertTrue(all(check["matches_original"] for check in checks))
            self.assertTrue(all(check["changes_dynamics"] for check in checks[1:]))


def run_gpu(result):
    import mujoco
    import torch
    import warp as wp
    from mjlab.sim.sim import Simulation, SimulationCfg, MujocoCfg
    from myoacross.rl.runners.multi_embodiment.zero_tendon import (
        ZERO_FIELDS, configure_zero_tendon_elision,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    xml = '''<mujoco>
      <worldbody>
        <site name="anchor" pos="0 0 0.4"/>
        <body>
          <joint name="joint" type="hinge" axis="0 1 0" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.7 0 0" size="0.03" mass="1"/>
          <site name="end" pos="0.7 0 0"/>
        </body>
      </worldbody>
      <tendon><spatial name="tendon" springlength="0.2">
        <site site="anchor"/><site site="end"/>
      </spatial></tendon>
    </mujoco>'''
    cfg = SimulationCfg(nconmax=16, njmax=32, mujoco=MujocoCfg(gravity=(0, 0, 0)))
    reference = Simulation(num_envs=8, cfg=cfg, model=mujoco.MjModel.from_xml_string(xml), device="cuda:0")
    optimized = Simulation(num_envs=8, cfg=cfg, model=mujoco.MjModel.from_xml_string(xml), device="cuda:0")
    configure_zero_tendon_elision(optimized)
    fields = sorted({f for group in ZERO_FIELDS.values() for f in group})

    def sync():
        wp.synchronize_device(reference.wp_device)
        torch.cuda.synchronize()

    def rollout(sim):
        sim.reset()
        sync()
        wp.to_torch(sim.wp_data.qpos)[:, 0].copy_(torch.linspace(0.2, 0.8, 8, device="cuda:0"))
        wp.to_torch(sim.wp_data.qvel).fill_(0.7)
        sync()
        sim.forward()
        for _ in range(16):
            sim.step()
        sim.forward()
        sync()
        return torch.cat([wp.to_torch(getattr(sim.wp_data, name)).flatten().clone()
                          for name in ("qpos", "qvel", "qacc")])

    checks = []
    zero_result = None
    for field in [None, *fields, "per_world_damping"]:
        for sim in [reference, optimized]:
            for name in fields:
                wp.to_torch(getattr(sim.wp_model, name)).zero_()
            if field == "per_world_damping":
                sim.expand_model_fields(("tendon_damping",))
                wp.to_torch(sim.wp_model.tendon_damping)[3, 0] = 1.0
            elif field is not None:
                wp.to_torch(getattr(sim.wp_model, field)).fill_(1.0)
            sync()
            sim.create_graph()
        expected = rollout(reference)
        actual = rollout(optimized)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert actual.dtype == torch.float32
        if field is None:
            zero_result = expected.clone()
            assert all(getattr(optimized.wp_model, flag) for flag in ZERO_FIELDS)
        changes = field is not None and not torch.allclose(expected, zero_result, rtol=1e-6, atol=1e-6)
        if field is not None:
            assert changes, f"Fixture did not exercise {field} dynamics"
        checks.append({"field": field, "matches_original": True,
                       "max_abs_difference": float((actual - expected).abs().max()),
                       "changes_dynamics": changes})
    Path(result).write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--run-gpu":
        run_gpu(sys.argv[2])
    else:
        import unittest
        unittest.main()
