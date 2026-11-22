import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from myoacross.retarget import shape_fitting as fit


class ShapeCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.xml = self.root / "model.xml"
        self.xml.write_text(
            '<mujoco><worldbody><body name="pelvis"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        self.ik = self.root / "ik.json"
        self.ik.write_text("{}")
        self.models = self.root / "smplh"
        (self.models / "neutral").mkdir(parents=True)
        (self.models / "neutral" / "model.npz").write_bytes(b"smpl model fixture")

    def save_fixture(
        self, robot_xml_path, ik_config_path, smplh_model_dir, options, save_path
    ):
        fit.save_fitted_shape(
            shape=torch.zeros((1, 16)),
            scale=torch.ones(1),
            smpl2robot_pos=np.zeros((1, 3)),
            smpl2robot_rot_mat=np.eye(3)[None],
            offset_z=0.0,
            height_scale=1.0,
            metrics={"converged": True, "final_loss": 0.001},
            save_path=str(save_path),
            human_joint_names=["pelvis"],
            local_offsets={
                "pos_offsets": {"pelvis": [0, 0, 0]},
                "rot_offsets": {"pelvis": [1, 0, 0, 0]},
            },
        )

    def ensure(self):
        return fit.ensure_fitted_shape(
            self.xml,
            self.ik,
            self.models,
            {"iterations": 500, "lr": 0.001},
            cache_root=self.root / "cache",
        )

    def test_reuse_and_invalidate_on_model_and_smpl_change(self):
        with patch.object(
            fit, "_fit_and_save", side_effect=self.save_fixture
        ) as fitting:
            first = self.ensure()
            self.assertEqual(first, self.ensure())
            self.assertEqual(fitting.call_count, 1)
            self.xml.write_text(self.xml.read_text().replace("0.1", "0.2"))
            second = self.ensure()
            self.assertNotEqual(first, second)
            (self.models / "neutral" / "model.npz").write_bytes(b"changed model")
            self.assertNotEqual(second, self.ensure())
            self.assertEqual(fitting.call_count, 3)

    def test_missing_local_offsets_rejected(self):
        with patch.object(fit, "_fit_and_save", side_effect=self.save_fixture):
            path = self.ensure()
        meta = path.with_name("shape_metadata.json")
        data = json.loads(meta.read_text())
        data["local_offsets"] = None
        meta.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            fit.validate_fitted_shape(path)


if __name__ == "__main__":
    unittest.main()
