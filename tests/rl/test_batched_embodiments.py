"""Exercise body-group copies and preflight failures without a CUDA simulator."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch

from myoacross.rl.runners.multi_embodiment.batched import BatchedEmbodiments, copy_body_groups
from tests.rl.test_motion_command import make_command
from tests.rl.test_motion_library import write_motion


class BodyGroupCopyTests(TestCase):
    def test_scalar_vector_matrix_and_empty_fields_match_expanded_copy(self):
        for shape in ((), (7,), (7, 3), (7, 3, 3), (0, 3)):
            for k in (1, 2, 4):
                with self.subTest(shape=shape, count=k):
                    source = torch.arange(
                        int(np.prod((k, *shape))), dtype=torch.float32
                    ).view(k, *shape)
                    target = torch.empty(8, *shape)
                    pointer = target.data_ptr()
                    copy_body_groups(target, source)
                    torch.testing.assert_close(
                        target, source.repeat_interleave(8 // k, dim=0), rtol=0, atol=0
                    )
                    self.assertEqual(target.data_ptr(), pointer)
                    copy_body_groups(target, source + 2)
                    torch.testing.assert_close(
                        target, (source + 2).repeat_interleave(8 // k, dim=0)
                    )

    def test_invalid_shapes_do_not_modify_target(self):
        target = torch.full((8, 3), 42.0)
        for source in (torch.zeros(3, 3), torch.zeros(2, 4), torch.zeros(0, 3)):
            with self.assertRaises(ValueError):
                copy_body_groups(target, source)
            self.assertTrue(torch.all(target == 42))

    def test_insufficient_bank_rejected_before_expanding_simulator(self):
        state = SimpleNamespace(
            env=SimpleNamespace(num_envs=8),
            training_data=SimpleNamespace(
                subjects=[SimpleNamespace(variants=[("v", Path("one"))])]
            ),
        )
        with self.assertRaisesRegex(ValueError, "distinct bodies"):
            BatchedEmbodiments(state, 2)

    def batch(self):
        batch = object.__new__(BatchedEmbodiments)
        batch.count = 2
        batch.signature = "signature"
        batch.layouts = {"body_mass": ((1, 2), np.dtype("float32"))}
        bundle = SimpleNamespace(
            model_signature="signature",
            arrays={"body_mass": np.ones((1, 2), np.float32)},
        )
        batch.state = SimpleNamespace(
            training_data=SimpleNamespace(load_bundle=Mock(return_value=bundle))
        )
        batch.env = SimpleNamespace(sim=Mock())
        selections = [SimpleNamespace(bundle_path=Path(str(i))) for i in range(2)]
        return batch, bundle, selections

    def test_duplicate_bodies_rejected_before_loading(self):
        batch, bundle, selections = self.batch()
        with self.assertRaisesRegex(ValueError, "distinct"):
            batch.install([selections[0]] * 2)
        batch.state.training_data.load_bundle.assert_not_called()

    def test_bad_dtype_or_nonfinite_bundle_rejected_before_device_access(self):
        for values in (
            np.ones((1, 2), np.float64),
            np.array([[1, np.nan]], np.float32),
        ):
            batch, bundle, selections = self.batch()
            bundle.arrays["body_mass"] = values
            with patch(
                "myoacross.rl.runners.multi_embodiment.batched.wp.synchronize_device"
            ) as sync:
                with self.assertRaises(ValueError):
                    batch.install(selections)
                sync.assert_not_called()

    def test_bad_reference_fails_before_changing_physics_or_old_library(self):
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{i}.npz" for i in range(2)]
            write_motion(paths[0])
            write_motion(paths[1], fps=2)
            command, env = make_command(paths)
            library = command.motion_library
            batch, bundle, selections = self.batch()
            batch.env = env
            batch.group_ids = torch.arange(2).repeat_interleave(4)
            for i, selection in enumerate(selections):
                selection.subject = SimpleNamespace(
                    subject_id=str(i), motions=[SimpleNamespace(retarget_path=paths[i])]
                )
            with patch(
                "myoacross.rl.runners.multi_embodiment.batched.wp.synchronize_device"
            ) as sync:
                with self.assertRaises(ValueError):
                    batch.install(selections)
                sync.assert_not_called()
            self.assertIs(command.motion_library, library)
