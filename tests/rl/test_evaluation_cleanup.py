"""Evaluation batches release environment cycles after their frame exits."""
import gc
import weakref
from types import SimpleNamespace
from unittest import TestCase
from myoacross.rl.runners.evaluate import _release_batch_resources

class CleanupTests(TestCase):
    def test_collects_environment_cycle_before_next_batch(self):
        references = []
        class Environment:
            pass
        @_release_batch_resources
        def batch(*, device):
            env = Environment()
            env.manager = SimpleNamespace(env=env)
            references.append(weakref.ref(env))
            return [{"completed": True}]
        enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(12):
                self.assertEqual(batch(device="cpu"), [{"completed": True}])
                self.assertTrue(all(reference() is None for reference in references))
        finally:
            if enabled:
                gc.enable()
