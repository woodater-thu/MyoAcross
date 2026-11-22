"""Zero guards must reject nonzero values and refresh after parameter changes."""
from types import SimpleNamespace
from unittest import TestCase
import numpy as np
from myoacross.rl.runners.multi_embodiment.zero_tendon import ZERO_FIELDS, refresh_zero_tendon_flags


class Array:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float32)

    def numpy(self):
        return self.values


def model():
    return SimpleNamespace(**{
        name: Array(np.zeros((2, 3, 2) if name.endswith('poly') else (2, 3)))
        for names in ZERO_FIELDS.values() for name in names
    })


class ZeroTendonGuardTests(TestCase):
    def test_zero_coefficients_enable_all_three_guards(self):
        value = model()
        refresh_zero_tendon_flags(value)
        for flag in ZERO_FIELDS:
            self.assertTrue(getattr(value, flag, False))

    def test_any_nonzero_world_or_polynomial_coefficient_keeps_original_computation(self):
        for field in {f for names in ZERO_FIELDS.values() for f in names}:
            for coefficient in [0.1, -0.1, 1e-30, float('nan')]:
                with self.subTest(field=field, coefficient=coefficient):
                    value = model()
                    refresh_zero_tendon_flags(value)
                    getattr(value, field).values.flat[-1] = coefficient
                    refresh_zero_tendon_flags(value)
                    for flag, fields in ZERO_FIELDS.items():
                        self.assertEqual(getattr(value, flag, False), field not in fields)

    def test_graph_refresh_can_reenable_elision_after_nonzero_parameters_are_removed(self):
        value = model()
        value.tendon_damping.values[:] = 0.2
        refresh_zero_tendon_flags(value)
        self.assertFalse(getattr(value, '_myoacross_zero_tendon_damping', False))
        value.tendon_damping.values[:] = 0.0
        refresh_zero_tendon_flags(value)
        self.assertTrue(getattr(value, '_myoacross_zero_tendon_damping', False))
