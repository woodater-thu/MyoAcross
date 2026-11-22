"""Elide only constant-zero tendon contributions in the adapted backend."""
from __future__ import annotations

from functools import wraps

MARKER = "MYOACROSS_ZERO_TENDON"
ZERO_FIELDS = {
    "_myoacross_zero_tendon_damping": ("tendon_damping", "tendon_dampingpoly"),
    "_myoacross_zero_tendon_armature": ("tendon_armature",),
    "_myoacross_zero_tendon_passive": (
        "tendon_stiffness", "tendon_stiffnesspoly", "tendon_damping", "tendon_dampingpoly",
    ),
}


def refresh_zero_tendon_flags(model) -> None:
    """Check every world and coefficient before CUDA graph capture."""
    for flag, fields in ZERO_FIELDS.items():
        # Exact zero only: tiny, negative, and nonfinite values use the original path.
        setattr(model, flag, all((getattr(model, field).numpy() == 0).all() for field in fields))


def configure_zero_tendon_elision(sim) -> None:
    """Refresh guards on graph rebuilds; bundle changes must not alter these fields."""
    from mujoco_warp._src import derivative
    from myoacross.rl.embodiments.bundle import WarpBundleSwitcher

    if not getattr(derivative, MARKER, False):
        # Single-body training can use the unadapted backend.
        return
    fields = {field for group in ZERO_FIELDS.values() for field in group}
    if fields.intersection(WarpBundleSwitcher.FIELD_NAMES):
        raise RuntimeError("Zero-tendon guards require coefficients that body bundles do not modify")

    original_create_graph = sim.create_graph

    @wraps(original_create_graph)
    def create_graph():
        refresh_zero_tendon_flags(sim.wp_model)
        original_create_graph()

    sim.create_graph = create_graph
    sim.create_graph()
