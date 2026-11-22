"""Call the unchanged KIT retargeter with experiment-local shape caches."""

from contextlib import contextmanager
from functools import partial
from pathlib import Path
from .motions import MOTION_SET, resolved_motions


@contextmanager
def fitted_shape_cache(cache_root):
    """Route cache writes only; leave IK initialization and iteration limits intact."""
    from myoacross.retarget import retargeter

    original = retargeter.ensure_fitted_shape
    retargeter.ensure_fitted_shape = partial(original, cache_root=Path(cache_root))
    try:
        yield
    finally:
        retargeter.ensure_fitted_shape = original


def retarget_subject_motions(config, records):
    from myoacross.retarget.run import run_motion

    root = Path(config["output_root"])
    sources = resolved_motions(config)
    with fitted_shape_cache(root / "retargets/fitted_shapes"):
        for record in records:
            sid = record["subject_id"]
            for motion in sources:
                run_motion(
                    Path(motion["path"]),
                    root / "embodiments/segment_scale" / sid / "model.xml",
                    config["retarget_config"],
                    root / "retargets/segment_scale" / sid / motion["motion_id"],
                    smplh_model_dir=Path(config["smplh_model_dir"]),
                    identity=dict(
                        target_subject_id=sid,
                        source_subject_id=motion["source_subject_id"],
                        motion_id=motion["motion_id"],
                        category=motion["category"],
                        motion_set=MOTION_SET,
                        embodiment="segment_scale",
                        target_split="test",
                    ),
                )
