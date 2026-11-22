"""Define fixed target-subject splits and shared source-motion sets.

Target subjects identify body models. Source motions identify reference clips.
Keep subject splits independent of motion sets.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import re

from myoacross.utils.paths import DATASET_ROOT


# Target subjects
# The fixed split contains 55 KIT and 86 BMLmovi subjects from 2026-09-28.
# Sort the 141 IDs, shuffle with seed 42, and assign 110 to train and 31 to test.
# Store each split in sorted order. Do not reshuffle when adding or excluding subjects.
TARGET_SUBJECT_SPLIT_SEED = 42

TRAIN_TARGET_SUBJECT_IDS: tuple[str, ...] = (
    "BMLmovi_Subject_12_F_MoSh",
    "BMLmovi_Subject_13_F_MoSh",
    "BMLmovi_Subject_15_F_MoSh",
    "BMLmovi_Subject_16_F_MoSh",
    "BMLmovi_Subject_1_F_MoSh",
    "BMLmovi_Subject_20_F_MoSh",
    "BMLmovi_Subject_21_F_MoSh",
    "BMLmovi_Subject_22_F_MoSh",
    "BMLmovi_Subject_23_F_MoSh",
    "BMLmovi_Subject_24_F_MoSh",
    "BMLmovi_Subject_25_F_MoSh",
    "BMLmovi_Subject_27_F_MoSh",
    "BMLmovi_Subject_28_F_MoSh",
    "BMLmovi_Subject_29_F_MoSh",
    "BMLmovi_Subject_2_F_MoSh",
    "BMLmovi_Subject_30_F_MoSh",
    "BMLmovi_Subject_31_F_MoSh",
    "BMLmovi_Subject_34_F_MoSh",
    "BMLmovi_Subject_37_F_MoSh",
    "BMLmovi_Subject_39_F_MoSh",
    "BMLmovi_Subject_3_F_MoSh",
    "BMLmovi_Subject_40_F_MoSh",
    "BMLmovi_Subject_41_F_MoSh",
    "BMLmovi_Subject_42_F_MoSh",
    "BMLmovi_Subject_43_F_MoSh",
    "BMLmovi_Subject_45_F_MoSh",
    "BMLmovi_Subject_46_F_MoSh",
    "BMLmovi_Subject_47_F_MoSh",
    "BMLmovi_Subject_48_F_MoSh",
    "BMLmovi_Subject_4_F_MoSh",
    "BMLmovi_Subject_50_F_MoSh",
    "BMLmovi_Subject_51_F_MoSh",
    "BMLmovi_Subject_52_F_MoSh",
    "BMLmovi_Subject_53_F_MoSh",
    "BMLmovi_Subject_54_F_MoSh",
    "BMLmovi_Subject_55_F_MoSh",
    "BMLmovi_Subject_56_F_MoSh",
    "BMLmovi_Subject_57_F_MoSh",
    "BMLmovi_Subject_58_F_MoSh",
    "BMLmovi_Subject_59_F_MoSh",
    "BMLmovi_Subject_5_F_MoSh",
    "BMLmovi_Subject_60_F_MoSh",
    "BMLmovi_Subject_62_F_MoSh",
    "BMLmovi_Subject_64_F_MoSh",
    "BMLmovi_Subject_66_F_MoSh",
    "BMLmovi_Subject_68_F_MoSh",
    "BMLmovi_Subject_69_F_MoSh",
    "BMLmovi_Subject_70_F_MoSh",
    "BMLmovi_Subject_72_F_MoSh",
    "BMLmovi_Subject_73_F_MoSh",
    "BMLmovi_Subject_74_F_MoSh",
    "BMLmovi_Subject_75_F_MoSh",
    "BMLmovi_Subject_79_F_MoSh",
    "BMLmovi_Subject_80_F_MoSh",
    "BMLmovi_Subject_81_F_MoSh",
    "BMLmovi_Subject_83_F_MoSh",
    "BMLmovi_Subject_85_F_MoSh",
    "BMLmovi_Subject_86_F_MoSh",
    "BMLmovi_Subject_87_F_MoSh",
    "BMLmovi_Subject_88_F_MoSh",
    "BMLmovi_Subject_89_F_MoSh",
    "BMLmovi_Subject_90_F_MoSh",
    "BMLmovi_Subject_9_F_MoSh",
    "KIT_10",
    "KIT_11",
    "KIT_12",
    "KIT_1229",
    "KIT_1346",
    "KIT_1347",
    "KIT_1487",
    "KIT_167",
    "KIT_1717",
    "KIT_1721",
    "KIT_1747",
    "KIT_183",
    "KIT_200",
    "KIT_205",
    "KIT_291",
    "KIT_314",
    "KIT_317",
    "KIT_348",
    "KIT_359",
    "KIT_379",
    "KIT_4",
    "KIT_423",
    "KIT_424",
    "KIT_425",
    "KIT_441",
    "KIT_442",
    "KIT_471",
    "KIT_513",
    "KIT_551",
    "KIT_572",
    "KIT_575",
    "KIT_576",
    "KIT_6",
    "KIT_63",
    "KIT_674",
    "KIT_675",
    "KIT_7",
    "KIT_725",
    "KIT_8",
    "KIT_883",
    "KIT_9",
    "KIT_912",
    "KIT_917",
    "KIT_948",
    "KIT_955",
    "KIT_965",
    "KIT_969",
)

TEST_TARGET_SUBJECT_IDS: tuple[str, ...] = (
    "BMLmovi_Subject_11_F_MoSh",
    "BMLmovi_Subject_14_F_MoSh",
    "BMLmovi_Subject_17_F_MoSh",
    "BMLmovi_Subject_18_F_MoSh",
    "BMLmovi_Subject_19_F_MoSh",
    "BMLmovi_Subject_32_F_MoSh",
    "BMLmovi_Subject_33_F_MoSh",
    "BMLmovi_Subject_35_F_MoSh",
    "BMLmovi_Subject_36_F_MoSh",
    "BMLmovi_Subject_38_F_MoSh",
    "BMLmovi_Subject_44_F_MoSh",
    "BMLmovi_Subject_61_F_MoSh",
    "BMLmovi_Subject_63_F_MoSh",
    "BMLmovi_Subject_65_F_MoSh",
    "BMLmovi_Subject_67_F_MoSh",
    "BMLmovi_Subject_6_F_MoSh",
    "BMLmovi_Subject_71_F_MoSh",
    "BMLmovi_Subject_76_F_MoSh",
    "BMLmovi_Subject_77_F_MoSh",
    "BMLmovi_Subject_78_F_MoSh",
    "BMLmovi_Subject_82_F_MoSh",
    "BMLmovi_Subject_84_F_MoSh",
    "BMLmovi_Subject_8_F_MoSh",
    "KIT_1226",
    "KIT_1297",
    "KIT_3",
    "KIT_378",
    "KIT_421",
    "KIT_950",
    "KIT_952",
    "KIT_987",
)

TARGET_SUBJECT_SPLITS: dict[str, tuple[str, ...]] = {
    "train": TRAIN_TARGET_SUBJECT_IDS,
    "test": TEST_TARGET_SUBJECT_IDS,
}

# Map excluded subject IDs to reasons. Apply exclusions to all embodiment strategies.
# Keep the original split membership.
TARGET_SUBJECT_EXCLUSIONS: dict[str, str] = {}


def get_target_subject_ids(
    *,
    split: str,
    embodiment: str | None = None,
) -> tuple[str, ...]:
    """Return target IDs for a split after explicit exclusions.

    Validate the embodiment name. Use the same split for all embodiments.
    Do not check model files or retarget quality here.
    """
    if split not in TARGET_SUBJECT_SPLITS:
        raise ValueError(f"Unknown target split: {split!r}")
    if embodiment not in (None, "baseline", "height_scale", "segment_scale"):
        raise ValueError(f"Unknown embodiment: {embodiment!r}")

    registered = set(TRAIN_TARGET_SUBJECT_IDS + TEST_TARGET_SUBJECT_IDS)
    for subject_id, reason in TARGET_SUBJECT_EXCLUSIONS.items():
        if subject_id not in registered:
            raise ValueError(f"Excluded target subject is not registered: {subject_id!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Missing exclusion reason for {subject_id!r}")

    return tuple(
        subject_id for subject_id in TARGET_SUBJECT_SPLITS[split]
        if subject_id not in TARGET_SUBJECT_EXCLUSIONS
    )


def get_subject_reference_motion(
    subject_id: str,
    *,
    dataset_root: str | Path = DATASET_ROOT,
) -> Path:
    """Find a registered subject's shape reference under the dataset root.

    Prefer shape.npz. Otherwise, select the first *_poses.npz by name.
    Measurement generation reads gender and betas from the selected file.
    """
    if subject_id not in TRAIN_TARGET_SUBJECT_IDS + TEST_TARGET_SUBJECT_IDS:
        raise ValueError(f"Unknown target subject: {subject_id!r}")
    dataset, _, subject = subject_id.partition("_")
    root = Path(dataset_root).expanduser().resolve()
    subject_dir = root / dataset / subject
    shape_path = subject_dir / "shape.npz"
    if shape_path.is_file():
        return shape_path.resolve()
    for path in sorted(subject_dir.glob("*_poses.npz")):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No shape.npz or *_poses.npz for {subject_id}: {subject_dir}")


@dataclass(frozen=True, slots=True)
class MotionSpec:
    """Describe a source clip by ID, dataset-relative path, and category.

    Keep IDs and source paths unique within each motion set.
    Check file existence when loading the clip.
    """

    motion_id: str
    source_path: Path | str
    category: str

    def __post_init__(self) -> None:
        """Validate the clip identity and convert its relative path to Path."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.motion_id):
            raise ValueError(f"Invalid motion ID: {self.motion_id!r}")
        path = Path(self.source_path)
        if (
            path.is_absolute()
            or len(path.parts) != 3
            or path.parts[0] != "KIT"
            or ".." in path.parts
            or path.suffix != ".npz"
        ):
            raise ValueError(f"Expected relative KIT/<subject>/<motion>.npz: {path}")
        if not self.category.strip():
            raise ValueError("Motion category must not be empty")
        object.__setattr__(self, "source_path", path)

    @property
    def source_subject_id(self) -> str:
        """Return the source performer ID from the clip path."""
        return f"KIT_{self.source_path.parts[1]}"


# Source motions
# Register 72 KIT clips in nine categories. Do not split motions into train and test.
# Training configurations select clips from this set. Evaluation can reuse them
# with held-out target subjects.
KIT_FLAT_LOCOMOTION_72: tuple[MotionSpec, ...] = (
    # circle_ccw (8 motions)
    MotionSpec(
        'KIT__11__WalkInCounterClockwiseCircle04_poses',
        'KIT/11/WalkInCounterClockwiseCircle04_poses.npz', 'circle_ccw',
    ),
    MotionSpec(
        'KIT__11__WalkInCounterClockwiseCircle10_poses',
        'KIT/11/WalkInCounterClockwiseCircle10_poses.npz', 'circle_ccw',
    ),
    MotionSpec(
        'KIT__4__WalkInCounterClockwiseCircle04_poses',
        'KIT/4/WalkInCounterClockwiseCircle04_poses.npz', 'circle_ccw',
    ),
    MotionSpec(
        'KIT__7__WalkInCounterClockwiseCircle03_poses',
        'KIT/7/WalkInCounterClockwiseCircle03_poses.npz', 'circle_ccw',
    ),
    MotionSpec(
        'KIT__7__WalkInCounterClockwiseCircle06_poses',
        'KIT/7/WalkInCounterClockwiseCircle06_poses.npz', 'circle_ccw',
    ),
    MotionSpec(
        'KIT__8__WalkInCounterClockwiseCircle04_poses',
        'KIT/8/WalkInCounterClockwiseCircle04_poses.npz', 'circle_ccw',
    ),
    MotionSpec(
        'KIT__9__WalkInCounterClockwiseCircle06_poses',
        'KIT/9/WalkInCounterClockwiseCircle06_poses.npz', 'circle_ccw',
    ),
    MotionSpec(
        'KIT__9__WalkInCounterClockwiseCircle10_poses',
        'KIT/9/WalkInCounterClockwiseCircle10_poses.npz', 'circle_ccw',
    ),
    # circle_cw (8 motions)
    MotionSpec(
        'KIT__11__WalkInClockwiseCircle01_poses',
        'KIT/11/WalkInClockwiseCircle01_poses.npz', 'circle_cw',
    ),
    MotionSpec(
        'KIT__11__WalkInClockwiseCircle04_poses',
        'KIT/11/WalkInClockwiseCircle04_poses.npz', 'circle_cw',
    ),
    MotionSpec(
        'KIT__4__WalkInClockwiseCircle07_poses',
        'KIT/4/WalkInClockwiseCircle07_poses.npz', 'circle_cw',
    ),
    MotionSpec(
        'KIT__7__WalkInClockwiseCircle04_poses',
        'KIT/7/WalkInClockwiseCircle04_poses.npz', 'circle_cw',
    ),
    MotionSpec(
        'KIT__7__WalkInClockwiseCircle07_poses',
        'KIT/7/WalkInClockwiseCircle07_poses.npz', 'circle_cw',
    ),
    MotionSpec(
        'KIT__8__WalkInClockwiseCircle06_poses',
        'KIT/8/WalkInClockwiseCircle06_poses.npz', 'circle_cw',
    ),
    MotionSpec(
        'KIT__9__WalkInClockwiseCircle08_poses',
        'KIT/9/WalkInClockwiseCircle08_poses.npz', 'circle_cw',
    ),
    MotionSpec(
        'KIT__9__WalkInClockwiseCircle09_poses',
        'KIT/9/WalkInClockwiseCircle09_poses.npz', 'circle_cw',
    ),
    # run (8 motions)
    MotionSpec(
        'KIT__183__walking_run02_poses',
        'KIT/183/walking_run02_poses.npz', 'run',
    ),
    MotionSpec(
        'KIT__183__walking_run05_poses',
        'KIT/183/walking_run05_poses.npz', 'run',
    ),
    MotionSpec(
        'KIT__3__walking_run02_poses',
        'KIT/3/walking_run02_poses.npz', 'run',
    ),
    MotionSpec(
        'KIT__317__walking_run01_poses',
        'KIT/317/walking_run01_poses.npz', 'run',
    ),
    MotionSpec(
        'KIT__317__walking_run06_poses',
        'KIT/317/walking_run06_poses.npz', 'run',
    ),
    MotionSpec(
        'KIT__424__walking_run02_poses',
        'KIT/424/walking_run02_poses.npz', 'run',
    ),
    MotionSpec(
        'KIT__424__walking_run03_poses',
        'KIT/424/walking_run03_poses.npz', 'run',
    ),
    MotionSpec(
        'KIT__9__walking_run06_poses',
        'KIT/9/walking_run06_poses.npz', 'run',
    ),
    # turn_left (8 motions)
    MotionSpec(
        'KIT__11__LeftTurn06_poses',
        'KIT/11/LeftTurn06_poses.npz', 'turn_left',
    ),
    MotionSpec(
        'KIT__12__LeftTurn08_poses',
        'KIT/12/LeftTurn08_poses.npz', 'turn_left',
    ),
    MotionSpec(
        'KIT__183__turn_left05_poses',
        'KIT/183/turn_left05_poses.npz', 'turn_left',
    ),
    MotionSpec(
        'KIT__317__turn_left07_poses',
        'KIT/317/turn_left07_poses.npz', 'turn_left',
    ),
    MotionSpec(
        'KIT__424__turn_left05_poses',
        'KIT/424/turn_left05_poses.npz', 'turn_left',
    ),
    MotionSpec(
        'KIT__7__LeftTurn03_poses',
        'KIT/7/LeftTurn03_poses.npz', 'turn_left',
    ),
    MotionSpec(
        'KIT__7__LeftTurn07_poses',
        'KIT/7/LeftTurn07_poses.npz', 'turn_left',
    ),
    MotionSpec(
        'KIT__9__turn_left01_poses',
        'KIT/9/turn_left01_poses.npz', 'turn_left',
    ),
    # turn_right (8 motions)
    MotionSpec(
        'KIT__11__RightTurn07_poses',
        'KIT/11/RightTurn07_poses.npz', 'turn_right',
    ),
    MotionSpec(
        'KIT__12__RightTurn06_poses',
        'KIT/12/RightTurn06_poses.npz', 'turn_right',
    ),
    MotionSpec(
        'KIT__183__turn_right09_poses',
        'KIT/183/turn_right09_poses.npz', 'turn_right',
    ),
    MotionSpec(
        'KIT__3__turn_right09_poses',
        'KIT/3/turn_right09_poses.npz', 'turn_right',
    ),
    MotionSpec(
        'KIT__317__turn_right10_poses',
        'KIT/317/turn_right10_poses.npz', 'turn_right',
    ),
    MotionSpec(
        'KIT__424__turn_right02_poses',
        'KIT/424/turn_right02_poses.npz', 'turn_right',
    ),
    MotionSpec(
        'KIT__7__RightTurn03_poses',
        'KIT/7/RightTurn03_poses.npz', 'turn_right',
    ),
    MotionSpec(
        'KIT__8__RightTurn06_poses',
        'KIT/8/RightTurn06_poses.npz', 'turn_right',
    ),
    # walk_backward (8 motions)
    MotionSpec(
        'KIT__11__WalkingStraightBackwards03_1_poses',
        'KIT/11/WalkingStraightBackwards03_1_poses.npz', 'walk_backward',
    ),
    MotionSpec(
        'KIT__11__WalkingStraightBackwards04_1_poses',
        'KIT/11/WalkingStraightBackwards04_1_poses.npz', 'walk_backward',
    ),
    MotionSpec(
        'KIT__12__WalkingStraightBackwards05_poses',
        'KIT/12/WalkingStraightBackwards05_poses.npz', 'walk_backward',
    ),
    MotionSpec(
        'KIT__12__WalkingStraightBackwards07_poses',
        'KIT/12/WalkingStraightBackwards07_poses.npz', 'walk_backward',
    ),
    MotionSpec(
        'KIT__4__WalkingStraightBackwards05_poses',
        'KIT/4/WalkingStraightBackwards05_poses.npz', 'walk_backward',
    ),
    MotionSpec(
        'KIT__7__WalkingStraightBackwards06_poses',
        'KIT/7/WalkingStraightBackwards06_poses.npz', 'walk_backward',
    ),
    MotionSpec(
        'KIT__8__WalkingStraightBackwards09_poses',
        'KIT/8/WalkingStraightBackwards09_poses.npz', 'walk_backward',
    ),
    MotionSpec(
        'KIT__9__WalkingStraightBackwards03_poses',
        'KIT/9/WalkingStraightBackwards03_poses.npz', 'walk_backward',
    ),
    # walk_fast (8 motions)
    MotionSpec(
        'KIT__183__walking_fast01_poses',
        'KIT/183/walking_fast01_poses.npz', 'walk_fast',
    ),
    MotionSpec(
        'KIT__183__walking_fast04_poses',
        'KIT/183/walking_fast04_poses.npz', 'walk_fast',
    ),
    MotionSpec(
        'KIT__3__walking_fast06_poses',
        'KIT/3/walking_fast06_poses.npz', 'walk_fast',
    ),
    MotionSpec(
        'KIT__3__walking_fast10_poses',
        'KIT/3/walking_fast10_poses.npz', 'walk_fast',
    ),
    MotionSpec(
        'KIT__317__walking_fast06_poses',
        'KIT/317/walking_fast06_poses.npz', 'walk_fast',
    ),
    MotionSpec(
        'KIT__317__walking_fast09_poses',
        'KIT/317/walking_fast09_poses.npz', 'walk_fast',
    ),
    MotionSpec(
        'KIT__424__walking_fast02_poses',
        'KIT/424/walking_fast02_poses.npz', 'walk_fast',
    ),
    MotionSpec(
        'KIT__424__walking_fast07_poses',
        'KIT/424/walking_fast07_poses.npz', 'walk_fast',
    ),
    # walk_medium (8 motions)
    MotionSpec(
        'KIT__11__WalkingStraightForwards09_poses',
        'KIT/11/WalkingStraightForwards09_poses.npz', 'walk_medium',
    ),
    MotionSpec(
        'KIT__12__WalkingStraightForwards09_1_poses',
        'KIT/12/WalkingStraightForwards09_1_poses.npz', 'walk_medium',
    ),
    MotionSpec(
        'KIT__183__walking_medium03_poses',
        'KIT/183/walking_medium03_poses.npz', 'walk_medium',
    ),
    MotionSpec(
        'KIT__3__walking_medium04_poses',
        'KIT/3/walking_medium04_poses.npz', 'walk_medium',
    ),
    MotionSpec(
        'KIT__424__walking_medium10_poses',
        'KIT/424/walking_medium10_poses.npz', 'walk_medium',
    ),
    MotionSpec(
        'KIT__7__WalkingStraightForwards10_poses',
        'KIT/7/WalkingStraightForwards10_poses.npz', 'walk_medium',
    ),
    MotionSpec(
        'KIT__8__WalkingStraightForwards10_poses',
        'KIT/8/WalkingStraightForwards10_poses.npz', 'walk_medium',
    ),
    MotionSpec(
        'KIT__9__WalkingStraightForwards06_poses',
        'KIT/9/WalkingStraightForwards06_poses.npz', 'walk_medium',
    ),
    # walk_slow (8 motions)
    MotionSpec(
        'KIT__183__walking_slow02_poses',
        'KIT/183/walking_slow02_poses.npz', 'walk_slow',
    ),
    MotionSpec(
        'KIT__183__walking_slow03_poses',
        'KIT/183/walking_slow03_poses.npz', 'walk_slow',
    ),
    MotionSpec(
        'KIT__3__walking_slow03_poses',
        'KIT/3/walking_slow03_poses.npz', 'walk_slow',
    ),
    MotionSpec(
        'KIT__317__walking_slow10_poses',
        'KIT/317/walking_slow10_poses.npz', 'walk_slow',
    ),
    MotionSpec(
        'KIT__4__WalkingStraightForward08_poses',
        'KIT/4/WalkingStraightForward08_poses.npz', 'walk_slow',
    ),
    MotionSpec(
        'KIT__424__walking_slow01_poses',
        'KIT/424/walking_slow01_poses.npz', 'walk_slow',
    ),
    MotionSpec(
        'KIT__424__walking_slow10_poses',
        'KIT/424/walking_slow10_poses.npz', 'walk_slow',
    ),
    MotionSpec(
        'KIT__9__WalkingStraightForwards02_poses',
        'KIT/9/WalkingStraightForwards02_poses.npz', 'walk_slow',
    ),
)

MOTION_SETS: dict[str, tuple[MotionSpec, ...]] = {
    "kit_flat_locomotion_72": KIT_FLAT_LOCOMOTION_72,
}


# Map excluded motion IDs to reasons. Apply exclusions to all motion consumers.
# Keep the original catalog and output directory names.
MOTION_EXCLUSIONS: dict[str, str] = {
    motion.motion_id: "Exclude running from standard experiments."
    for motion in KIT_FLAT_LOCOMOTION_72
    if motion.category == "run"
}


def get_motion_specs(*, motion_set: str) -> tuple[MotionSpec, ...]:
    """
    Return a registered motion set after explicit exclusions.

    Require unique IDs and paths. Reject empty selections.
    Keep motion selection independent of target subject splits.
    """
    if motion_set not in MOTION_SETS:
        raise ValueError(f"Unknown or unregistered motion set: {motion_set!r}")
    specs = tuple(MOTION_SETS[motion_set])
    if not specs:
        raise ValueError(f"Motion set is empty: {motion_set!r}")
    if len({spec.motion_id for spec in specs}) != len(specs):
        raise ValueError(f"Duplicate motion IDs in {motion_set!r}")
    if len({spec.source_path for spec in specs}) != len(specs):
        raise ValueError(f"Duplicate source paths in {motion_set!r}")
    
    registered = {spec.motion_id for entries in MOTION_SETS.values() for spec in entries}

    for motion_id, reason in MOTION_EXCLUSIONS.items():
        if motion_id not in registered:
            raise ValueError(f"Excluded motion is not registered: {motion_id!r}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Missing exclusion reason for {motion_id!r}")
    selected = tuple(spec for spec in specs if spec.motion_id not in MOTION_EXCLUSIONS)
    if not selected:
        raise ValueError(f"Motion selection is empty after exclusions: {motion_set!r}")
    return selected
