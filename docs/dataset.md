# Datasets

MyoAcross uses KIT and BMLmovi source data in AMASS format.
Use [datalists.py](../myoacross/utils/datalists.py) for subject splits and motion selection.

## Source Layout

Store source files under `data/dataset/`:

```text
data/dataset/
  KIT/
    subjects.json
    <subject>/
      <motion>_poses.npz
  BMLmovi/
    subjects.json
    Subject_<S>_F_MoSh/
      shape.npz
      Subject_<S>_F_<N>_poses.npz
```

Preserve dataset names, subject names, and source file names.
In BMLmovi file names, `N` identifies a clip for one subject.
It does not identify the same action across subjects.

Keep the tracked `subjects.json` files.
They record subject attributes, including gender, age, height, and body mass.
Keep these attributes separate from fitted body shape parameters.

`get_subject_reference_motion()` selects a subject's shape reference.
It selects `shape.npz` when the file exists.
Otherwise, it selects the first `*_poses.npz` file in name order.
Measurement extraction reads `gender` and `betas` from this reference.
Treat `shape.npz` as a shape reference, not as a registered motion clip.

Read [Installation and Data Preparation](install.md) for environments and model parameters.

## Target Subjects

A target subject identifies a body for retargeting, training, or evaluation.
The target catalog contains 55 KIT subjects and 86 BMLmovi subjects.
The fixed split contains 110 training subjects and 31 test subjects.

| Item | Definition in `datalists.py` |
| --- | --- |
| Training subjects | `TRAIN_TARGET_SUBJECT_IDS` |
| Test subjects | `TEST_TARGET_SUBJECT_IDS` |
| Subject exclusions | `TARGET_SUBJECT_EXCLUSIONS` |
| Effective selection | `get_target_subject_ids()` |

The original split used seed 42.
The code stores the split membership explicitly.
Do not recalculate it from files found on disk.
The current subject exclusion map is empty.
Add an exclusion reason when a subject must be excluded.
Keep its original split membership.

Use IDs such as `KIT_3` and `BMLmovi_Subject_1_F_MoSh`.
Use the same split for `baseline`, `height_scale`, and `segment_scale`.

## Source Motions

A source motion identifies a reference clip and its original performer.
Keep this identity separate from the target body identity.
Training and test bodies use the same selected source motions.
The current protocol tests transfer to held-out bodies.
It does not define a held-out motion split.

The registered set `kit_flat_locomotion_72` contains 72 KIT clips.
Each of the nine categories contains eight clips:

| Category | Active clips |
| --- | --- |
| `circle_ccw` | 8 |
| `circle_cw` | 8 |
| `run` | 0 |
| `turn_left` | 8 |
| `turn_right` | 8 |
| `walk_backward` | 8 |
| `walk_fast` | 8 |
| `walk_medium` | 8 |
| `walk_slow` | 8 |

`MOTION_EXCLUSIONS` excludes the eight running clips from standard experiments.
`get_motion_specs()` returns the remaining 64 clips.
Keep the registered set name unchanged when applying exclusions.
The set name is a selection identifier, not an additional output directory.

Each `MotionSpec` records a motion ID, source path, and category.
The source path is relative to `data/dataset/`.
For example, `KIT__3__walking_fast06_poses` refers to `KIT/3/walking_fast06_poses.npz`.
BMLmovi supplies target body shapes.
The current source motion catalog contains KIT clips only.

## Selection Changes

Maintain fixed subject and motion selections in `datalists.py`.
Select the registered motion set through `data.motion_set` in training configurations.
Use the shared exclusion maps instead of repeating exclusion lists in YAML files.

Before adding a motion, check its source file and category.
Retarget it to the required target bodies before using it for training or evaluation.
A body bundle does not contain the retargeted motion files.
Record changes to the experiment protocol in [Paper Experiments](experiments.md).
Read [Retargeting Development](retarget.md) for export paths and motion screening.
