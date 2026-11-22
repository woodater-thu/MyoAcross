# Retargeting Development

## Procedure

1. Fit neutral SMPL-H shape to the target body.
2. Estimate scale and local alignment offsets.
3. Apply the fitted shape to the source motion.
4. Solve the two inverse kinematics stages.
5. Resample the motion and correct ground penetration.
6. Export tracking arrays, derivatives, and quality information.

This procedure transfers kinematic reference motion.
It does not calculate muscle activations.

## Modules

| Module | Responsibility |
| --- | --- |
| `run.py` | Collect target-motion jobs and control execution. |
| `shape_fitting.py` | Fit and cache target-specific SMPL-H parameters. |
| `retargeter.py` | Construct retargeters and solve motion frames. |
| `postprocess.py` | Resample and correct the result. |
| `export.py` | Build and save tracking arrays, reports and numerical screening. |
| `acceleration.py` | Accelerate IK and source-frame processing. |
| `viewer.py` | Replay exported motion with its target model. |

Shape-fit cache keys include the compiled model and fitting settings.
Equal compiled models can share a fitted shape.
Do not reuse a cache entry when its model or fitting settings differ.
Reset per-motion state when reusing a retargeter.
Select source-shape alignment explicitly when that method is required.
The default configuration uses target shape fitting.

## Output Paths

Store motions in `data/myoacross/retargets/<embodiment>/<subject_id>/<motion_id>/`.
Store baseline motions directly in `retargets/baseline/<motion_id>/`.
Store `motion.npz` and `report.json` in each motion directory.
Keep fitted-shape caches in `data/myoacross/retargets/fitted_shapes/`.
Record the method and motion-set identity in metadata. Do not add them as path layers.
Select another `--output-root` when maintaining a separate motion collection.

## Jobs and Quality

Train/test selects target subjects.
Each selected target uses the same selected source motion library.
Use the target's fitted reference for its RL environment.
Do not pair a target with its source performer's model by mistake.

```bash
uv run --project tools/gmr_fit python -m myoacross retarget \
  --config myoacross/configs/retarget/gmr_fit.yaml --embodiment segment_scale --split train
```

Use `--dry-run` to inspect jobs.
Use `--skip-completed` to reuse complete products with matching identity and configuration.
Use `--workers` for parallel target processing.
Keep motions within each target sequential to reuse fitting and inverse kinematics state.

Export screening checks finite values, rotations, ground correction, and root trajectories.
It also checks body speeds and foot IK errors.
A completed solve can still fail quality screening.
Do not discard failed motions silently.

## Developer Checks

Use the GMR-Fit environment for retargeting tests.
Compare array keys, shapes, and numerical values after an implementation change.
Use repeated motions to detect cached state leaks.
Preserve quality thresholds during structural changes.

## Export Screening

Call `annotate_exported_result()` after `save_result()` completes.
Record numerical screening and its actual thresholds in `report.json`.
Do not save source, target model, output motion, or configuration hashes in retarget reports.
Use file presence, identity, configuration, and recorded screening when reusing outputs.
RL loading checks model acceptance, motion identity, paths, and recorded quality.
These checks do not detect content changes at unchanged paths.

Generate a dataset inventory with the auxiliary command:

```bash
.venv/bin/python -m myoacross.utils.analysis.retarget_manifest \
  --embodiment segment_scale --output /tmp/myoacross-retarget-manifest.json
```

The inventory checks file presence and reads recorded screening results.
It does not recompute numerical metrics.
The RL catalog fingerprint still includes report content and model XML bytes.
Changing reports therefore affects checkpoint recovery.
Do not rewrite checkpoint fingerprints automatically.
