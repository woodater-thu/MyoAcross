# Embodiment Development

## Responsibilities

Generate MyoFullBody models from subject measurements and body mass.
Keep these supported strategies:

| Strategy | Method |
| --- | --- |
| `baseline` | Use the original MyoFullBody model. |
| `height_scale` | Optimize one global scale from 17 skeletal measurements. |
| `segment_scale` | Optimize regional scales from skeletal measurements. |

Both fitted strategies divide measurement residuals by the configured tolerances.
Both strategies use a prior on the logarithm of scale.
Use shoulder–ankle height ratios for the prior and initial scale.
The `height_scale` strategy accepts every successful solver result.
Record its measurement errors.
Do not reject a successful fit because of measurement error.
The `segment_scale` strategy retains its measurement quality acceptance check.

## Measurements

`embodiment/smplh_to_measurements.py` reads subject shape and metadata.
It evaluates SKEL in a T-pose and exports measurement targets.
Store subject measurement JSON files in `data/myoacross/embodiments/measurements/`.
`embodiment/measurements.py` measures the compiled MuJoCo model.
Both implementations use the registered measurement definitions.
Keep the shared measurement definitions in `embodiment/smplh_to_measurements.py`.
Import SKEL and Torch only when computing T-pose geometry.
Run target measurement generation in the SKEL environment:

```bash
uv run --project tools/skel python -m myoacross measure --split train
uv run --project tools/skel python -m myoacross measure --split test
```

The fitted comparison uses 17 skeletal measurements.
The additional height field is not one of these 17 measurements.
Preserve units and left/right identities in reports.

## Model Generation

`generate.py` selects registered target subjects and dispatches the strategy.
`height_scale/model.py` performs single-scale optimization.
`segment_scale/model.py` performs regional optimization.
Validate targets and calculate the prior in each strategy's `solve()` method.
Use `OptimizationFailure` from `embodiment/model.py` for solver failures.
`export.py` writes the model and report.
Measure the final compiled model and validate its physical parameters.
Keep scale bounds, prior weight, measurement tolerances, and solver limits in the configuration.
Keep the standard segment-scale configuration in `myoacross/configs/embodiment/`.
Keep baseline and global-scale comparison configurations in `myoacross/configs/experiments/skeletal_fitting/`.
Use symmetry, adjacency, and measurement quality limits only for `segment_scale`.

```bash
uv run python -m myoacross embodiment --config myoacross/configs/embodiment/segment_scale.yaml --split train
```

Use `--help` for subject selection and overwrite options.
Use `--split test` for held-out target subjects.

## Extension and Tests

Implement model construction inside the applicable embodiment strategy.
Use shared measurements and export functions.
Preserve mass calibration, model topology, and acceptance checks during structural changes.
Check subject selection and generated reports with a small subject set.
Read [Installation](install.md) for SKEL dependencies.

## Global Scale Generation

```bash
uv run python -m myoacross embodiment --config myoacross/configs/experiments/skeletal_fitting/height_scale.yaml --split train --subject KIT_1487
```

The report identifies the method as `global_measurement_fit`.
It records the uniform scale, solver status, active bound, and 17 fitting errors.
A failed solver saves a failure report without a model XML.
The global root remains fixed. All other bodies use the same scale.
Keep the subject mass calibration and shared muscle scaling procedure.

Generate matching retargeted motions after changing a body model.
Clear or move existing outputs before replacing them.
