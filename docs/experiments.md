# Paper Experiments

## Common Protocol

Inherit the applicable standard RL configuration. Keep only experiment overrides.
Inspect resolved settings with the [configuration command](development.md#command-checks) before changing experiment conditions.

Use the standard configuration for the MyoAcross FiLM policy.
It uses 10 virtual bodies per training subject, the reference generation prior, and seed 42.
Current virtual training includes each original subject bundle in addition to generated bodies.
The standard training set has 11 candidate bodies per subject.
Sample the candidates uniformly within each subject.
Record this protocol change when comparing with earlier virtual-only training runs.
Train on four GPUs with 2048 environments and eight body groups per GPU.
Each body group has 256 environments.
Train for 15000 iterations.
The paper motion selection contains 64 motions.
Select `kit_flat_locomotion_72` in configurations.
Use the fixed subjects and motion exclusions in [Datasets](dataset.md).
The subject split contains 110 training subjects and 31 test subjects.
Do not train on test subjects.

Resolve data and output paths relative to the repository root.
Run all commands from that root.
Install the environments as specified in [install.md](install.md).
Multi-body training and evaluation automatically prepare an isolated backend before task initialization.

Use the following standard workflow:

```bash
.venv/bin/python -m myoacross rl generate --config myoacross/configs/rl/generate.yaml
.venv/bin/torchrun --standalone --nproc-per-node=4 -m myoacross rl train --config myoacross/configs/rl/train.yaml
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/rl/evaluate.yaml
```

Use `data/myoacross/rl/` for standard outputs.
Use `data/experiments/<experiment_name>/` for experiment outputs.
Check that the standard `checkpoint/final.pt` stores iteration 15000.
Follow [Fixed Outputs and Recovery](rl.md#fixed-outputs-and-recovery) before starting or resuming a run.

Generate each bank once.
Reuse it when the body count, covariance, bounds, seed, and source models match.
Read generation settings and source identity from `bank.json`, `index.json`, and bundle metadata.
Generation does not save a separate configuration copy or source-code revision record.

## Reference Generation Prior

Use the following local perturbation prior around each original subject model.
The prior is a candidate reference for morphology augmentation.
It is not a measured joint distribution of healthy adults.
Original subjects already contain between-subject variation. These factors add within-subject variation.

| Covariance field | Standard value | Role |
| --- | --- | --- |
| `global_size_std` | 0.01 | Shared size variation. |
| `axial_std` | 0.005 | Shared axial variation. |
| `upper_limb_std` | 0.005 | Shared upper-limb variation. |
| `lower_limb_std` | 0.005 | Shared lower-limb variation. |
| `region_std` | 0.008 for upper arm, forearm, thigh, and shank; 0.005 for the other regions | Regional variation. |
| `asymmetry_std` | 0.005 | Opposite left and right changes in paired regions. |
| `mass_size_coefficient` | 2.0 | Mass response to the shared size factor. |
| `mass_residual_std` | 0.03 | Mass variation independent of the size factor. |

Use relative shape bounds of 0.06 and a relative mass bound of 0.15 for the normal reference group.
These bounds apply to parameter changes, not to absolute population measurements.
Keep `max_attempts: 1000`.

A Q3D-CT study of 20 healthy participants reports a mean absolute tibial length difference of about 0.77%.
Use this result as a lower-limb reference, not as a population-wide calibration ([Vuurberg et al., 2022](https://doi.org/10.1007/s00276-022-02940-9)).
Assume a zero-mean Gaussian asymmetry factor with standard deviation 0.005.
The first-order expected absolute left-right difference is about 0.80% before rejection.
This estimate describes the added change. The original model's asymmetry also affects the final body.
A skeletal study reports different asymmetry patterns for upper and lower limbs, including directional upper-limb asymmetry ([Auerbach and Ruff, 2006](https://pubmed.ncbi.nlm.nih.gov/16310833/)).
One shared zero-mean factor setting does not reproduce all these regional distributions.

Adult population analyses report body-weight scaling exponents close to 2 after adjustment for age and adiposity ([Heymsfield et al., 2014](https://ajcn.nutrition.org/article/S0002-9165%2823%2904846-3/fulltext)).
Use `mass_size_coefficient: 2.0` as a first-order augmentation assumption.
The remaining factor standard deviations and bounds are engineering choices. They have not been fitted to a healthy population.
Report measured distributions from the generated banks before making population claims.

The previous generation prior used `asymmetry_std: 0.002` and `mass_size_coefficient: 3.0`.
Changing YAML does not update existing banks or checkpoints.
Read the actual prior from bank metadata when analyzing stored results.
Clear or relocate old outputs explicitly before generating replacement banks.
Use the reference prior for the normal group.
Apply only the specified asymmetry and bound changes for the enhanced group.
Keep earlier checkpoints as results of their recorded training protocol. Train again to measure the revised prior.

## Skeletal Fitting

**Hypothesis:** Segment-specific scaling improves agreement with subject measurements.

Compare the unchanged MyoFullBody model, global scaling, and segment scaling.
The current `height_scale` implementation fits one scale from all 17 measurements.
Use `height_scale.yaml` in `myoacross/configs/experiments/skeletal_fitting/` for that method.
Use `baseline.yaml` in the same directory to export the original MyoFullBody model.
Use all 141 subjects and the 17 fitted measurements.
Store mean absolute measurement error in metres. Plot errors in millimetres.
This analysis uses existing fitted models. It does not require an RL configuration.
Keep the standard `segment_scale.yaml` in `myoacross/configs/embodiment/`.
Generate the comparison models with these commands:

```bash
.venv/bin/python -m myoacross embodiment --config myoacross/configs/experiments/skeletal_fitting/baseline.yaml
.venv/bin/python -m myoacross embodiment --config myoacross/configs/experiments/skeletal_fitting/height_scale.yaml --split train
.venv/bin/python -m myoacross embodiment --config myoacross/configs/experiments/skeletal_fitting/height_scale.yaml --split test
```

The baseline has no subject ID. Generate it once.
The configuration move does not change model output paths.
Reuse existing models when their generation settings match.

```bash
.venv/bin/python -m myoacross.utils.analysis.compute_skeletal_fitting
```

The default output is `data/experiments/skeletal_fitting/evaluations/report/`.
Use `--output` to select another directory. The command rejects an existing directory.
Read `results.json` for subject errors, split summaries, and source report paths.
The command also exports three independent figures:

| Figure | Content |
| --- | --- |
| `subject_mae` | Box plots and subject points for the three methods. |
| `measurement_mae` | Mean absolute error for each of the 17 measurements. |
| `paired_subject_mae` | Global and segment MAE for each matched subject. |

Each figure uses millimetres and includes the subject count.
Use the SciencePlots styles `science`, `nature`, `no-latex`, and `bright`.
Install the optional analysis dependency as specified in [Installation](install.md#analysis-figures).
The boxes show the median and the first and third quartiles.
The whiskers end at the most distant values within 1.5 interquartile ranges.
The points show all subjects, including values beyond the whiskers.
Each figure is saved as editable SVG, PDF, 600 dpi TIFF, and preview PNG.
The paired scatter includes an equality line and the fraction with lower segment error.
Use subjects as the statistical unit. Do not treat measurements as independent subjects.

Record the model method with each result.
Use `global_measurement_fit` for the current global-scale solver.
Do not combine height-ratio results with results from this solver.
Report geometric fitting error separately from policy tracking error.
No current fitting result table is included here.

## Policy Comparison

Store the original MyoFullBody bank in `data/experiments/policy_comparison/embodiments_bundles/myofullbody/subject/`.
Prepare its source model with the shared baseline command in [Skeletal Fitting](#skeletal-fitting).

**Hypothesis:** Morphology conditioning improves motion tracking across unseen bodies.

Compare MyoFullBody MLP, segment-scale MLP, Concat, and FiLM.
Keep the shared network and PPO settings fixed.
Use the same motion set and training budget.
Change only the settings required by each policy scheme.
For Concat, encode morphology as an observation group with widths `[64, 128, 256]`.
Use separate actor and critic group encoders.
Use fixed training-set morphology statistics before encoding.
Concatenate the encoded groups before the backbone input projection.
The MyoFullBody scheme uses `baseline` subject models, encoder mode `none`, and one body group per GPU.
The other schemes use the shared standard V10 bank.
Evaluate all policies on segment-scale test bodies and their matching reference motions.
Do not fine-tune the MyoFullBody policy during transfer evaluation.
Keep the checkpoint normalization statistics.

Configurations are in `myoacross/configs/experiments/policy_comparison/`.
The standard FiLM configuration remains in `configs/rl/`; do not duplicate it.

```bash
.venv/bin/python -m myoacross rl generate --config myoacross/configs/experiments/policy_comparison/generate_myofullbody_train.yaml
.venv/bin/torchrun --standalone --nproc-per-node=4 -m myoacross rl train --config myoacross/configs/experiments/policy_comparison/train_myofullbody_mlp.yaml
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/experiments/policy_comparison/evaluate_myofullbody_mlp_test.yaml
```

Use `train_segment_mlp.yaml` and `train_segment_concat.yaml` for the other schemes.
Use their matching `evaluate_<scheme>_test.yaml` files.
The `evaluate_<scheme>_train.yaml` files provide training-split diagnostics.

Apply [Result Interpretation](#result-interpretation) to the comparison.
No verified policy comparison result table is included here.

## Virtual Body Count

**Hypothesis:** Virtual morphology diversity changes generalization to unseen subjects.

Compare V0, V5, and V10: zero, five, and ten virtual bodies per subject.
Include one original subject body in each scheme.
The corresponding training candidate counts are 1, 6, and 11 per subject.
The original body's sampling probabilities are 1, 1/6, and 1/11.
This experiment changes both virtual diversity and the original body's sampling share.
Keep earlier virtual-only results separate from results under this protocol.
Use the reference generation prior for V5 and V10.
Keep seed 42 and the FiLM policy.
For V0, use `data.morphology_source: subject` and reuse the standard subject bank.
V0 does not require virtual generation or a separate generation configuration.
Generate V5 with the current reference prior.
Store the bank in `data/experiments/virtual_embodiment_count/embodiments_bundles/virtual5/`.
Do not reuse V5 banks with different sampling settings.
Do not select subsets from the V10 bank.
Reuse the standard configuration and bank for V10.

Configurations are in `myoacross/configs/experiments/virtual_embodiment_count/`.

```bash
.venv/bin/torchrun --standalone --nproc-per-node=4 -m myoacross rl train --config myoacross/configs/experiments/virtual_embodiment_count/train_virtual0_film.yaml
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/experiments/virtual_embodiment_count/evaluate_virtual0_film_test.yaml
```

For V5, generate with `generate_virtual5_train.yaml`, train with `train_virtual5_film.yaml`, and evaluate with `evaluate_virtual5_film_test.yaml`.
Use the standard workflow for V10.
Compare all counts at the same completed training iteration.
Apply [Result Interpretation](#result-interpretation) to the comparison.
No verified body-count result table is included here.

## Input Sensitivity

**Hypothesis:** MyoAcross retains motion tracking performance under biased morphology and reference inputs.

Evaluate held-out subjects with unchanged physical models and source motions.
Compare segment MLP, segment Concat, and segment FiLM under the same V10 training protocol.
Use V0 and V5 FiLM as additional body-count comparisons.
Keep the checkpoint and normalization statistics fixed.

| Target | Morphology magnitude | Reference magnitude |
| --- | --- | --- |
| `morphology` | 2%, 5%, 10% | 0 degrees |
| `reference` | 0% | 2.5, 5, 10 degrees |
| `combined` | 2%, 5%, 10% | 2.5, 5, 10 degrees, paired by level |

Run one clean baseline per subject and motion.
Run three sign samples at each nonzero level.
Each configuration has ten conditions per subject and motion.
For each sample, assign independent positive or negative signs to all 18 morphology features
and the 12 lower-limb angular joints.
Keep signs fixed within the episode. Reuse them across amplitudes and policies.
Apply the full signed magnitude. Do not sample a uniform interval around zero.
The MLP receives no morphology input. It remains a control under morphology biases.
For the MLP, combined conditions apply the reference bias only.

Use hip flexion, hip adduction, hip rotation, knee, ankle, and subtalar angles on both sides.
Preserve root motion and joint velocities. Do not clip biased targets to joint limits.
Recompute target body positions with the subject model.
Use biased joint targets and consistent future body positions for policy input.
Keep the clean reference for reset, reward, termination, and tracking metrics.
Do not retarget motions again or export biased motions.
Treat these levels as stress-test settings. They are not calibrated measurement-error distributions.

Configurations are in `myoacross/configs/experiments/input_sensitivity/`.
Use 64 parallel environments per subject.

```bash
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/experiments/input_sensitivity/evaluate_segment_film_morphology_test.yaml
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/experiments/input_sensitivity/evaluate_segment_film_reference_test.yaml
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/experiments/input_sensitivity/evaluate_segment_film_combined_test.yaml
```

Use matching configurations for the other policies.
Results use `data/experiments/input_sensitivity/evaluations/<scheme>/test_iter15000/<target>/`.
Report absolute completion, coverage, and tracking errors with paired baseline changes.
Average samples and motions within each subject. Weight subjects equally.
Read condition results from `sensitivity.csv` and sample details from `trials.csv`.
Interpret tracking errors with completion and coverage. Early failures shorten scored trajectories.
Do not treat motions or perturbation samples as independent subjects.
Use multiple training seeds before making claims across trained policies.
Do not infer clinical deployment performance from these synthetic biases.

Preserve completed ordinary and physical asymmetry evaluations.
This change does not require those evaluations to run again.
Keep old single-feature sensitivity results under their original output paths.
Do not use those results as evidence for this protocol.
No verified sensitivity result table is included here.

## Asymmetry Generalization

**Hypothesis:** Stronger morphology asymmetry during training improves transfer to asymmetric bodies.

Compare two generation priors. Use the normal reference prior as the control.
Use the enhanced prior as a proxy for pathological morphology.
The enhanced prior is not calibrated to a diagnosed patient population.

| Setting | Normal reference | Asymmetry enhanced |
| --- | --- | --- |
| `asymmetry_std` | 0.005 | 0.025 |
| Relative shape bound | 0.06 | 0.12 |
| Relative mass bound | 0.15 | 0.15 |
| Other covariance factors | Standard values | Standard values |
| Virtual bodies per subject | 10 | 10 |
| Candidate training bodies per subject | 11, including the original | 11, including the original |
| Seed | 42 | 42 |
| Training iterations | 15000 | 15000 |
| Policy | FiLM | FiLM |

The enhanced asymmetry factor has five times the reference standard deviation.
Before rejection, its expected absolute added left-right difference is about 4% for a paired region.
Actual differences also depend on the original subject and sampling bounds.
The current sampler applies independent asymmetry factors to upper arm, forearm, thigh, shank, and foot.
It does not enforce a shared affected side across regions.
The enhanced prior changes asymmetry and shape bounds together. This comparison does not isolate either change alone.
Do not describe this scheme as a complete model of disability or a specific diagnosis.
See [Reference generation prior](#reference-generation-prior) for the reference assumptions.

Reuse `myoacross/configs/rl/generate.yaml` and `train.yaml` for the normal group.
Keep its bank, checkpoint, and logs in `data/myoacross/rl/`.
Use `evaluate_segment_film_test.yaml` in this experiment for physical asymmetry evaluation.
Store enhanced-group configurations in `myoacross/configs/experiments/asymmetry_generalization/`.
Store its bank in `data/experiments/asymmetry_generalization/embodiments_bundles/asymmetry_enhanced/`.
Store checkpoints, logs, and evaluations under the scheme name `asymmetry_enhanced_film`.

Run the enhanced group with these commands:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m myoacross rl generate --config myoacross/configs/experiments/asymmetry_generalization/generate_asymmetry_enhanced_train.yaml
.venv/bin/torchrun --standalone --nproc-per-node=4 -m myoacross rl train --config myoacross/configs/experiments/asymmetry_generalization/train_asymmetry_enhanced_film.yaml
.venv/bin/python -m myoacross rl generate --config myoacross/configs/experiments/asymmetry_generalization/generate_asymmetry_test.yaml
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/experiments/asymmetry_generalization/evaluate_asymmetry_enhanced_film_test.yaml
```

Generation inherits `workers: 16` from the standard configuration.
Run one generation configuration at a time on a 16-core machine.
Set `workers: 1` in YAML to generate sequentially.
Generate the fixed test bank once with `generate_asymmetry_test.yaml`.
Use its inherited 16 subject workers. Both policies use this bank.
Do not repeat test-bank generation before the second policy evaluation.
Evaluate the normal policy with this command:

```bash
.venv/bin/python -m myoacross rl evaluate --config myoacross/configs/experiments/asymmetry_generalization/evaluate_segment_film_test.yaml
```

Use the same held-out subjects and motion selection for both policies.
Evaluate 21 physical conditions: the original model, and left or right shortening at every integer percentage from 1% through 10%. Use the two 15,000-iteration final checkpoints.
Multiply the selected side's thigh and shank region scales by `1 - relative_shortening`.
Keep all other region scales and total body mass unchanged.
Regional scaling also changes cross-section geometry, inertia, and muscle geometry.
It does not apply a length-only bone transformation.
The 0% condition retains the original subject's asymmetry.
These levels define controlled test conditions. They do not define clinical severity groups.

Reuse each original subject's retargeted motions in every condition.
Do not generate new retargeted motions for these bodies.
Supply measurements from the physical test model to the policy.
Keep policy weights and checkpoint normalization statistics unchanged.
This test measures control under changed body geometry with fixed source references.
Reference position errors can include the geometric difference between the test body and the source body.
Compare completion, coverage, and tracking errors together.

The test generation command creates fixed models under `data/experiments/asymmetry_generalization/embodiments/<condition>/segment_scale/<subject_id>/`.
The original condition references the existing subject XML.
Store test bundles under `data/experiments/asymmetry_generalization/embodiments_bundles/asymmetry_test/<condition>/`.
Save `conditions.json` at the test-bank root. Save `bank.json` and subject indices within each condition.
Both policies read these bundles. Evaluation does not create XML or export bundles.
Derive reported asymmetry from stored physical measurements.
Keep the shared test bank unchanged during evaluation.
Generation rejects existing output directories. Clear or relocate prior test models and banks before regeneration.
Training banks retain their current bundle-only output.
Store evaluations under `evaluations/<scheme>/test_iter15000/<condition>/`.
Each condition contains trial, subject, category, summary, progress, timing, and body metadata files.
The root `trials.csv` contains all conditions and paired `body_delta_<metric>` values against the original subject-motion trial.
The root `body_conditions.csv` contains subject-weighted metrics and their changes from the original condition.
The root `summary.json`, `subjects.csv`, and `categories.csv` describe only the original condition.
Record measured thigh, shank, and combined thigh-plus-shank asymmetry in the trial and condition tables.
Calculate asymmetry as `200 * abs(right - left) / (right + left)`.
Treat combined thigh-plus-shank length as a model proxy. It is not a clinical leg-length measurement.
No verified policy result table is available for this test yet.

The former covariance-scale configurations have been removed.
Existing covariance-scale outputs remain historical data at their original paths.
They are not banks or checkpoints for this two-group protocol.
Use the bank reuse requirements in [Common Protocol](#common-protocol).

Analyze the two stored banks without resampling. Reconstruct selected extreme bodies only for appearance figures:

```bash
.venv/bin/python -m myoacross.utils.analysis.analyze_asymmetry
```

Install the optional analysis dependency as specified in [Installation](install.md#analysis-figures).
The default output is `data/experiments/asymmetry_generalization/evaluations/sampling/`.
Both banks must exist. The command does not skip missing banks.
Use a new output directory. Existing directories are rejected.
To analyze only the enhanced bank, specify its generation configuration:

```bash
.venv/bin/python -m myoacross.utils.analysis.analyze_asymmetry --config myoacross/configs/experiments/asymmetry_generalization/generate_asymmetry_enhanced_train.yaml --output data/experiments/asymmetry_generalization/evaluations/asymmetry_enhanced
```

Repeat `--config` for a selected comparison.
Compare the same subjects, variant IDs, and original bodies across schemes.
Read actual sampling settings from the stored banks. Scheme labels identify the selected configurations.
Compute each parameter and feature change relative to its own original subject.
Keep original bodies out of the accepted virtual sample distribution.

The command exports these figures as editable SVG, PDF, 600 dpi TIFF, and PNG previews:

| Figure | Content |
| --- | --- |
| `parameter_distributions` | Relative generation parameters for each scheme. |
| `feature_distributions` | Relative measured lengths and mass for each scheme. |
| `accepted_spread` | Observed standard deviations of parameter and feature changes. |
| `accepted_correlations` | Parameter correlations on the same -1 to 1 color scale. |
| `extreme_bodies` | Original and maximum-deviation virtual bodies in front and side views. |

Select one extreme sample per scheme by the largest RMS percentage change across the 17 geometry features.
Exclude body mass from this selection. The selected sample need not contain every parameter's maximum change.
Use stored absolute generation parameters to reconstruct the selected body. Do not draw new perturbations.
Use the model default pose and the same camera distance for all bodies.
Align the framing to the lowest body position.
Read the subject, variant, bundle path, selection score, and parameter and feature changes from `extreme_samples.json`.
On a server without a display, prefix the analysis command with `MUJOCO_GL=egl` to use EGL rendering.

Boxes show the median and interquartile range.
Whiskers show the nearest observed values within the 2.5th and 97.5th percentiles.
Read per-sample percentage changes from `samples.csv`.
Read means, standard deviations, quantiles, prior spread, and prior and accepted correlations from `summary.json`.
A constant parameter has an undefined correlation, recorded as JSON `null`.
The samples remain grouped by source subject. These statistics do not provide independent-population confidence intervals.
The asymmetry factor variance increases by 25 before rejection. The other factor variances remain unchanged.
Rejection bounds also change. Report observed accepted spread separately from the prior.
Stored banks do not record rejected draws, so this analysis does not estimate acceptance rates.
These figures describe generated morphology distributions. They do not establish policy performance.
No verified tracking result table is included here.

## Result Interpretation

Use the definitions in [Evaluation Result Fields](rl.md#evaluation-result-fields).
Report unaligned position error, completion, and reference coverage together.
Tracking errors include executed frames only.
Early failures can reduce apparent tracking error.
Compare equally weighted subject means at the same completed training iteration.
Use repeated seeds before making claims of statistical significance.

## Result Records

For each result, record the hypothesis and configuration.
Record the code revision, checkpoint iteration, seed, subjects, and motions.
Identify the evaluation implementation.
State the units and aggregation method for each metric.
Keep raw per-trial results and uncertainty calculations with the summary.
Explain failed trials and missing cases before comparing schemes.
Record changes to the protocol before running a new study.
Preserve the actual iteration counts and source names in result records.

## Image2MSKSim

### Claim

Test whether a pretrained policy can control a model reconstructed from a real photograph.
Provide one image, measured height, and measured body mass. A selected video frame can supply the image.
Use HSMR to estimate body proportions. Use measured height to calibrate overall scale.
Height calibration does not recover body proportions that HSMR does not distinguish.
Do not fine-tune the policy for the individual.
Control success applies to the reconstructed model. It does not establish reconstruction accuracy against the person's anatomy.

### Method

Use the HSMR environment to infer shape parameters for the selected person.
Use the checkpoint's SKEL model identity.
Calculate the 17 measurements in the standard pose. Do not measure joint lengths from the image pose.
Calculate reconstructed height from the shaped SKEL skin template along its vertical axis.
Divide measured height by reconstructed height to obtain one scale factor.
Multiply all 17 length measurements by this factor. Keep measured body mass unchanged.
Keep the original HSMR parameters in `parameters.npz`. Do not scale the shape coefficients or image pose.
Record measured height, reconstructed height, and the scale factor in `source.json`.
Use measured body mass and the standard segment-scale fitting procedure.
Retarget the checkpoint's source motions to this new model.
Evaluate with the existing parallel evaluator and checkpoint normalization statistics.
Keep the new subject outside the fixed training and test subject lists.


### Licensed KIT motions for the single-image demo

The code package includes the prebuilt image-derived model but no AMASS motion files. Register at https://amass.is.tue.mpg.de/ and accept the AMASS terms. Install the root and GMR-Fit environments, the SMPL-H resources, and the standard V10 checkpoint. Place the checkpoint at `data/myoacross/rl/checkpoint/final.pt`. Then run:

```bash
.venv/bin/python -m myoacross.utils.image2msksim.prepare_single_image_demo
```

The default command prepares one normal walking clip and opens a policy-controlled viewer. It does not run full evaluation or image reconstruction. Use a graphical Linux session with an NVIDIA GPU. Close the viewer to finish.

Use `--headless` to evaluate the walking clip without a display. Use `--evaluate --headless` to retarget and evaluate all 64 reference motions. With `--evaluate` alone, the walking viewer opens after evaluation. Use `--device` to select the GPU.

When source motions are missing, the command obtains the official gender-specific SMPL-H KIT archive using your AMASS account and extracts the selected clips into `data/dataset/KIT/`. It keeps the archive under `data/downloads/amass/`. Credentials are not written to disk. Pass `--archive /path/to/KIT.tar.bz2` to use an existing archive. Use `--prepare-only` to stop after extraction; combine it with `--evaluate` to extract all 64 clips.

Outputs go to `data/experiments/single_image_demo/`. The command does not overwrite generated models. Select a fresh directory with `--output-root` or move the previous output directory before another preparation run. The packaged XML stores the individual model parameters; the demo resolves mesh paths against the installed `musclemimic_models` package and the texture path against `data/myoacross/assets/` in the generated XML. No mesh copies are required. Use the generated XML for direct playback.

Replay the prepared walking motion without downloading or retargeting again:

```bash
.venv/bin/python -m myoacross.play \
  --model data/experiments/single_image_demo/embodiments/segment_scale/demo_self_67kg/model.xml \
  --motion data/experiments/single_image_demo/retargets/segment_scale/demo_self_67kg/KIT__11__WalkingStraightForwards09_poses/motion.npz \
  --checkpoint checkpoints/standard_v10/final.pt \
  --camera front-side
```

Adjust the output and checkpoint paths if needed. The included photograph is anonymized; reconstruction used the original image and a measured mass of 67 kg. Third-party attribution remains under `data/demo/single_image/third_party/`.

### Configuration and Commands

Prepare the [HSMR environment and resources](install.md#image2msksim-environment).
Set `subject_id`, `height_m`, and `body_mass_kg` in `myoacross/configs/experiments/image2msksim/build.yaml`.
Specify height in meters and mass in kilograms.
Set exactly one of `image_path` and `video_path`. Leave the other field as `null`.
For video input, set `video_frame_index` to the selected frame. Frame indices start at zero.
Select a clear frame that shows the full body. The command does not select a frame automatically.
The command uses one frame for reconstruction. It does not use video motion for retargeting.
Use the HSMR environment's OpenCV package to decode the frame.
Save the frame as `reconstructions/<subject_id>/input.png`.
Record the video path, decoded frame index, and video frame rate in `source.json`.
Set `person_index` when the image contains multiple detected people. Detection indices start at zero.
Set `checkpoint_path` and `policy_scheme` for the policy under test.
Use `motion_ids: null` for the checkpoint's full motion selection.
Use a subset for a short check.

```bash
.venv/bin/python -m myoacross.utils.image2msksim.run \
  --config myoacross/configs/experiments/image2msksim/build.yaml
```

The command runs reconstruction, body construction, retargeting, and evaluation in separate processes.
Each stage uses its applicable environment.
Store outputs under `data/experiments/image2msksim/`:

| Directory | Content |
| --- | --- |
| `resources/` | HSMR weights, joint regressors, and a link to shared SKEL parameters |
| `cache/` | Detector and PyTorch caches |
| `reconstructions/<subject_id>/` | Shape parameters, image pose, detected bounding boxes, camera translation, and source information |
| `embodiments/measurements/` | Height-calibrated standard-pose measurements, measured height, and measured body mass |
| `embodiments/segment_scale/<subject_id>/` | Model XML and export report |
| `retargets/segment_scale/<subject_id>/<motion_id>/` | Reference motion and quality report |
| `evaluations/<policy_scheme>/test_iter<iteration>/` | Trials, subject and category summaries, evaluation settings, and subject bundles |
| `cache/backend/` | Shared isolated evaluation backend |

The output root can contain prepared resources and caches.
The full command creates `config.yaml` and fails if this file already exists.
It does not overwrite existing outputs.
Use the saved settings to run a separate stage after you resolve a failure:

```bash
.venv/bin/python -m myoacross.utils.image2msksim.run \
  --config data/experiments/image2msksim/config.yaml --stage embodiment
tools/gmr_fit/.venv/bin/python -m myoacross.utils.image2msksim.run \
  --config data/experiments/image2msksim/config.yaml --stage retarget
.venv/bin/python -m myoacross.utils.image2msksim.run \
  --config data/experiments/image2msksim/config.yaml --stage evaluate
```

Clear or relocate the failed stage's outputs before repeating that stage.
Keep generated bodies and references unchanged when comparing policies.
Change only `checkpoint_path` and `policy_scheme` in a separate evaluation configuration.
Use one input subject per single-subject invocation. Select a distinct output root for another subject.
Set `cache_root` to the shared experiment cache. Reuse detector weights, backend copies, and GPU compilation caches.
Keep `checkpoint_path` at the repository default in committed configurations. Set the actual checkpoint path in a local configuration before execution.

For the full MoVi image experiment, use `myoacross/configs/experiments/image2msksim/batch.yaml`:

```bash
.venv/bin/python -m myoacross.utils.image2msksim.run \
  --config myoacross/configs/experiments/image2msksim/batch.yaml
```

Read image paths, measured heights, and measured body masses from `data/experiments/image2msksim/inputs/subjects.json`.
Keep these experiment inputs in this experiment directory. Do not update the shared dataset metadata.
The heights and body masses come from [MoVi Table 1](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253157#pone.0253157.t001).
Earlier outputs without height calibration remain results of the earlier protocol.
Do not describe those outputs as height-calibrated results. Repeat body construction, retargeting, and evaluation for the new protocol.
The current input set contains 88 photographs extracted at frame 150. Each photograph must show the selected person clearly.
The batch command runs subjects in sequence. Each subject uses the standard parallel motion evaluator.
Store each subject's configuration, model, retargets, and evaluation under `subjects/<subject_id>/` in the experiment output root.
Store the batch configuration and `dataset_summary.json` under `evaluations/<policy_scheme>/test_iter<iteration>/`.
Record failed stages. Continue with other subjects after a subject failure. Return a nonzero exit code if any subject fails.
Do not overwrite existing batch or subject outputs. Use the saved single-subject configuration and `--stage` for manual recovery.
Report successful-subject metrics for the complete input set and separately for source training, source test, and unregistered subjects.
Report the requested and successful subject counts for each group. Do not omit failures from the reported counts.
Use the source test group for claims about held-out subject generalization.
A new reconstruction ID does not change the source person's training membership.
Do not treat a failed reconstruction or retarget as a completed policy trial.

### Results

No image reconstruction or policy result is recorded yet.
Report construction failures and retarget screening failures with the control results.

## Gait120 Physiological Evaluation

### Protocol

Fit each subject's SMPL-H shape with MoSh++ Stage I and labelled Gait120 markers.
Use the available level-walking and static recordings for shape fitting.
Do not fit Gait120 motion sequences with Stage II for this protocol.
Generate measurements and a segment-scale individual model through the existing core APIs.

Select the eight `walk_medium` clips from the registered `kit_flat_locomotion_72` motion set.
Keep motion IDs in `utils/datalists.py`. Do not copy their list into the configuration.
Retarget these KIT clips to each subject model with the original GMR-Fit procedure.
Do not initialize the root from the first pelvis target or increase the first-frame iteration limit.
Require the standard numerical screen before evaluation.
Route fitted-shape caches to the experiment output directory.

Evaluate the five configured policies at iteration 15000.
Keep checkpoint weights and normalization statistics unchanged.
Use eight environments per subject in the existing parallel evaluator.
Run policies sequentially. Each policy evaluates all selected subject-motion pairs together.
Do not fine-tune policies or optimize waveform phase shifts.

### Adapter Responsibilities

| Module in `utils/gait120/` | Responsibility |
| --- | --- |
| `run.py` | Load configuration and dispatch stages to their environments. |
| `convert.py` | Read labelled TRC files, collect subjects, and prepare marker inputs. |
| `fitting.py` | Fit and export individual shape with MoSh++ Stage I. |
| `modeling.py` | Call measurement and model-generation APIs. |
| `motions.py` | Select registered KIT clips and build the evaluation catalog. |
| `retarget.py` | Call original retargeting and redirect only its shape cache. |
| `evaluation.py`, `grf.py` | Capture signals through existing evaluator and substep hooks. |
| `references.py`, `mcos.py` | Read human references and assign force plates to feet. |
| `signals.py` | Define channels and normalize cycles and complete stances. |
| `analysis.py` | Aggregate valid motions and report correlations and completion. |

### Input Corrections

Join available Step files on their original timestamps. Remove one duplicated endpoint.
Accept a trial with only one available Step. Retain its source-frame mapping.
For the known static-header mismatch, remove the four header-only leg markers when the
43-marker header describes exactly 39 coordinate triples in the verified body-marker order.
Remove an incomplete final static row before fitting. Record these corrections in `input_audit.json`.
Do not modify source TRC files. Reject other inconsistent row layouts.

### Signal Collection and Analysis

Save activation, ten joint angles, actuator joint moments, per-foot three-dimensional GRF,
vertical GRF in body-weight units, contacts, reference frames, and simulation timestamps.
Capture state signals before each control step.
Capture force and solver moments at the last native physics interval before `forward()`.
Retain actual force-interval timestamps. Do not shift force samples to the state timestamp.
Actuator generalized moments are not human inverse-dynamics net joint moments.
Do not report a human torque correlation without a comparable human reference.

For activation and joint angles, extract complete right heel-strike cycles at 100 phase points.
Compare activation with all 12 processed EMG channels. Both soleus channels use the same model soleus.
Report bilateral hip, knee, and ankle flexion correlations. Report hip rotation separately.
Use the MM-EvalKit joint sign convention.

For GRF, compare complete observed stance intervals from contact onset to toe-off.
Use the established 0.05 BW threshold. Exclude clipped contacts, ambiguous plate assignments,
and contacts shorter than 0.2 seconds. Do not interpret unmeasured off-plate forces as swing zeros.
Report vertical GRF Pearson correlation and BW-normalized RMSE.
Do not label this measure as full-cycle GRF agreement.

Average cycles within each completed motion. Weight available valid motions equally within each subject.
Average human available Steps within each trial, then weight available trials equally.
Compute each subject's correlations against their own human references. Weight subjects equally.
Keep subjects with fewer than eight valid motions. Record their valid and completed motion counts.
Retain subjects with no valid waveforms as missing correlations, not zero scores.
Report all attempted motions in the completion-rate denominator.
Keep missing-reference coverage separate from control completion.

### Configuration and Commands

Use `myoacross/configs/experiments/gait120/pipeline.yaml` as the run configuration.
It contains repository-relative paths for data, resources, checkpoints, interpreters, and outputs.
Update those paths in this file if the local installation differs.
Use a new output root for the final protocol. Existing experimental outputs are not overwritten.
The `policies` mapping selects checkpoint paths and output scheme names.
The adapter checks each checkpoint's stored iteration against `checkpoint_iteration`.
Use the root environment for orchestration, references, evaluation, and analysis.
Reuse SKEL for measurement and GMR-Fit for retargeting. Only MoSh++ needs an additional environment.

```bash
python -m myoacross.utils.gait120.run --config myoacross/configs/experiments/gait120/pipeline.yaml
python -m myoacross.utils.gait120.run --config myoacross/configs/experiments/gait120/pipeline.yaml --stage prepare
```

Stages are `prepare`, `fit`, `measure`, `embodiment`, `retarget`, `references`, `evaluate`, and `analysis`.
`prepare` writes the source manifest, subject metadata, and resolved KIT motion IDs.
`fit` writes `shapes/<subject>/shape_identity.npz` and marker correction records.
Model and retarget stages write `embodiments/` and `retargets/segment_scale/`.
Evaluation writes `evaluations/<scheme>/test_iter15000/`.
Each evaluation directory contains the resolved configuration, completion metrics, and `profiles/`.
Analysis writes per-subject counts, per-channel metrics, mean profiles, and protocol metadata.

### Model-construction comparisons

Use `data/experiments/gait120/model_comparison/` for Template, Global scale and Regional scale comparisons.
The shared Gait120 directory supplies subject measurements, Regional scale models, reference motions and measured signals.
`height_scale` denotes the global scale fitted to all 17 skeletal measurements.

```bash
python -m myoacross.utils.gait120.compare_models prepare
# Run retargeting in the GMR-Fit environment.
tools/gmr_fit/.venv/bin/python -m myoacross.utils.gait120.compare_models retarget --condition template
tools/gmr_fit/.venv/bin/python -m myoacross.utils.gait120.compare_models retarget --condition height_scale
python -m myoacross.utils.gait120.compare_models evaluate --condition template
python -m myoacross.utils.gait120.compare_models evaluate --condition height_scale
python -m myoacross.utils.gait120.compare_models evaluate --condition regional_scale
```

Use `--root` and `--gait-root` to select alternative data directories.
Use `--checkpoint`, `--subjects`, `--motions` and `--output` to evaluate a selected checkpoint and data subset.
Select empty output directories for new runs.

### Statistical analysis

Run these commands with the analysis dependencies installed:

```bash
python -m myoacross.utils.gait120.compare_analysis --output data/experiments/gait120/model_comparison/analysis
python -m myoacross.utils.gait120.comparison_statistics --output data/experiments/gait120/model_comparison/analysis
python -m myoacross.utils.gait120.matched_motions --output data/experiments/gait120/model_comparison/analysis
python -m myoacross.utils.gait120.activation --output data/experiments/gait120/model_comparison/activation
```

For the TNSRE analysis, use 10,000 subject bootstrap resamples and 95% percentile intervals for all body-mass and muscle-level associations. Refit height/speed adjustment within each resample. `body_mass` and `muscle_statistics` use this count by default. To recompute from existing `body_mass/` and `per_muscle/` subject summaries without rerunning simulation, use `python -m myoacross.utils.gait120.recompute_confidence_intervals --input-root INPUT --output-dir OUTPUT --verify-previous`. The optional verification reproduces the earlier 5,000/2,000-resample intervals before calculating the current results. Keep temporary inputs outside data directories; preserve original experiment outputs.

`compare_analysis` exports subject-level metrics and paired model comparisons.
`matched_motions` repeats the comparisons using common completed motions within each subject.
`activation` exports muscle-level activation, body-mass correlations, adjusted slopes and bootstrap intervals.
Analysis uses saved simulation signals and measured reference summaries.


## Plot Existing Results

Install the analysis dependencies with `uv sync --extra analysis`.
Keep plotting code in `myoacross/utils/plotting/`.
Read existing outputs. Do not run training or evaluation to generate figures.
Use `--data-root` to select the data directory. The default is the repository's `data/` directory.
Set `--output-dir` to a separate figure directory.

```bash
python -m myoacross.utils.plotting.plot_policy_performance --output-dir /tmp/myoacross-figures/policies
python -m myoacross.utils.plotting.plot_gait120 --output-dir /tmp/myoacross-figures/gait120
```

| Module | Input results |
| --- | --- |
| `plot_policy_performance` | Policy comparison |
| `plot_policy_paired_changes` | Paired subject results |
| `plot_motion_categories` | Results by motion category |
| `plot_virtual_augmentation` | V0 and V10 comparisons |
| `plot_asymmetry_performance` | Physical asymmetry evaluation |
| `plot_input_sensitivity` | Measurement perturbation evaluation |
| `plot_image_evaluation` | Image-based model evaluation |
| `plot_gait120` | Gait120 control and waveform metrics |
| `plot_evaluation_details` | Detailed fitting, policy, and sensitivity results |

Use `draw_policy_comparison`, `draw_virtual_augmentation`, and `draw_input_perturbations`
to draw the experiment schematics. Each command requires `--output-dir`.
Use `draw_image_workflow --assets-dir PATH --output-dir PATH` for the image workflow.
The asset directory must contain an `assets/` subdirectory with
`demo_input_anonymized.png`, `demo_skel_overlay.png`, `demo_model.png`, and
`demo_policy_simulation.png`. These illustration assets are not included in the code release.
The commands export individual panels. They do not assemble manuscript pages.

Keep shared training banks in `data/myoacross/rl/embodiments_bundles/`.
Keep the separate held-out subject bank in
`data/experiments/policy_comparison/embodiments_bundles/segment_scale_test/`.
Keep Gait120 source data in `data/experiments/gait120/dataset/`, analysis outputs in
`data/experiments/gait120/analysis/`, and run records in
`data/experiments/gait120/provenance/`.

Physical-condition evaluation collects unreachable environment cycles between batches and logs device memory. Set resume: true only for an existing physical evaluation with identical saved configuration; completed conditions require a finished status and the full subject-motion trial set.
