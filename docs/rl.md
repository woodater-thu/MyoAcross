# Reinforcement Learning Development

## Inputs and Sampling

RL uses body bundles and target-specific retargeted motions.
The paper's subject split has 110 training subjects and 31 test subjects.
Motion selection is independent of the subject split.
The registered library contains 72 clips.
Use `MOTION_EXCLUSIONS` in `utils/datalists.py` to exclude running and select 64 clips.
Apply the same exclusions to retargeting, training, evaluation, and dataset inventories.
Keep `motion_set: kit_flat_locomotion_72` for motion selection and report identity.
Read subject motions from `retargets/<embodiment>/<subject_id>/<motion_id>/`.
Read baseline motions directly from `retargets/baseline/<motion_id>/`.
Omit repeated `motion_ids` lists from standard and paper configurations.
Use explicit `motion_ids` only to select a smaller active subset.
Save resolved motion IDs in training outputs and checkpoints.

The sampler selects a subject uniformly and then selects a body variant uniformly.
For `data.morphology_source: virtual`, load both `subject/` and `virtual/` from `data.bundle_root`.
Use the original subject bundle and all generated variants as training candidates.
Keep each bundle in its generation bank. Training merges their indices without copying files.
Fit morphology normalization statistics from both banks for a new policy.
For `data.morphology_source: subject`, use the original bundle only.
The standard bank has 10 virtual bodies and one original body per training subject.
Virtual variants use their source target's reference motion library.
They do not have separately retargeted motions.
Original bodies use that same subject's reference motion library.
Each environment samples clips independently within its assigned reference range.

## Policy Modes

| Mode | Morphology path |
| --- | --- |
| `none` | No morphology input. |
| `concat` | An observation group encoder processes standardized morphology features. |
| `rl` | A learned encoder supplies FiLM parameters. |

Fit morphology normalization statistics on training data only.
The feature vector contains 17 measurements and body mass.
Use the checkpoint's statistics during inference.

For `concat`, use `ObservationEncoder` for the morphology group.
Use linear layers with SiLU activations and widths `[64, 128, 256]`.
Give the actor and critic separate morphology group encoders.
Concatenate the encoded observation groups before the backbone input projection.
Keep morphology outside the running observation normalizers.
The previous Concat architecture used the 18 features directly.
Its checkpoints cannot load into this architecture. Train a new Concat policy.

Keep the learned encoder in `networks/morphology_encoder.py`.
Use `MorphologyEncoderConfig` and the `morphology_encoder` configuration field.
The encoder contains linear layers, SiLU activations, and a latent LayerNorm.
It has no decoder or reconstruction-weight fields.
Load supported legacy `rl` checkpoints by converting `autoencoder.encoder.*` names and omitting `autoencoder.decoder.*` weights.
Read legacy architecture settings from the saved `autoencoder` field.
New checkpoints save only the encoder settings and parameters.
Preserve the optimizer state for supported legacy checkpoints.
Independent pretraining and reconstruction training are not supported.

## Training

Keep `run_generate()`, `run_train()`, and `run_evaluate()` in `rl/run.py` as task dispatch functions.
Use `run_training()` in `runners/train_rl.py` to manage the training process.
This function checks the configuration and prepares recovery, outputs, and initialization.
Use `train_policy()` to execute PPO with an initialized `TrainingState`.
Keep training index discovery and `_create_wandb_run()` in the training runner.
Use the shared `resolve_path()` in `utils/paths.py` for command paths.
Close the environment before leaving the training process context.

The runner uses RSL-RL rollout storage and return calculation.
`MorphologyPPO` owns one optimizer for all trainable policy parameters.
The policy outputs muscle excitation through a bounded action distribution.
The task contains root, joint, muscle, reference, contact, and lookahead observations.
Keep observation layouts compatible with the checkpoint.

`training.embodiments_per_rank` specifies body groups within each GPU.
All groups contribute to one rollout buffer and PPO update.
The current standard collects 32 steps from 2048 environments per GPU.
Each GPU contributes 65536 transitions per iteration.
Four GPUs contribute 262144 transitions per iteration.

The runner changes body selections between groups of PPO updates.
For compatible bundles, installation updates existing model arrays.
Initialization prepares GPU kernels and captures CUDA graphs.
Later bundle switches reuse them.
`updates_per_embodiment: 10` selects a new body batch after ten iterations.
Body installation also updates morphology features, joint limits, and reference ownership.
It resets the environments before the next rollout.

The training runner preloads the selected reference clips into CPU memory before PPO starts.
Each process has its own cache.
The cache stores decoded arrays. It does not store GPU motion libraries.
Body switches reuse these arrays and build the active GPU motion library.
Close the environment and release the cache when training ends.
Do not change reference files during training.

Adaptive sampling stores failure counts in one continuous array.
Each clip uses a view of its own bins.
Update failure counts in batches. Sample only clips used by the reset environments.
Keep clip boundaries and the sampling distribution unchanged.

CPU cache memory depends on the selected clips.
A local test with 110 subjects and 64 motions per subject used approximately 12.6 GiB per process.
Preloading took approximately 47 seconds on that computer.
Four processes keep four cache copies. Allow approximately 50 GiB for those arrays.
These values do not include other training memory.

## Backend and Distributed Training

The command entry prepares a separate backend copy when `training.embodiments_per_rank > 1`.
See the [training commands](../README.md#generate-train-and-evaluate).
Training and evaluation use the same preparation function before importing task implementations.
Single-body training and generation do not require adaptation.
The adapted backend skips tendon damping, armature, and passive-force kernels only when all associated coefficients are exactly zero in every world. It checks these coefficients again when rebuilding CUDA graphs. Nonzero coefficients retain the original computation.
Select the GPU count with `torchrun --nproc-per-node`.
Set `CUDA_VISIBLE_DEVICES` to select the devices available to those processes.
The training YAML specifies environments and body groups per GPU. It does not specify the GPU count.
Use the same GPU count when resuming a checkpoint. Resume requires the saved `world_size`.
Preserve `--resume` and torchrun environment variables when starting the adapted child process.
Serialize shared cache preparation with a file lock.
The adapted child continues task dispatch without restarting again.
Record checkpoint, model, and reference motion paths in evaluation outputs.

`distributed.py` selects each process's GPU and initializes NCCL.
It broadcasts initial parameters and averages gradients before clipping.
Each process retains its own simulation and rollout storage.
Rank zero writes shared logs and checkpoints.

## Fixed Outputs and Recovery

Training configurations specify `output.checkpoint_dir` and `output.logs_dir`.
Public configuration paths are relative to the repository root.
Load configurations through `utils/config.py`. Hydra composes defaults and explicit overrides.
OmegaConf resolves interpolations before runners receive plain dictionaries.
Keep command syntax and the current working directory unchanged.
Save the resolved training dictionary in `logs/config.yaml` and the checkpoint.
Resume uses those resolved settings.
Generation and evaluation configurations also specify their output paths.
Require `data.bundle_root` for generation and training. Resolve it against the repository root.
Keep public dataset, model, resource, and intermediate data locations in `utils/paths.py`.
Do not duplicate RL output defaults in Python.
The runner rejects occupied checkpoint or log directories for a new run.

The final iteration is stored in `final.pt`.
Earlier save intervals use `checkpoint_<iteration>.pt`.
The standard configuration ends at iteration 15000.
Check the stored iteration before using an existing `final.pt`.

Use these default output locations:

```text
data/myoacross/rl/
  embodiments_bundles/
    subject/<subject_id>/
    virtual/<subject_id>/
  checkpoint/
    morphology_normalizer.json
    checkpoint_<iteration>.pt
    final.pt
  logs/
    config.yaml
    metrics.jsonl
    wandb/
  evaluations/test_iter15000/
    config.json
    progress.json
    timing.json
    subjects.json
    body_groups.json
    trials.csv
    subjects.csv
    categories.csv
    summary.json
    subject_bundles/
```

Save morphology statistics for `concat` and `rl` policies. Omit that file for `none`.
The default W&B mode is offline. Save its local files under `logs/wandb/`.
Evaluation uses a specified `bundle_root` or prepares subject bundles inside its output directory.
Fixed physical asymmetry evaluation requires the generated condition bank.
Save `sensitivity.csv` when perturbation settings are supplied. Include the shared clean baseline.
Do not export rollout trajectories.
Use the same file roles under experiment checkpoint, log, and evaluation paths.

```bash
.venv/bin/python -m myoacross rl train --config CONFIG --resume CHECKPOINT
```

Recovery restores policy parameters, normalization, optimizer state, and absolute counters.
It starts new physics episodes and random generators.
Use empty output directories for the continued segment.
Preserve the source checkpoint before the user clears fixed output directories.

## Evaluation and Transfer

Use `python -m myoacross rl evaluate --config CONFIG` for ordinary and input-sensitivity evaluation.
Keep evaluation execution in `runners/evaluate.py`.
Use `evaluate_policy()` as its public entry point.
Keep single-environment viewer initialization in `rl/viewer.py`.

Use `perturbation: null` to evaluate one baseline per subject and motion.
Set `perturbation.target` to `morphology`, `reference`, or `combined` for input sensitivity.
Use three fixed sign samples per subject, motion, and nonzero level.
Use morphology magnitudes of 0.02, 0.05, and 0.10.
Use reference magnitudes of 2.5, 5, and 10 degrees.
Pair the levels for combined perturbations. Do not form a full cross product.
Run one clean baseline per subject and motion.
Accept `none`, `concat`, and `rl`. An MLP ignores morphology input.

Use `_prepare_morphology_perturbation()` for the 17 lengths and total body mass.
Sample an independent sign for each feature. Keep each sign fixed within the episode.
Apply the bias to physical measurements before checkpoint normalization.
Use `_prepare_reference_perturbation()` for fixed lower-limb joint-angle offsets.
Use the same offset for current and future target frames.
Compute biased body positions with the subject model and MuJoCo kinematics.
Prepare reference position arrays once per batch. Keep them in memory only.
Reference preparation adds setup time and device memory for each angle and sample.
Use a smaller subject selection if the full reference bank does not fit on the device.
Keep root motion, anchor targets, joint velocities, and phase unchanged.
Replace policy joint targets, position errors, and future body positions only.
Keep the clean command for initial states, rewards, termination, and scoring.
Do not change shared training commands, observations, motion libraries, or networks.

Derive signs from seed, subject ID, motion ID, and sample index.
Reuse signs across policies and magnitudes. Do not use world IDs or queue order.
Record magnitudes, samples, feature factors, and joint offsets in trial rows.
Keep ordinary and physical evaluation output fields unchanged.
Keep completed evaluation outputs. Do not rerun them for this structural change.
The former single-feature protocol and its configuration fields are not supported.
Keep its stored results separate from the new multivariate protocol.

Prepare fixed physical asymmetry conditions with `rl generate` before evaluation.
Set generation `data.split: test` and `body_perturbation`.
Specify `model_root`, `relative_shortening`, and `sides` (`left` or `right`).
Export the original body once. Export each nonzero shortening level on each selected side.
Scale that side's thigh and shank regions together. Preserve other scales and total mass.
Reuse `VirtualEmbodiment` and the embodiment exporter to save fixed test XML files.
Export bundles from the in-memory models. Do not reload the saved XML for bundle export.
Keep XML output limited to fixed test conditions. Training banks do not export XML.
Use generation `workers` to prepare models and bundles in independent subject processes.
Save `conditions.json` last to record the complete condition set.
Reject existing generation destinations. Clear or relocate old outputs explicitly.

Set evaluation `bundle_root` to the shared test-bank root.
Read condition settings and measurements from the generated metadata.
Load bundles directly. Do not generate models or export bundles during this evaluation.
Check subject availability, simulation settings, and model layout at the loading boundary.
Keep each original subject's retargeted motion records unchanged.
Keep checkpoint normalization statistics unchanged.
Do not combine physical changes with input-sensitivity settings in one evaluation.
Use the existing parallel evaluator for each body condition.
Save combined `trials.csv` and `body_conditions.csv` at the output root.
Record `body_delta_<metric>` against the matching original subject-motion trial.
Save ordinary result files and runtime metadata in each condition directory.
Keep root subject, category, and summary files limited to the original condition.
See [Asymmetry Generalization](experiments.md#asymmetry-generalization) for configurations and measured geometry fields.

Set `envs_per_subject` to the number of parallel worlds in each body group.
Select motions from the checkpoint's training selection; explicit motion IDs must be a subset.
Reset only worlds whose tasks end.
Disable automatic resets, observation corruption, and initialization noise.
Score post-step states against saved pre-step references, including the final executed transition.

Save `trials.csv`, baseline `subjects.csv`, `categories.csv`, and subject-weighted `summary.json`.
Save `sensitivity.csv` only when perturbation settings are present.
Store one baseline trial per subject and motion; derive zero-change sensitivity rows without rerunning it.
Save configuration, progress, timing, subject metadata, and body groups in the evaluation output.
Save temporary subject bundles there only when `bundle_root` is omitted.
Do not save trajectories, `rollout.npz`, muscle time series, joint-force time series, or GRF samples.

`target_embodiment` selects test models independently from the training embodiment.
For MyoFullBody transfer, train with `baseline` and evaluate with `segment_scale` on `split: test`.
Do not fine-tune the policy for each test subject.
The evaluator uses each target's matching references and checkpoint motion selection.

Read [Evaluation Result Fields](#evaluation-result-fields) for metric definitions and aggregation.
Use [Result Interpretation](experiments.md#result-interpretation) for paper comparisons.

For supported historical checkpoints, resume ignores a removed zero reconstruction loss setting.
Resume compares data paths after resolving them relative to the repository root.
Relative and absolute paths to the same location are equivalent.
Changed bundle or retarget paths are invalid for resume.

## Body Data and Virtual Generation

Use `run_generation(config)` to check generation settings, select source subjects, prepare simulation settings, and export the requested banks.
Keep this procedure in `runners/generate_embodiment_bundles.py`.
Keep `run_generate()` as a task dispatch function.
Use `generate_embodiment_bundles()` in `runners/generate_embodiment_bundles.py` for both subject and virtual bundles.
Set `workers` in the generation YAML to select the process count. The standard configuration uses 16 workers.
Use `workers: 1` for serial generation.
Generate the subject bank first, then the virtual bank. Process subjects in parallel within each bank.
Each worker loads its models and writes one subject directory. Keep variants within a subject sequential.
Derive each subject seed from the global seed and subject ID. Preserve sampling when the worker count changes.
Check model signatures within each subject and across subjects. Write `bank.json` only after all subjects succeed.
The direct export function defaults to one worker. Evaluation keeps this setting.

Keep morphology features and normalization in `embodiments/features.py`.
Keep bundle storage and parameter installation in `embodiments/bundle.py`.
Keep generation parameters, perturbation sampling, and model construction in `embodiments/virtual/`.
Use `embodiments/virtual/generator.py` to generate models around accepted subject models.

Define covariance factors in `sampling.covariance` in the generation configuration.
An asymmetry factor adds opposite relative changes to the right and left parameters of a paired region.
The mass perturbation includes the global size factor multiplied by `mass_size_coefficient`, plus an independent residual.
These factors define local perturbations around each subject model.
See [Reference generation prior](experiments.md#reference-generation-prior) for assumptions and source evidence.
Read stored bank metadata to identify the settings used to generate existing bodies.
Configuration changes do not modify stored banks.

## Evaluation Result Fields

Use `config.json` to identify the checkpoint, body schemes, subject split, and motion selection.
It records the seed, device, step cap, and environment count per subject.
It also records input perturbations and resolved evaluation conditions.
The file contains the checkpoint training configuration and scoring conventions.
It does not separately record the checkpoint's actual stored iteration.
Read that value from the checkpoint, not the output directory name.

Use `trials.csv` for one row per subject, motion, and evaluated condition.
Record the target subject, source performer, and motion category.
Record model and reference paths.
For input biases, record the parameter, relative change, unit, true value, and supplied value.
Save the complete policy input measurement vector.
Record steps, durations, coverage, and completion.
Record the end reason, termination terms, reward sum, and tracking errors.
Define coverage as executed steps divided by reference horizon steps.
Set completion only when the motion finishes without failure.
Distinguish failure, motion completion, time limit, other truncation, and an explicit step cap.
Average tracking errors over the executed steps. An early failure does not include unexecuted reference frames.
Record `delta_<metric>` as the condition value minus the matching zero-perturbation trial.

| Metric | Definition |
| --- | --- |
| `body_position_error_m` | Mean world-position distance over tracked bodies, in meters. |
| `aligned_body_position_error_m` | Body-position error after matching reference horizontal anchor position and yaw to the robot; preserve height error. |
| `body_rotation_error_deg` | Mean quaternion angular error over tracked bodies, in degrees. |
| `joint_angle_error_deg` | Mean absolute position error over hinge joints, in degrees. |
| `joint_velocity_error_deg_s` | Mean absolute velocity error over hinge joints, in degrees per second. |

Use `subjects.csv` to average baseline motion metrics separately for each subject.
Use `categories.csv` to average baseline trials within each motion category.
Use `summary.json` for baseline subject count, trial count, and equally weighted subject means.
Interpret averaged `completed` as the completion fraction, not a percentage.
Exclude perturbed trials from these three summaries.

Use `sensitivity.csv` to group results by target and the two perturbation magnitudes.
Average sign samples and motions within each subject, then weight subjects equally.
Record the same metrics, paired differences, subject count, trial count, and `completion_drop`.
Define `completion_drop` as baseline completion minus perturbed completion.
Include zero-change summary rows without repeating zero-change simulation.
Perturb policy morphology and reference inputs only. Keep physical models and stored reference motions unchanged.

Use `subjects.json` to inspect model and reference paths for each subject.
It also records true measurements, feature names, and units.
Use `body_groups.json` to inspect subject identity, bundle variant, bundle path, and environment count for each group.
Keep the generated subject parameter bundles and their bank and index metadata in `subject_bundles/`.

Use `progress.json` for initialization, running, finished, or failed status.
Record completed and total episodes, active environments, and running rollout time where applicable.
Record final trial count on completion and the error on failure.
This file reports batch execution status, not a transactional completion marker for every exported summary file.

Use `timing.json` for setup, rollout, and batch execution times.
It records useful transitions, useful transitions per second, and batch steps.
It also records environment count and simulated transitions.
Simulated transitions equal batch steps multiplied by environment count. Useful transitions count active trials only.
The batch timer excludes final CSV export and summary writing.
Do not export rollout trajectories or video files from evaluation.

## Explicit Evaluation Catalogs

`evaluate_policy` accepts an optional `catalog: DatasetRecord`.
Use this input for subjects that are outside the fixed dataset split.
Construct the body and retarget records before evaluation.
The catalog must match the evaluation split and checkpoint motion selection.
When no catalog is supplied, evaluation uses the existing fixed subject selection.
Image2MSKSim uses this interface. It does not change training or standard evaluation.
