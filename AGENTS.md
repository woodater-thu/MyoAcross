# MyoAcross Development Rules

## Purpose

MyoAcross is a research framework.
It generates subject-specific musculoskeletal models, retargets motions, and trains policies with morphology input.
The repository also supports paper experiments.

Use this file for continued development.
Use ASD-STE100 style in `AGENTS.md` and `docs/`.
Use ordinary technical English in `README.md`.
Use short sentences and active voice in controlled documentation.
Use one consistent term for each technical concept.
Use the imperative form for instructions.
Keep code identifiers and external names unchanged in technical descriptions.

## Implementation Principles

- State material assumptions before implementation.
- Ask when an unclear requirement changes behavior or experiment conditions.
- Use the simplest implementation that satisfies the request.
- Do not create abstractions for code used only once.
- Do not add unrequested features, configuration options, compatibility branches, or fallback mechanisms.
- Add defensive code only for a known failure risk.
- Support each check with an interface requirement or an observed failure.
- Validate data once at the applicable input or export boundary.
- Do not add unnecessary hash checks or repeat NaN and finite-value checks on unchanged data.
- Repeat validation only when a data change or a specific failure risk requires it.
- Make each change serve the current task.
- Keep unrelated code, comments, and formatting unchanged.
- Remove imports, variables, and functions that the current change makes unused.
- Report unrelated unused code. Remove it only when the task includes that cleanup.
- Define success criteria that can be checked.
- Select checks for the affected behavior and the impact of the change.
- Use focused checks for simple changes. Do not run every check mechanically.

## Module Responsibilities

| Location | Responsibility |
| --- | --- |
| `myoacross/embodiment/` | Generate, scale, measure, and export subject models. |
| `myoacross/retarget/` | Fit body shape, retarget motions, and check motion quality. |
| `myoacross/rl/` | Generate virtual embodiments, train policies, and evaluate policies. |
| `myoacross/utils/` | Prepare datasets and provide shared support functions. |
| `myoacross/utils/analysis/` | Compare results and calculate statistics. |
| `myoacross/utils/plotting/` | Generate figures. |
| `tools/skel/` | Define the separate SKEL environment. |
| `tools/gmr_fit/` | Define the separate GMR-Fit environment. |
| `tools/hsmr/` | Define the separate HSMR environment. |
| `myoacross/utils/image2msksim/` | Reconstruct image-derived bodies and dispatch existing core interfaces. |
| `tests/` | Check supported behavior. |

Keep `paths.py` and `datalists.py` in `myoacross/utils/`.
Keep SMPL/SKEL target measurement generation in `embodiment/smplh_to_measurements.py`.
Use `python -m myoacross measure` in the SKEL environment.
Keep SKEL imports local to T-pose geometry computation.
Put auxiliary implementations in the applicable `utils/` subdirectory.
Keep environment files in `tools/`.

Auxiliary tools can call core modules.
Core modules must not depend on plotting, analysis, or benchmark tools.
Keep command parsing separate from algorithms.
Keep compiled model hashing in `embodiment/model.py`.
Keep retarget numerical screening in `retarget/export.py`.
Keep retarget dataset inventories in `utils/analysis/retarget_manifest.py`.
Do not copy training or evaluation implementations for individual experiments.

## Names

- Use `snake_case` for Python files, functions, configuration fields, and directories.
- Use `PascalCase` for classes.
- Name task dispatch functions `run_generate`, `run_train`, and `run_evaluate`.
- Name execution functions after their operation, such as `generate_embodiment_bundles`, `train_policy`, and `evaluate_policy`.
- Use `UPPER_SNAKE_CASE` for constants.
- Name experiment directories after research questions.
- Use `generate_<body_scheme>_<split>.yaml` for experiment generation configurations.
- Use `train_<policy_scheme>.yaml` for experiment training configurations.
- Use `evaluate_<policy_scheme>_<split>.yaml` for experiment evaluation configurations.
- Use `train` or `test` to identify the subject split in generation and evaluation names.
- Keep the standard RL configuration names and embodiment method names unchanged.
- Put seeds, GPU counts, and iteration limits in configuration fields.
- Include these values in names only when they identify different schemes.
- Preserve external model names, dataset names, subject IDs, and motion IDs.
- Maintain subject and motion exclusions in `utils/datalists.py`.
- Keep registered motion-set names for selection and report identity.
- Omit repeated motion ID lists from experiment configurations.
- Save resolved motion IDs in training configurations and checkpoints.
- Update all references when you change a project name.

Configuration examples are `train_segment_concat.yaml`, `generate_asymmetry_enhanced_train.yaml`, and `evaluate_segment_film_test.yaml`.

## Commands and Training Support

Use `python -m myoacross` for core commands.
Use `python -m myoacross rl train --config ...` for training.
Use `torchrun` with the same command for training on multiple GPUs.
Use `python -m myoacross rl evaluate --config ...` for ordinary and measurement-sensitivity evaluation.
Use parallel evaluation for both procedures.
Run each subject-motion baseline once.
Use original subject retargets for fixed physical asymmetry evaluation.
Do not retarget motions again for these body conditions.
Generate fixed physical test XML files and bundles before evaluation.
Store shared physical test models and bundles under the applicable experiment output directory.
Read these bundles directly during evaluation.
Do not export XML files for training banks.
Do not export evaluation trajectories or `rollout.npz`.

| RL location | Responsibility |
| --- | --- |
| `algorithms/` | Implement PPO and its updates. |
| `networks/` | Implement the morphology encoder, policy networks, and value networks. |
| `embodiments/virtual/` | Define generation parameters, sample perturbations, and generate virtual bodies. |
| `embodiments/` | Manage subject records, morphology features, bundles, training data, and model configuration. |
| `tasks/` | Define environments, observations, actions, rewards, and termination conditions. |
| `viewer.py` | Initialize single-environment policy playback and manage viewers. |
| `runners/` | Control generation, training, and evaluation procedures. |
| `run.py` | Parse commands and dispatch tasks. |
| `runners/train_rl.py` | Prepare training, manage its runtime, and execute PPO. |
| `runners/generate_embodiment_bundles.py` | Prepare generation settings and generate subject, virtual, and fixed test body bundles. |
| `runners/evaluate.py` | Select conditions, initialize body groups, execute parallel evaluation, and summarize metrics. |
| `runners/multi_embodiment/batched.py` | Assign body groups, install parameters, and assign reference motions. |
| `runners/multi_embodiment/backend.py` | Prepare the adapted backend and check compatibility. |
| `runners/distributed.py` | Manage processes, communication, gradients, and shared statistics. |

`training.embodiments_per_rank > 1` selects multiple body groups on each GPU.
The environment count must be divisible by the body group count.
Each body must have a compatible topology and parameter layout.

The backend preparation procedure copies and adapts the dependency package.
It does not start training or change the installed package.
Prepare the adapted backend automatically in the command entry for multi-body training and evaluation.
Load the adapted backend before the process imports the original backend.
Preserve checkpoint recovery arguments and torchrun environment variables in child processes.
Serialize shared backend cache preparation across ranks.
Use the same backend support for training and batch evaluation.

Each GPU has its own simulation and rollout storage.
Synchronize initial policy parameters.
Average gradients before gradient clipping and optimizer updates.

## Supported Policy Modes

| Mode | Purpose |
| --- | --- |
| `none` | Use an MLP without morphology input. |
| `concat` | Encode morphology as an observation group and concatenate the group outputs. |
| `rl` | Learn a morphology representation and apply FiLM. |

Maintain these three modes.
Use `ObservationEncoder` for the Concat morphology group.
Give the actor and critic separate group encoders.
Use `[64, 128, 256]` for the Concat morphology group layers.
Normalize morphology with fixed training-set statistics before group encoding.
Keep the encoder and shared functions that the `rl` mode requires.
Keep only encoder layers in `networks/morphology_encoder.py`.
Do not create decoder modules or reconstruction losses.
Convert encoder parameter names when loading supported legacy `rl` checkpoints.
Omit inactive decoder weights.
Preserve optimizer state when resuming supported checkpoints.
Unsupported historical checkpoints do not require loading compatibility.
Do not delete stored checkpoints because their policy mode is unsupported.

### MyoFullBody Transfer Baseline

Maintain the MyoFullBody transfer baseline within the `none` policy mode.
Use this baseline as a training scheme within `none`.

Train the policy on the original MyoFullBody model with these settings:

```yaml
encoder:
  mode: none
data:
  embodiment: baseline
  morphology_source: subject
training:
  embodiments_per_rank: 1
```

One GPU can train this scheme.
Multiple GPUs can also train it with one body group per GPU.
Do not apply the standard eight-body setting to this scheme.
Do not generate virtual bodies for this scheme.

Evaluate the checkpoint on the normal held-out subject split with these settings:

```yaml
split: test
target_embodiment: segment_scale
subject_ids: null
```

Use each test subject's model and matching retargeted motions.
Keep the policy weights and checkpoint normalization statistics unchanged.
Do not perform subject-specific fine-tuning.
Use the checkpoint's motion selection for the comparison.
Record both the training embodiment and the evaluation embodiment.

Store this scheme's configurations in `configs/experiments/policy_comparison/` under `myoacross/`.
Use `train_myofullbody_mlp.yaml` and `evaluate_myofullbody_mlp_test.yaml` as configuration names.
Store its outputs under `data/experiments/policy_comparison/` with the scheme name `myofullbody_mlp`.
Keep this scheme separate from the standard segment-scale FiLM configuration.

## Configurations

Use these directories:

```text
myoacross/configs/
  embodiment/
    segment_scale.yaml
  retarget/
  rl/
    generate.yaml
    train.yaml
    evaluate.yaml
  experiments/
    skeletal_fitting/
      baseline.yaml
      height_scale.yaml
    <experiment_name>/
```

Keep one standard generation, training, and evaluation configuration in `rl/`.
Put experiment-specific configurations in `experiments/<experiment_name>/`.
Use the standard configuration when an experiment uses the same settings.
Do not maintain an identical copy.
Create configuration files only when the procedure requires them.

Use `.yaml` for configuration file names.
Load YAML through `utils/config.py` with Hydra Compose and OmegaConf.
Return resolved plain dictionaries to core implementations.
Inherit the applicable `/rl/generate`, `/rl/train`, or `/rl/evaluate` configuration in RL experiment files.
Use `# @package _global_` and target `@_global_`.
Put `_self_` last in `defaults`.
Write only explicit experiment overrides.
Use `null` to disable inherited optional settings.
Preserve meaningful null values.
Lists replace inherited lists.
Keep the `--config PATH` interface.
Resolve data paths relative to the repository root.
Do not change the working directory during configuration loading.
Save the resolved training configuration, including the actual motion selection.

Keep standard settings in `configs/rl/` under `myoacross/`.
Use [Common Protocol](docs/experiments.md#common-protocol) for the paper training budget and launch settings.
Record protocol changes in `docs/experiments.md`.

## Data and Outputs

Use these locations:

```text
data/
  dataset/
  models/
  myoacross/                   # Shared data and default outputs
    embodiments/
      measurements/
    retargets/
      baseline/<motion_id>/
      segment_scale/<subject_id>/<motion_id>/
      height_scale/<subject_id>/<motion_id>/
      fitted_shapes/
    rl/
      embodiments_bundles/
        subject/<subject_id>/
        virtual/<subject_id>/
      checkpoint/
      logs/
      evaluations/
  experiments/
    skeletal_fitting/
      evaluations/report/
    <experiment_name>/
      embodiments_bundles/
      checkpoint/<scheme>/
      logs/<scheme>/
      evaluations/<scheme>/
```

- Use the same experiment name in configuration and output directories.
- Store HSMR resources, reconstruction outputs, and download caches under `data/experiments/image2msksim/`. Link the shared SKEL parameters from this directory.
- Create only the output directories that the experiment requires.
- Use one fixed path for each scheme.
- Do not add timestamps or run numbers to output paths.
- Use the same scheme name for checkpoints, logs, and evaluations.
- Record the evaluation split and checkpoint iteration.
- Use named subdirectories when a scheme has multiple evaluation conditions.
- Specify paths through configuration fields or command arguments.
- Keep shared dataset, model, resource, and intermediate data paths in `utils/paths.py`.
- Specify RL bundle, checkpoint, log, and evaluation destinations in YAML.
- Require `data.bundle_root` for generation and training.
- Do not add a Python fallback path.
- Do not add retarget method or motion-set layers to the default retarget path.
- Select a separate output root for another motion collection.
- Keep motion-set identity in reports.
- Resolve configuration data and output paths relative to the repository root.
- Use `final.pt` for the final checkpoint. Check its stored iteration before evaluation.
- Do not put server-specific absolute paths in implementations.
- Save the actual configuration and source information with the outputs.
- Do not overwrite existing outputs automatically.
- Select checkpoint recovery explicitly for continued training.
- Let the user clear the applicable outputs before a new run.

### Shared Bundles

A bundle contains parameters from a compiled model, morphology features, and metadata.
Training uses these parameters in an existing simulation.
The bundle does not contain GPU kernels.

- Store standard bundles in `data/myoacross/rl/embodiments_bundles/`.
- Store experiment-specific bundles in the applicable experiment directory.
- Store the default subject and virtual banks directly in `embodiments_bundles/subject/` and `embodiments_bundles/virtual/`.
- Name experiment-specific bundle directories after their generation scheme.
- Keep generation settings and seeds in configuration fields and bank metadata.
- Store one copy when multiple experiments use the same bundle.
- Reference shared bundles through training configurations.
- Do not change shared bundles during training.
- For `data.morphology_source: virtual`, train on the original subject bundle and all virtual bundles from the same `data.bundle_root`.
- Sample subjects uniformly. Sample the original and virtual variants uniformly within each subject.
- Keep original bundles in `subject/` and generated bundles in `virtual/`. Do not copy original bundles into the virtual bank.
- Count generated bodies only in `count_per_subject`. Each training subject has `count_per_subject + 1` candidate bodies.
- Fit morphology normalization statistics from the complete training body set.
- Check source subjects, sampling parameters, seeds, simulation settings, and compatibility before reuse.
- Record generation settings and source identity in `bank.json`, `index.json`, and bundle metadata.
- Clear or relocate old outputs explicitly before replacing the default generation scheme.
- Use separate experiment scheme directories when generation parameters or source data differ.

Record complete sampling settings, including rejection limits.
Treat generation factors as local augmentation priors unless data support population calibration.
Record the settings changed in each experimental comparison.
Use stored bank metadata to identify earlier generation protocols.
Configuration changes do not update stored banks or checkpoints.

## Documentation

| File | Content |
| --- | --- |
| `README.md` | Project introduction and basic operation. |
| `AGENTS.md` | Continued development rules. |
| `docs/install.md` | Environments, dependencies, and data preparation. |
| `docs/dataset.md` | Source data layout, subject metadata, fixed splits, and motion selection. |
| `docs/development.md` | Development commands, tests, and fault diagnosis. |
| `docs/architecture.md` | Module responsibilities, dependencies, and data flow. |
| `docs/embodiment.md` | Body model implementation, interfaces, and extension methods. |
| `docs/retarget.md` | Retargeting implementation, interfaces, and extension methods. |
| `docs/rl.md` | Training and evaluation implementation, interfaces, and extension methods. |
| `docs/experiments.md` | Paper claims, methods, configurations, commands, results, and analysis. |

Keep experiment instructions in `docs/experiments.md`.
Experiment configuration directories do not require separate README files.
Record actual training budgets and evaluation checkpoints.
Do not describe planned experiments as completed results.

## Tests and Agent Work

- Read this file before changes.
- Check the branch, worktree changes, and applicable implementation.
- Preserve user changes.
- Check actual paths before you use them.
- Check callers before you move or remove code.
- Do not infer a function's purpose from its file name alone.
- Separate structural changes from algorithm changes.
- Keep tests under `tests/embodiment/`, `tests/retarget/`, `tests/rl/`, and `tests/utils/`.
- Use `test_<functionality>.py` for test files.
- Check imports, commands, configurations, paths, and documentation references after changes.
- Keep tests for body parameter installation, reference ownership, touch sensors, and distributed gradient synchronization.
- Keep tests for MyoFullBody checkpoint evaluation on segment-scale test subjects.
- Use small data selections for generation, training, and evaluation checks.
- Put temporary configurations and outputs in a separate temporary directory.
- Do not overwrite experiment outputs during tests.
- Run checks that apply to the change.
- Do not repeat successful checks without a reason.
- Report the checks performed, their results, and checks that you could not run.
- Do not claim GPU validation without test evidence.
- Do not start long training runs without user instructions.
- Update affected documentation with code changes.
