# Architecture

## Data Flow

```text
Dataset identity and body mass
  -> SKEL measurements
  -> subject-specific MyoFullBody model
  -> target-fitted SMPL-H and retargeted motions
  -> original or virtual parameter bundles
  -> motion tracking environment
  -> PPO policy
  -> subject evaluation
```

## Modules

`embodiment/` scales body models, solves measurement fits, and exports models.
`retarget/` fits body shape, solves inverse kinematics, and checks exported motions.
`rl/` generates virtual bodies, defines simulation tasks, and trains and evaluates policies.
`utils/` contains shared data catalogs, path definitions, analysis tools, and plotting tools.
`tools/` contains the SKEL and GMR-Fit environment definitions.

The CLI parses commands and calls the applicable module.
`utils/config.py` composes YAML with Hydra and resolves OmegaConf values to plain dictionaries.
Embodiment and retarget loaders apply their task validation after composition.
RL runners receive the resolved dictionary.
Core modules do not import plotting or benchmark implementations.

## Subjects and Motions

`utils/datalists.py` registers target subjects independently from source motions.
A source performer identifies motion provenance.
A target subject identifies the simulated body.
Do not use one identity as the other.

Each target uses its selected shared motions.
Missing or invalid products cause errors.
The training catalog must not silently discard them.
Quality screening checks numerical motion validity.
It does not establish that a trained policy can complete the motion.

## Bundles

`rl/embodiments/bundle.py` stores compiled model parameters and morphology features.
It validates model signatures and array layouts.
`WarpBundleSwitcher` injects prepared parameters into allocated model arrays.
Bundle generation occurs before training.
Initialization compiles the simulation and loads its GPU kernels.
A compatible bundle change updates allocated arrays.
It does not compile another model XML or new GPU kernels.

## Multiple Bodies and GPUs

`rl/runners/multi_embodiment/batched.py` divides one batch simulation into equal body groups.
Each group uses one selected bundle and its source subject's reference motions.
Groups on one GPU share a policy and rollout storage.
The current implementation shares render meshes that do not participate in collisions.

`rl/runners/multi_embodiment/backend.py` adapts a dependency copy for per-environment touch site sizes.
Load the copy before importing the simulator.
This adapter does not change the installed dependency.

`rl/runners/distributed.py` manages one training rank per GPU.
Processes own separate simulations and rollout storage.
They synchronize policy parameters, gradients, and statistics.
They do not exchange simulation state.

Keep learned morphology encoding in `rl/networks/morphology_encoder.py`.
Keep physical features and bundle management in `rl/embodiments/`.
Keep virtual parameter definitions, sampling, and generation in `rl/embodiments/virtual/`.
The virtual generator calls the shared embodiment build and validation procedures.

## Parallel Evaluation

The evaluator accepts an explicit external motion catalog when `allow_external_motions=True`.
Keep the checkpoint motion selection unchanged.
Record both the training and evaluation motion IDs.
An optional signal collector receives initialization, reset, pre-step, post-step, and episode-end events.
The default path has no collector.
Keep dataset-specific collection and analysis outside the core evaluator.

Gait120 uses `utils/gait120/` for marker preparation, MoSh++ Stage I shape fitting, KIT references, and physiological analysis.
Keep orchestration in `run.py`, shape fitting in `fitting.py`, and core model calls in `modeling.py`.
Keep KIT selection and catalog construction in `motions.py`. Use original retargeting through `retarget.py`.
Keep signal processing in `signals.py` and cohort summaries in `analysis.py`.
It calls the existing measurement, embodiment, retarget, and parallel evaluation interfaces.
Only MoSh++ requires an additional environment. Reference preparation and analysis use the root environment.
Keep raw force-plate side assignment and reference export in `utils/gait120/references.py`.
The GRF collector uses a native mjlab substep metric to retain the last solver-interval force before `forward()` recomputes contacts.
It does not change XML sensors or installed backend code.

Dispatch ordinary and input-sensitivity evaluation through `runners/evaluate.py`.
Keep body initialization and subject-motion scheduling in the same evaluation module.
Reuse `runners/multi_embodiment/` for body installation and backend adaptation.
Run each zero-bias baseline once. Apply nonzero biases only to policy measurements.
Keep reference motions and physical bodies unchanged during each trial.
Calculate metrics on the device and export trial and summary tables.
Keep viewer initialization in `rl/viewer.py`.
