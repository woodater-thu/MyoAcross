# Development

## Test Environments

Run the tests in the environment that supplies their dependencies.

```bash
MPLCONFIGDIR=/tmp/myoacross-mpl .venv/bin/python -m unittest discover -s tests/embodiment
MPLCONFIGDIR=/tmp/myoacross-mpl .venv/bin/python -m unittest discover -s tests/utils
MPLCONFIGDIR=/tmp/myoacross-mpl .venv/bin/python -m unittest discover -s tests/rl
MPLCONFIGDIR=/tmp/myoacross-mpl tools/gmr_fit/.venv/bin/python -m unittest discover -s tests/retarget
```

The root environment cannot run all retargeting imports.
Use the separate GMR-Fit environment instead of installing its dependencies into the root environment.

## Test Responsibilities

Keep tests in the four domain directories.
Name each file after the behavior that it checks.

- Use `tests/embodiment/` for model generation and global scale fitting.
- Use `tests/retarget/` for job selection, cache behavior, batch execution, and export screening.
- Use `tests/rl/` for bundles, motion references, policy updates, recovery, backend preparation, and evaluation.
- Use `tests/utils/` for configuration loading, dataset selection, and result analysis.

Use `test_evaluation_batch.py` for parallel evaluation ownership and metrics.
Use `test_measurement_sensitivity.py` for measurement input conditions and paired summaries.
Use `test_body_asymmetry.py` for physical body conditions and shared test banks.
Use `test_retarget_runtime.py` for cache and batch behavior. It does not measure execution speed.
Unit tests with simulated interfaces do not validate GPU parameter installation or touch sensor values.
Run the applicable GPU checks separately.

## Distributed Checks

```bash
MPLCONFIGDIR=/tmp/myoacross-mpl .venv/bin/torchrun --standalone --nproc-per-node=2 -m tests.rl.test_distributed
TEST_DEVICE=cuda .venv/bin/torchrun --standalone --nproc-per-node=4 -m tests.rl.test_distributed
```

The first command checks CPU communication with Gloo.
The second command requires four GPUs and checks NCCL communication.
These tests compare distributed updates with updates from a combined reference batch.

### Verified GPU Checks

GPU checks passed on four RTX 5090 GPUs on 2026-10-04.
Use the locked dependency versions for these checks.

- Four-process CUDA/NCCL validation passed all five tests on each rank.
  PPO updates matched the combined reference batch for `none`, `concat`, and `rl`.
- Small distributed training passed two PPO updates with eight environments and two body groups per GPU.
  The temporary harness selected one virtual body from each of `KIT_10` and `KIT_11`.
  It switched both groups to another variant before the second update.
  No new GPU kernel load or compilation was recorded after the first update.
- Exact parameter installation, reference motion ownership, and touch observations passed on all four GPUs.
  Each rank recorded nonzero touch forces.
- Checkpoint and log checks passed. The final checkpoint recorded iteration 2 and world size 4.
  The logs recorded 128 and 256 cumulative environment steps.
- Parallel evaluation passed six four-step trials on the two subjects.
  The conditions were baseline and body-mass input biases of minus and plus five percent.
- Small generation exported one subject bundle and one virtual bundle. Both loaded successfully.

These checks used a reduced network and temporary configurations, data copies, and outputs.
They validate execution and output handling. They do not establish policy quality or full-scale training performance.

## Command Checks

Inspect a composed configuration without starting a task:

```bash
.venv/bin/python -c 'from myoacross.utils.config import load_config; from pprint import pprint; pprint(load_config("myoacross/configs/experiments/policy_comparison/train_segment_concat.yaml"))'
.venv/bin/python -m unittest tests.utils.test_config
```

Use the `.yaml` extension for configuration files. Hydra does not load `.yml` files.
Use an absolute path for a temporary YAML file outside the repository.
Temporary files can inherit `/rl/train@_global_` and the other standard configurations.
Hydra composition does not create a run directory or change the current working directory.

```bash
.venv/bin/python -m myoacross --help
.venv/bin/python -m myoacross rl --help
.venv/bin/python -m myoacross.rl.runners.multi_embodiment.backend --help
.venv/bin/python -m myoacross.utils.analysis.retarget_manifest --help
```

Use small data selections for generation, training, and evaluation checks.
Use temporary output directories.
Do not clear or overwrite formal results during tests.
Report unavailable GPU checks explicitly.

## Backend Diagnosis

Check the backend marker if multi-body initialization rejects the environment.
Start from the installed package when preparing an adapter.
Do not prepare an adapter from another adapted copy.
If source checks fail, inspect the dependency version and sensor implementation.
Do not bypass source compatibility checks.

## Plotting Tools

Import plotting modules without generating artifacts.
Run each module through its command entry point.

```bash
.venv/bin/python -m myoacross.utils.plotting.plot_morphology_conditioning --help
.venv/bin/python -m myoacross.utils.plotting.plot_morphology_conditioning --output-dir /tmp/myoacross-figures
```

The architecture command saves PNG, PDF, and SVG files.
It shows MyoFullBody MLP, Segment MLP, Segment Concat, and Segment FiLM.
It reads layer counts and dimensions from the resolved training configurations.
The two MLP schemes use the same architecture and different training bodies.
Its default output directory is `data/myoacross/figures/`.
Use `--output-dir` to write figures to a temporary directory.
Inspect the files before replacing published figures.

Inspect the sampling figure command:

```bash
.venv/bin/python -m myoacross.utils.analysis.analyze_asymmetry --help
```

The sampling command reads stored bundle samples.
Use repeated `--config` arguments to select generation configurations.
The command reads each `data.bundle_root` from the resolved YAML.
Use `--output` to select a new analysis directory. Existing directories are rejected.
Read [Paper Experiments](experiments.md) for the sampling comparison.
