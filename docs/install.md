# Installation and Data Preparation

Use Linux for the current workflow.
GPU simulation requires an NVIDIA GPU and a compatible driver.
Install Git, uv, and a C/C++ build toolchain.

## Environments

| Project | Python | Use |
| --- | --- | --- |
| Repository root | 3.11 | Embodiment generation, RL, evaluation, and playback |
| `tools/skel` | 3.12 | SKEL measurement extraction |
| `tools/gmr_fit` | 3.11 | Motion retargeting |
| `tools/moshpp` | 3.7 | Gait120 marker fitting with MoSh++ |

```bash
uv sync --locked
uv sync --locked --project tools/skel
uv sync --locked --project tools/gmr_fit
uv run python -m myoacross --help
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Keep the lockfiles.
The root and GMR-Fit environments include Hydra and OmegaConf for configuration loading.
Use the SKEL environment for measurement extraction.
Use the GMR-Fit environment for retargeting and its tests.
The root environment does not contain SKEL or GMR-Fit.

## Analysis Figures

Install the optional figure dependencies:

```bash
uv sync --locked --extra analysis
```

The analysis scripts use SciencePlots with `science`, `nature`, `no-latex`, and `bright`.
They do not require a LaTeX installation.

## External Models

Download model parameters from their providers.
Install model parameters separately from Python packages.

| Model | Location |
| --- | --- |
| SKEL | `data/models/skel/skel_male.pkl` and `skel_female.pkl` |
| SMPL-H | `data/models/smplh/SMPLH_MALE.pkl`, `SMPLH_FEMALE.pkl`, and `SMPLH_NEUTRAL.pkl` |

Obtain parameters from the [SKEL provider](https://skel.is.tue.mpg.de/) and the [SMPL-H provider](https://mano.is.tue.mpg.de/).
Check each provider's license.

SKEL measurement extraction uses ten shape coefficients.
Retargeting uses sixteen SMPL-H shape coefficients.
MyoFullBody assets come from the locked `musclemimic-models` dependency.
Store only SKEL and SMPL-H model parameters under `data/models/`.
Keep the `smplx` Python package. It supplies the SMPL-H implementation.

## Gait120 Dependencies

Use the root environment for references, evaluation, and analysis.
Install MM-EvalKit with `uv sync --locked --extra gait120`.
Reuse the SKEL and GMR-Fit environments for measurement and retargeting.
Use only one additional MoSh++ environment for marker fitting.

The released fast-derivative archive contains CPython 3.7 binaries without build sources.
Use Python 3.7 for this environment. Python 3.10 cannot load these binaries.
Create the Conda environment under `tools/moshpp/` from the repository root:

```bash
conda env create --prefix tools/moshpp/.venv --file tools/moshpp/environment.yaml
moshpp_env="$(pwd)/tools/moshpp/.venv"
moshpp_python="$moshpp_env/bin/python"
"$moshpp_python" -m pip install --no-build-isolation chumpy==0.70
```

Install the tested upstream revisions with the MoSh++ interpreter:

```bash
"$moshpp_python" -m pip install --no-build-isolation --no-deps \
  "human-body-prior @ git+https://github.com/nghorbani/human_body_prior.git@f9afbbd4dac4d58270bf156b56b6f37831e9febd" \
  "body-visualizer @ git+https://github.com/nghorbani/body_visualizer.git@5d0ac4615f32563f5dbd2f09c24aee75e3ae62ff" \
  "psbody-mesh @ git+https://github.com/MarilynKeller/mesh.git@e8b04be3aaa1d262a3b7a05ef9e4085b4836b049"
CUDA_VISIBLE_DEVICES="" PYTORCH3D_FORCE_NO_CUDA=1 MAX_JOBS=4 \
  PATH="$moshpp_env/bin:$PATH" "$moshpp_python" -m pip install --no-build-isolation --no-deps \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
"$moshpp_python" -m pip install --no-build-isolation --no-deps \
  --src "$moshpp_env/sources" \
  -e "moshpp @ git+https://github.com/nghorbani/moshpp.git@0addb01096d1a6469837b172e9f778f3562b5e0e"
```

The editable MoSh++ source must remain in `tools/moshpp/.venv/sources/moshpp/`.
MoSh++ finds its `support_data` relative to that source. Do not place the checkout in a directory named `src`.
The Gait120 configuration already points to `tools/moshpp/.venv/bin/python`.
Policy evaluation uses the existing root environment and its GPU dependencies.

Obtain these archives from the registered [SOMA download portal](https://soma.is.tue.mpg.de/):

| Archive | Use |
| --- | --- |
| [smpl-fast-derivatives.tar.bz2](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=smpl-fast-derivatives.tar.bz2) | Install `psbody/smpl` in the Python 3.7 environment. |
| [extra_smplx_data.tar.bz2](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=smplx/extra_smplx_data.tar.bz2) | Supply `pose_body_prior.pkl`. |

The verified prior has eight components and 63 body-pose dimensions.
Its `means`, `covars`, and `weights` fields match the MoSh++ SMPL-H interface.
Reuse the native MANO hand components and means from the existing SMPL-H model.
Keep the prepared hand prior and its source record under `data/experiments/gait120/resources/moshpp/smplh/`.
Do not substitute the extra SMPL-X hand prior for these SMPL-H parameters.

Prepare a compatible copy of `SMPLH_MALE.pkl` under `data/experiments/gait120/resources/moshpp/smplh/`.
Convert its dense `J_regressor` to a SciPy CSC matrix without changing entries.
Decode the scalar byte-array fields `bs_style` and `bs_type` as ASCII strings.
Preserve all model parameters and record the source and transformations.
The Gait120 run configuration points `moshpp.model_path` to this copy.
Keep the repository model unchanged.

Build the MoSh++ mesh-distance extension with the Eigen and TBB libraries from the Conda environment.
Install the registered fast-derivative archive into the same environment, then check the complete fitting import:

```bash
(cd "$moshpp_env/sources/moshpp/src/moshpp/scan2mesh/mesh_distance" && \
  PATH="$moshpp_env/bin:$PATH" CPATH="$moshpp_env/include" \
  LIBRARY_PATH="$moshpp_env/lib" LD_LIBRARY_PATH="$moshpp_env/lib" \
  "$moshpp_python" setup.py build_ext --inplace)
tar -xjf data/experiments/gait120/resources/moshpp/downloads/smpl-fast-derivatives.tar.bz2 \
  -C "$moshpp_env/lib/python3.7/site-packages"
"$moshpp_python" -c "from moshpp.chmosh import mosh_stagei"
```

Keep model parameters, priors, run configurations, and outputs under `data/`.
Read [Gait120 Physiological Evaluation](experiments.md#gait120-physiological-evaluation) for commands and validation status.

## Datasets and Splits

Store AMASS KIT and BMLmovi source data under `data/dataset/`.
Preserve the tracked `subjects.json` files.
They contain subject attributes, including body mass.

Read [Datasets](dataset.md) for source paths, subject metadata, fixed splits, and motion selection.

## Generated Data

Store measurements, body models, and retargeted motions under `data/myoacross/`.
Store standard RL outputs under `data/myoacross/rl/`.
Store experiment outputs under `data/experiments/`.
Read the [basic workflow](../README.md) and [experiment procedures](experiments.md).

## Image2MSKSim Environment

Keep HSMR in a separate environment. Do not install it in the root environment.
Install the pinned Git dependencies with uv:

```bash
uv sync --locked --project tools/hsmr
```

HSMR and its SKEL dependency use fixed Git revisions in `tools/hsmr/pyproject.toml`.
Do not clone either repository manually.
Use Python 3.11 and a CUDA build of PyTorch that supports the GPU.
Use the pinned `setuptools==75.3.4` dependency. Detectron2 uses its `pkg_resources` interface.
Detectron2 requires a C++ build toolchain. A CUDA build also requires the CUDA toolkit.

The upstream HSMR wheel omits some inference modules and configuration files.
The reconstruction entry downloads the missing original files from the same Git revision on first use.
It adds these files to the HSMR environment. It does not overwrite installed source files.
Prepare these files separately when needed:

```bash
tools/hsmr/.venv/bin/python -m myoacross.utils.image2msksim.reconstruct --prepare-environment
```

Follow the [HSMR resource instructions](https://github.com/IsshikiHugh/HSMR/blob/main/docs/SETUP.md).
Put the released HSMR model in `data/experiments/image2msksim/resources/HSMR-ViTH-r1d1/`.
This directory must contain `.hydra/config.yaml` and `checkpoints/hsmr.ckpt`.
Put the three HSMR joint regressors in `data/experiments/image2msksim/resources/body_models/`.
Use the existing SKEL v1.1.1 parameters through a symbolic link:

```bash
mkdir -p data/experiments/image2msksim/resources/body_models
ln -s ../../../../models/skel data/experiments/image2msksim/resources/body_models/skel
```

The reconstruction entry sets the HSMR input directory to the parent of `hsmr.model_root`.
It sets the HSMR output directory to the experiment output root.
The upstream detector downloads its ViTDet weights on first use.
Set `cache_root` to `data/experiments/image2msksim/cache/`.
Share this cache across single-subject and batch runs.
Store detector, PyTorch, backend, and GPU compilation caches in this directory.
HSMR model weights, detector weights, and SKEL parameters are separate resources.
Environment installation does not supply them.

## Independent Checkout

Clone MyoAcross into its own directory. Install each environment from its manifest.
Do not set `PYTHONPATH` to a previous project checkout.
Use the root environment for RL and analysis. Use each auxiliary environment for its documented task.
Keep the root environment in `.venv` and each auxiliary environment under `tools/<name>/.venv`.
Keep run configurations under `myoacross/configs/`.

## Use Existing Experiment Data

Keep policy weights, normalization statistics, and numerical results unchanged.
Set evaluation checkpoint, bundle, retarget, and output paths explicitly.
Use a separate empty output directory for each new evaluation.

Generated XML files can contain absolute mesh and texture paths.
Bundle indexes and retarget reports can contain absolute model and dataset paths.
After moving data, update these runtime path fields to the new locations.
Check that every new path identifies the same model or source data.
Use the model package installed by the new checkout's lockfile.
Keep historical run configurations and logs unchanged when updating paths.

The Python package name does not change saved policy parameter names.
Restore checkpoints with their saved network settings and normalization statistics.
Evaluation can use explicit paths to the relocated data.

Exact training recovery checks saved paths and a fingerprint of model and retarget metadata.
Path-only edits can change this fingerprint. Do not bypass this check or change an original
checkpoint to claim exact recovery. Validate recovery separately before continuing a historical run.
New training runs use the current data paths and metadata.
