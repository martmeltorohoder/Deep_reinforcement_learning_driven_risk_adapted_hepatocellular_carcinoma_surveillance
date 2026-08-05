# Deep reinforcement learning-driven risk-adapted hepatocellular carcinoma surveillance in metabolic dysfunction-associated steatotic liver disease

This repository contains the SC-DSMDP surveillance environment and learning system. It couples a calibrated MASLD natural-history simulator with an Implicit Quantile Network, lower-tail Conditional Value-at-Risk action selection, constrained Markov decision process optimization, conservative value regularization, and imaging-adequacy action masks. The policy selects among 18 interval-modality combinations and tracks detection, missed cancers, screening burden, cost, and quality-adjusted survival.

## Method scope

The patient state has 14 values: five fibrosis indicators, three fibrosis-trajectory indicators, BMI, age, diabetes, imaging adequacy, time since screening, and the preceding result. Actions combine 3-, 6-, and 12-month intervals with ultrasound, ultrasound plus AFP, GALAD, abbreviated MRI, liquid biopsy, or no screening. Severe ultrasound visualization limitation removes ultrasound actions before selection.

The return model uses three 256-unit ReLU state layers, a 64-term cosine quantile embedding, 64 online and target samples, and an action-conditional head. Decisions maximize the lower 5% tail mean. The objective combines quantile Huber error, CQL regularization, and Lagrangian penalties for missed cancers, guideline interval violations, and annual budget excess.

## Installation

Python 3.10 and a CUDA 12.1-capable Linux environment are the supported training target.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Conda:

```bash
conda env create -f environment.yml
conda activate scdsmdp
pip install -e .
```

Docker:

```bash
docker build -t scdsmdp .
docker run --gpus all --rm -v "$PWD/runs:/workspace/runs" scdsmdp
```

## Data

All external data portals are listed in `dataset_links.txt`. The simulator runs from published aggregate parameters and does not include person-level records.

NHANES 2017–2018 public-use components are available as XPT files from the CDC. The cycle is used to calibrate age, obesity, diabetes, and fibrosis-stage population distributions. Access is public under the CDC data-use terms. Download only the components required for a calibration analysis; the simulator itself does not require those files at runtime.

SEER incidence data are supplied by the National Cancer Institute. The current research release requires an access request and acceptance of the applicable data agreement. No credentials, access tokens, extracts, or restricted records are stored here.

GBD 2021 estimates are accessed through the IHME Global Health Data Exchange results tool under its terms. They are used for population-level incidence validation rather than model training.

The fibrosis transition, diagnostic accuracy, and economic inputs reported in the manuscript come from published aggregate studies. They are encoded in `scdsmdp/environment` and require no patient-level download.

## Training

The primary configuration retains the reported values: batch size 256, learning rate `3e-4`, cosine annealing, discount `0.95`, CQL coefficient `1.0`, CVaR level `0.05`, target Polyak rate `0.005`, and approximately 680,000 episodes. Twenty independent seeds are defined in `settings/main.yaml`.

```bash
scdsmdp-train --config settings/main.yaml --output runs/seed_0
```

The paper reports training on one NVIDIA A100 80GB with approximately 9.8 GPU-hours for the primary method. Running all twenty seeds requires approximately 196 A100 GPU-hours if executed serially. Parallel execution can use the supplied launcher after mapping seeds to available devices. The implementation does not require 80GB for the network alone; the reported hardware is retained as the reference compute profile.

```bash
bash tools/train_all_seeds.sh
```

The launcher expects twenty visible devices and maps one run to each. For smaller clusters, invoke the training command separately for each seed after changing `seed` in a copied YAML configuration.

## Evaluation

Evaluate a fixed clinical or risk-adaptive policy with 50,000 held-out trajectories:

```bash
scdsmdp-evaluate --config settings/main.yaml --policy aasld --episodes 50000
scdsmdp-evaluate --config settings/main.yaml --policy amri --episodes 50000
scdsmdp-evaluate --config settings/main.yaml --policy risk_adaptive --episodes 50000
```

The manuscript aggregates the primary metrics across twenty seeds as median and interquartile range. Bootstrap intervals use 5,000 resamples. Paired method comparisons use the Wilcoxon signed-rank test with Bonferroni correction for fifteen primary comparisons.

The reported principal policy targets are early-stage detection `65.2%`, sensitivity `80.4%`, missed cancer rate `0.7%`, overscreening reduction `31.2%`, ICER `$62,400/QALY`, and `1.42` screens per year. Simulation estimates are stochastic and depend on the complete calibrated parameter table; differences should be investigated at the transition, observation, reward, and cohort-calibration levels rather than hidden by post-processing.

## Configuration boundaries

The manuscript does not state weight decay, warmup duration, gradient accumulation, mixed-precision mode, CUDA version, PyTorch version, or replay capacity. The dependency and replay choices in this repository are engineering selections. They are not presented as reported experimental facts. The paper does report the model dimensions, optimizer learning rate, scheduler family, batch size, risk level, discount, target rate, CQL coefficient, dual learning rate, stopping criteria, seed count, held-out fraction, and evaluation rollout count.

## Verification

```bash
pytest -q
ruff check .
mypy --strict scdsmdp
```

The test suite covers action encoding, masking, population vectors, deterministic transitions, finite-horizon execution, IQN tensor structure, quantile loss, CQL loss, masked selection, parameter updates, clinical metrics, calibration, bootstrap inference, corrected paired testing, atomic training output, and resume state restoration.

## Responsible use

This software is a research simulation and is not a medical device. It does not provide patient-specific clinical recommendations. Deployment would require external validation, local calibration, prospective evaluation, governance review, security controls, and monitoring of performance and disparities across age, sex, BMI, diabetes, etiology, and care setting.
