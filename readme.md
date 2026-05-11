# crispr-explainability

CRISPR off-target prediction with a neural Structural Causal Model (SCM).
The goal is not just predictive performance: the model is structured to
**isolate the thermodynamic mechanism** (PAM gate + positional penalties along
the spacer) and to support **counterfactual reasoning** (`do(...)`) over
interventions on the guide and on the off-target.

Work is organized in four phases, tracked in [`doc/findings.md`](doc/findings.md).

---

## Current state (Phase 4)

- **Active model**: `experiments/results/Exp15_Positional_ExtendedOneCycle/`
  — `positional_mlp` architecture with a GC context head (`context_dim=3`),
  trained on CHANGEseq positives + negatives, evaluated cross-assay on GUIDEseq.
- **Batch counterfactual pipeline**: `explainability/simulate_intervention_batch.py`
  runs abduction + intervention + prediction over 67k CHANGEseq pairs and 1.6k
  GUIDEseq pairs, producing Pareto trade-off plots and noise distributions.
- **Key findings**:
  - F1 — cross-assay calibration bias (AUPRC CHANGEseq → GUIDEseq −57%).
  - F7 — deep complexity hurts generalization; the linear bypass with hard
    priors is preferable.
  - F8 — `y_obs_on=99%` saturation made on-target abduction degenerate.
    Fix: `--on-target-mode {drop, per_run}`.
  - F9 — the `U_off` gap between CHANGEseq (in vitro, mean +2.34) and GUIDEseq
    (in vivo, mean −0.14) is a scale mismatch between cell-free and cellular
    regimes, not a model error.
  - F10 — fixed interventions (5' truncation, pos15→A mutation) do not pass the
    Pareto test on either dataset; a guide-specific rescue mutation is needed.

Numerical details and tables in [`doc/findings.md`](doc/findings.md).

---

## Repository structure

```
dag/                       DAG-based feature engineering (mismatch, PAM,
                           energetics, independence tests, parametric SCM)
models/
  baseline/                XGBoost + CatBoost with DAG features
  deep/
    encoding.py            BiologicalMismatchEncoder, PairwiseTokenEncoder,
                           ContextAwareMismatchEncoder
    modules.py             PAMModule, SpacerRegionModule, MismatchVectorModule,
                           TypedMismatchModule
    neural_scm.py          NeuralSCM (8 architectures: positional_mlp, deep_scm,
                           mini_mlp, typed_mlp, learned_mlp, context_aware_mlp,
                           linear_bypass, …)
    train.py               training loop with Focal Loss, OneCycleLR, IRM
evaluation/                unified interface (predict_proba, explain)
explainability/
  benchmark_models.py      AUROC/AUPRC comparison across variants
  explain_thermodynamics.py  positional profile (W_i * s_i) per mismatch chemistry
  simulate_intervention.py       single-pair counterfactual
  simulate_intervention_batch.py batch counterfactual CSV + Pareto + U-dist
  ig.py / shap_utils.py / attention.py  attribution methods
experiments/
  exp_01_baseline/         XGBoost + CatBoost
  exp_02_scm/              parametric SCM + independence tests
  exp_03_neural_scm/       Neural SCM (run.py + config.yaml)
  results/                 output of all experiments (Exp03 → Exp15)
doc/
  findings.md              finding tracker (F1–F11)
  phase_1.md / phase_2.md / plan.md / project_report.md
data/
  raw/{changeseq,guideseq}/    positive + negative datasets
  processed/                   feature engineering outputs
```

---

## Setup

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```powershell
uv --version  # if missing:
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv sync --extra dev
copy .env.example .env   # then fill in DATA_DIR, RESULTS_DIR, WANDB_KEY
```

Torch is installed from the CUDA 13.0 index (see `pyproject.toml`); for CPU-only
builds edit the `[tool.uv.sources]` section.

---

## Workflow

### Training (baseline + Neural SCM)

```powershell
make data           # extracts DAG features into data/processed/features/
make train          # crispr-exp01 (baseline) + exp02 (SCM) + exp03 (Neural SCM)
make eval           # cross-experiment benchmark in experiments/results/
```

Without `make`:

```powershell
uv run python -m dag.features --input data/raw --output data/processed/features/features.parquet
uv run crispr-exp01
uv run crispr-exp02
uv run crispr-exp03
uv run crispr-benchmark --results-dir experiments/results --output experiments/results/benchmark_metrics.csv
```

To train a specific Neural SCM variant, edit
`experiments/exp_03_neural_scm/config.yaml` (`architecture` field) and re-run
`crispr-exp03`. The checkpoint lands in `experiments/results/<run_name>/`.

### Counterfactual analysis

**Single-pair** (sanity check on a known example):

```powershell
uv run python explainability/simulate_intervention.py
```

**Batch** (full population, output CSV + Pareto + U-distribution):

```powershell
# GUIDEseq (in vivo) — per_run on-target abduction
uv run python explainability/simulate_intervention_batch.py --dataset guideseq

# CHANGEseq (in vitro) — drop on-target abduction (no run column)
uv run python explainability/simulate_intervention_batch.py --dataset changeseq

# Override model or batch size
uv run python explainability/simulate_intervention_batch.py `
    --dataset guideseq `
    --model_path experiments/results/Exp15_Positional_ExtendedOneCycle/neural_scm.pt `
    --batch_size 1024
```

Outputs in `explainability/batch_results/`:

- `<dataset>_batch_results.csv`            — one row per pair (21 columns)
- `<dataset>_per_guide_medians.csv`        — median within guide (defragments
                                              the bias from guides with many off-targets)
- `<dataset>_pareto.png`                   — (Δon, Δoff) scatter per intervention
- `<dataset>_U_distribution.png`           — histograms of inferred noise

See F8–F11 in `findings.md` for the correct reading of these plots.

### Thermodynamic explainability

Positional profile `|W_i × s_i|` per mismatch chemistry (Match / Wobble /
Transition / Transversion) — visualizes the "causal X-ray" of the
`positional_mlp` model:

```powershell
uv run python explainability/explain_thermodynamics.py
```

Output: `explainability/plots/thermodynamic_profile.png`.

---

## Tests

```powershell
uv run pytest -q
```

---

## Architectural principles

1. `dag/` is independent of `models/` and `explainability/`. DAG features are
   an *input* for the baseline and a *structural guide* for the neural SCM.
2. `evaluation/` exposes a uniform `predict_proba(...)` and `explain(...)` API
   across all models (baseline + Neural SCM).
3. `experiments/<exp>/run.py` is orchestration only: loads config, invokes
   modules, saves output. No model logic inside.
4. The Neural SCM is built from **independent modules** assembled in an
   explicit DAG (PAM gate + non-seed + seed-extension + proximal + optional GC
   context). Each architecture is a different combination of modules on the
   same interface (`models/deep/neural_scm.py`).

---

## Open work streams

Tracked under `## Todo Phase 4` in `findings.md`:

- Guide-specific rescue mutation (replaces the fixed pos15 mutation).
- Stratification of `U_off` on CHANGEseq by GC% and chromatin accessibility, to
  validate the "U_off = cell-free saturation" hypothesis (F9).
- Bootstrap CIs on per-guide means of counterfactual Δ values.
- SCM architecture with an assay-specific calibration head (in vitro vs in vivo)
  to address the F1/F9 bias in a principled way.
