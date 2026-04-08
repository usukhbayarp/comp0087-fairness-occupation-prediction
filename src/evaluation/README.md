# Fairness and Performance Evaluation Pipeline

This module implements the fairness metrics and visualisation pipeline used to evaluate occupation prediction models, ensuring consistency and reproducible charting.

The pipeline focuses on computing model accuracy against inherent bias discrepancies, mapping how different model capacities handle implicit gender associations and assessing how masking interventions scale.

---

# Overview of the Pipeline

The workflow consists of two main stages:

## 1. Metric Computation

Computes both predictive performance (Macro-F1, Accuracy) and bias signals across demographic groupings (Male vs Female). Ensure all prediction structures rely strictly on canonical ID groupings.

**Inputs:**

- Prediction JSONL files under `results/predictions/`, `results/pythia/`, and `results/pythia_finetuned/`.

**Outputs:**

- `results/tables/summary_results.csv`: High-level aggregated statistics covering Demographic Parity (DP), Equal Opportunity (EO) gaps, and F1 explicitly per model.
- `results/tables/detailed_fairness.csv`: Granular per-occupation bias gaps calculated via One-vs-Rest distributions.

**Core files:**

- `scripts/evaluate.py`: The main entry point combining predictions into summary CSVs.
- `src/evaluation/fairness.py`: Utility functions computing selection rates and One-vs-Rest TPR/FPR disparities.

---

## 2. Visualisation

Takes canonical CSV tables and produces aesthetic metric dashboards demonstrating the limits of mitigating interventions natively.

**Outputs:**

- **Pareto Frontier (Unlabelled)** (`pareto_frontier.png`): Plotting model configurations mapping Macro-F1 against the Equalized Odds Gap.
- **Pareto Frontier (Labelled)** (`pareto_frontier_labelled.png`): The same as before, but labels have been manually added (using Microsoft PowerPoint)
- **Job Bias Analysis** (`job_bias_comparison.png`): Explicit visual mappings showing exact TPR/FPR reduction metrics mapping directly against job types between explicitly paired `(Masked)` and `(Unmasked)` variants.
- **Scaling Laws** (`scaling_performance.png`, `scaling_fairness.png`): Effects of scaling model sizes (e.g., 160M -> 410M -> 1.4B) over capabilities versus bias magnification.
- **Correlation Plots**: 
  - `results/figures/Correlation Plots/Amplification/`: Maps gender ratio percentage representations directly tracking bias scaling.
  - `results/figures/Correlation Plots/Delta/`: Evaluates specific models' success in limiting explicitly biased correlations via scatter reductions.

**Core file:**

- `src/evaluation/plots.py`: Evaluator executing matplotlib routines enforcing explicit label parsing natively.

---

# File Structure

## Scripts

- `scripts/evaluate.py`
  Runs end-to-end evaluation compiling predictions while selectively enforcing explicitly uniform labelling strategies mapping models.

## Source Code

- `src/evaluation/fairness.py`
  Provides `compute_fairness_gaps` natively defining gaps bridging `M` and `F`.

- `src/evaluation/plots.py`
  Stand-alone graphic generation compiling unified `(Masked)` and `(Unmasked)` annotations against pre-defined data pools.

---

# Expected Inputs

## Prediction JSONL files

Each jsonl under evaluation expects:

- `id`: Universal matching index preventing disjoint subsets.
- `label_pred`: Actual string inferences mapped natively.
- `label_true`: Original mappings matching validation natively.
- `gender`: Grounded `'M'` or `'F'` evaluation flags.

---

# How to Run (from Project Root)

> **Important:** All scripts must be run as Python modules from the project root, not as direct script paths. This ensures `src` modules map uniformly.
> 
> ```bash
> # ❌ Do NOT run — causes ModuleNotFoundError: No module named 'src'
> python scripts/evaluate.py
> 
> # ✅ Correct usage
> python -m scripts.evaluate
> ```

### Step 1 — Run Evaluation Metrics

Compiles evaluation mappings calculating exact Demographic and TPR/FPR properties against isolated prediction paths smoothly.

```bash
python -m scripts.evaluate
```

### Step 2 — Generate Plots

Relies directly on `results/tables` existing outputs correctly parsed into uniform plotting subroutines explicitly rendering cleanly.

```bash
python -m src.evaluation.plots
```

---

## Output Structure

```text
results/
  tables/
    detailed_fairness.csv
    summary_results.csv
  figures/
    Correlation Plots/
      Amplification/
        1.4b_finetuned_extract.png
        ...
      Delta/
        1.4b_finetuned_extract.png
        ...
    job_bias_comparison.png
    pareto_frontier.png
    scaling_fairness.png
    scaling_fairness_breakdown.png
    scaling_performance.png
```

## Reproducibility Notes

- Correlation auto-grouping pairs equivalent model arrays smoothly using `(Unmasked)` mapped to `(Masked)` pairs accurately. All scaling functions match directly natively onto generic log domains natively!
