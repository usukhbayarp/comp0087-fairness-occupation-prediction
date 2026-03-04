## File Descriptions

### 1. `clf_report_pythia_<size>_finetuned.txt`

A plain-text classification report following the format of scikit-learn's `classification_report`. It lists per-class precision, recall, F1-score, and support for each of the 20 occupation categories.

**Occupation classes:** accountant, architect, attorney, comedian, composer, dentist, dietitian, filmmaker, journalist, model, nurse, painter, photographer, physician, poet, professor, psychologist, software\_engineer, surgeon, teacher.

### 2. `metrics_pythia_<size>_finetuned.json`

A JSON file containing high-level evaluation metrics with the following fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Model name (`pythia-1.4b`) |
| `regime` | string | Training regime (`finetuned`) |
| `accuracy` | float | Overall accuracy |
| `macro_f1` | float | Macro-averaged F1 score |
| `num_samples` | int | Total number of evaluation samples |
| `gender_accuracy` | object | Accuracy broken down by gender |

### 3. `preds_pythia_<size>_finetuned.jsonl`

A JSONL file (one JSON record per line) containing per-sample predictions for all 95,468 samples. Each record includes the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | int | Sample index |
| `text` | string | Input biography text |
| `label_true` | string | Ground-truth occupation label |
| `label_pred` | string | Model-predicted occupation label |
| `gender` | string | Gender annotation (`M` / `F`) |
| `model` | string | Model name |
| `regime` | string | Training regime |
| `score` | float | Raw model score (log-probability) |
| `conf` | float | Prediction confidence (0–1) |

---

## Quick Start

```python
import json
import pandas as pd

# Load summary metrics
with open("metrics_pythia_1.4b_finetuned.json") as f:
    metrics = json.load(f)

# Load per-sample predictions as a DataFrame
preds = pd.read_json("preds_pythia_1.4b_finetuned.jsonl", lines=True)
```
