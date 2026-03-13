import pandas as pd
import numpy as np
import json
import glob
import os
from sklearn.metrics import f1_score, accuracy_score
from src.evaluation.fairness import compute_fairness_gaps

def load_predictions(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded predictions from {file_path}")
    return pd.DataFrame(data)

def _load_canonical_ids(roots):
    """Return the canonical set of 3 000 sample IDs that every Pythia
    prediction file shares.  These IDs are used to filter larger files
    (e.g. encoder predictions with 95 k rows) so that every model is
    evaluated on the exact same subset.

    Parameters
    ----------
    roots : list[str]
        Directory paths to scan for JSONL files (searched recursively).
        The first file with <= 3 000 rows is used as the canonical source.
    """
    for root in roots:
        for path in glob.glob(os.path.join(root, "**/*.jsonl"), recursive=True):
            df = load_predictions(path)
            if len(df) <= 3000:
                return set(df["id"].tolist())
    raise RuntimeError(
        "Could not find a <= 3000-row JSONL in the provided roots to "
        "establish the canonical ID set."
    )

def run_evaluation():
    summary_data = []
    detailed_data = []
    
    # Gather every prediction file
    pred_files = (
        glob.glob("results/predictions/**/*.jsonl", recursive=True)
        + glob.glob("results/pythia/**/*.jsonl", recursive=True)
        + glob.glob("results/pythia_finetuned/**/*.jsonl", recursive=True)
    )
    assert pred_files, "No prediction JSONL files found under results/."
    
    # Canonical ID set: the 3 000 sample IDs present in every Pythia file.
    # All prediction files are filtered to exactly these IDs so that
    # performance and fairness metrics are comparable across models.
    canonical_ids = _load_canonical_ids(["results/pythia", "results/pythia_finetuned"])
    
    for file in pred_files:
        df = load_predictions(file)
        
        # Keep only the canonical 3 000 IDs and verify none are missing.
        df = df[df["id"].isin(canonical_ids)]
        missing = canonical_ids - set(df["id"].tolist())
        assert not missing, (
            f"{file}: missing {len(missing)} of {len(canonical_ids)} "
            f"canonical IDs (first 5: {sorted(missing)[:5]})"
        )

        # performance metrics
        macro_f1 = f1_score(df['label_true'], df['label_pred'], average='macro')
        accuracy = accuracy_score(df['label_true'], df['label_pred'])
        
        # fairness metrics — dict mapping each occupation to its
        # per-occupation gaps: {occ: {"Demographic_Parity", "EO_TPR_Gap", "EO_FPR_Gap"}}
        fairness = compute_fairness_gaps(df)
        
        # average gaps across occupations
        dp_gaps  = [v["Demographic_Parity"] for v in fairness.values()]
        tpr_gaps = [v["EO_TPR_Gap"] for v in fairness.values()]
        fpr_gaps = [v["EO_FPR_Gap"] for v in fairness.values()]
        
        avg_dp       = np.nanmean(dp_gaps)
        avg_tpr_gap  = np.nanmean(tpr_gaps)
        avg_fpr_gap  = np.nanmean(fpr_gaps)
        eo_diff      = max(avg_tpr_gap, avg_fpr_gap)  # standard EO difference
        
        # Extract clean model name from the file path
        model_name = os.path.basename(file).replace("preds_", "").replace(".jsonl", "")
        
        summary_data.append({
            "model_name":   model_name,
            "macro_f1":     macro_f1,
            "accuracy":     accuracy,
            "dp_diff":      avg_dp,
            "eo_diff":      eo_diff,
            "avg_tpr_gap":  avg_tpr_gap,
            "avg_fpr_gap":  avg_fpr_gap,
        })
        
        # per-occupation gaps for detailed analysis
        for occ, metrics in fairness.items():
            detailed_data.append({
                "model_name":  model_name,
                "occupation":  occ,
                "dp_gap":      metrics["Demographic_Parity"],
                "eo_tpr_gap":  metrics["EO_TPR_Gap"],
                "eo_fpr_gap":  metrics["EO_FPR_Gap"],
            })
    
    # Save outputs to the tables directory
    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame(summary_data).to_csv("results/tables/summary_results.csv", index=False)
    pd.DataFrame(detailed_data).to_csv("results/tables/detailed_fairness.csv", index=False)
    print("Evaluation complete. Results saved to results/tables/")

if __name__ == "__main__":
    run_evaluation()
