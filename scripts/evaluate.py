import pandas as pd
import numpy as np
import json
import glob
import os
from sklearn.metrics import f1_score, accuracy_score
from src.evaluation.fairness import compute_fairness_gaps, calculate_exact_eo_diff
from scipy.stats import bootstrap

def load_predictions(file_path):
    """
    Load predictions from a JSONL file.

    Parameters
    ----------
    file_path : str
        Path to the JSONL file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the predictions.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded predictions from {file_path}")
    return pd.DataFrame(data)

def _load_canonical_ids(roots):
    """Return the canonical set of 3 000 sample IDs that every
    prediction file shares.  These IDs are used to filter larger files
    (e.g. encoder predictions with > 3000 rows) so that every model is
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
    """
    Run evaluation on all prediction files.

    Two CSV files are produced:
    - summary_results.csv: summary metrics for each model
    - detailed_fairness.csv: per-occupation fairness metrics for each model
    """
    summary_data = []
    detailed_data = []
    
    # Gather every prediction file
    pred_files = (
        glob.glob("results/predictions/**/*.jsonl", recursive=True)
        + glob.glob("results/pythia/**/*.jsonl", recursive=True)
        + glob.glob("results/pythia_finetuned/**/*.jsonl", recursive=True)
    )
    
    # Ignore files that have '_extract' in their path
    pred_files = [f for f in pred_files if '_extract' not in f]
    
    assert pred_files, "No prediction JSONL files found under results/."
    
    # Canonical ID set: the 3 000 sample IDs present in every prediction file.
    # All prediction files are filtered to exactly these IDs so that
    # performance and fairness metrics are comparable across models.
    canonical_ids = _load_canonical_ids(["results/pythia", "results/pythia_finetuned", "results/predictions"])
    
    for file in pred_files:
        # load predictions for the given model
        df = load_predictions(file)
        
        # Keep only the canonical 3 000 IDs and verify none are missing.
        df = df[df["id"].isin(canonical_ids)]
        missing = canonical_ids - set(df["id"].tolist())
        assert not missing, (
            f"{file}: missing {len(missing)} of {len(canonical_ids)} "
            f"canonical IDs (first 5: {sorted(missing)[:5]})"
        )

        # Normalize occupation names: encoder models (RoBERTa, DistilBERT) use
        # spaces (e.g. "software engineer") while Pythia uses underscores.
        # Standardize to underscores for consistency across all models.
        df['label_true'] = df['label_true'].str.replace(' ', '_')
        df['label_pred'] = df['label_pred'].str.replace(' ', '_')

        # performance metrics for the given model
        macro_f1 = f1_score(df['label_true'], df['label_pred'], average='macro')
        accuracy = accuracy_score(df['label_true'], df['label_pred'])
        
        # fairness metrics 
        # fairness variable is a dict mapping each occupation to its
        # per-occupation gaps: {occ: {"Demographic_Parity", "EO_TPR_Gap", "EO_FPR_Gap"}}
        fairness = compute_fairness_gaps(df)
        
        # average gaps across occupations
        # for the given model will return the DP, TPR, FPR, and EO gaps for each occupation
        dp_gaps  = [v["Demographic_Parity"] for v in fairness.values()]
        tpr_gaps = [v["EO_TPR_Gap"] for v in fairness.values()]
        fpr_gaps = [v["EO_FPR_Gap"] for v in fairness.values()]
        
        avg_dp       = np.nanmean(dp_gaps)
        avg_tpr_gap  = np.nanmean(tpr_gaps)
        avg_fpr_gap  = np.nanmean(fpr_gaps)
        eo_diff      = max(avg_tpr_gap, avg_fpr_gap)  # standard EO difference
        
        # Extract clean model name from the file path
        raw_name = os.path.basename(file).replace("preds_", "").replace(".jsonl", "")
        
        # Clearly dictate whether it is masked or unmasked for the tables
        if 'unmasked' in raw_name.lower():
            model_name = f"{raw_name} (Unmasked)"
        elif 'masked' in raw_name.lower():
            model_name = f"{raw_name} (Masked)"
        else:
            model_name = f"{raw_name} (Unmasked)"
        

        # Prepare the index array for the sampler
        indices = np.arange(len(df))
        
        # Run the bootstrap
        res = bootstrap(
        (indices,), 
        lambda idx: calculate_exact_eo_diff(idx, df), # No need to pass all_occ
        n_resamples=5000, 
        method='BCa', 
        confidence_level=0.95,
        vectorized=False
        )

        summary_data.append({
            "model_name":   model_name,
            "macro_f1":     macro_f1,
            "accuracy":     accuracy,
            "dp_diff":      avg_dp,
            "eo_diff":      eo_diff,
            "eo_ci_low":    res.confidence_interval.low,
            "eo_ci_high":   res.confidence_interval.high,
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
