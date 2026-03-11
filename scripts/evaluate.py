import pandas as pd
import numpy as np
import json
import glob
from sklearn.metrics import f1_score, accuracy_score
from src.evaluation.fairness import compute_fairness_gaps

def load_predictions(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def run_evaluation():
    summary_data = []
    detailed_data = []
    
    # Matches the shared JSONL format
    pred_files = glob.glob("preds_*.jsonl")
    
    for file in pred_files:
        df = load_predictions(file)
        
        # performance metrics
        macro_f1 = f1_score(df['label_true'], df['label_pred'], average='macro')
        accuracy = accuracy_score(df['label_true'], df['label_pred'])
        
        # fairness metrics
        fairness = compute_fairness_gaps(df)
        
        # average gaps across occupations (using np.nanmean to safely ignore np.nan values)
        dp_gaps = [v["Demographic_Parity"] for v in fairness.values()]
        eo_gaps = [v["Equal_Opportunity"] for v in fairness.values()]
        
        avg_dp = np.nanmean(dp_gaps)
        avg_eo = np.nanmean(eo_gaps)
        
        summary_data.append({
            "model_name": file.replace("preds_", "").replace(".jsonl", ""),
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "dp_diff": avg_dp,
            "eo_diff": avg_eo
        })
        
        # also keep per-occupation gaps for detailed analysis
        for occ, metrics in fairness.items():
            detailed_data.append({
                "model_name": file.replace("preds_", "").replace(".jsonl", ""),
                "occupation": occ,
                "dp_gap": metrics["Demographic_Parity"],
                "eo_gap": metrics["Equal_Opportunity"]
            })
    
    pd.DataFrame(summary_data).to_csv("summary_results.csv", index=False)
    pd.DataFrame(detailed_data).to_csv("detailed_fairness.csv", index=False)
    
if __name__ == "__main__":
    run_evaluation()
    
