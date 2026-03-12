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

def run_evaluation():
    summary_data = []
    detailed_data = []
    
    # Read from the new predictions directory
    pred_files = glob.glob("results/predictions/**/*.jsonl", recursive=True)
    if not pred_files:
        pred_files = glob.glob("results/predictions/*.jsonl")
    
    for file in pred_files:
        df = load_predictions(file)
        
        # performance metrics
        macro_f1 = f1_score(df['label_true'], df['label_pred'], average='macro')
        accuracy = accuracy_score(df['label_true'], df['label_pred'])
        
        # fairness metrics
        fairness = compute_fairness_gaps(df)
        
        # average gaps across occupations
        dp_gaps = [v["Demographic_Parity"] for v in fairness.values()]
        eo_gaps = [v["Equal_Opportunity"] for v in fairness.values()]
        
        avg_dp = np.nanmean(dp_gaps)
        avg_eo = np.nanmean(eo_gaps)
        
        # Extract clean model name from the file path
        model_name = os.path.basename(file).replace("preds_", "").replace(".jsonl", "")
        
        summary_data.append({
            "model_name": model_name,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "dp_diff": avg_dp,
            "eo_diff": avg_eo
        })
        
        # per-occupation gaps for detailed analysis
        for occ, metrics in fairness.items():
            detailed_data.append({
                "model_name": model_name,
                "occupation": occ,
                "dp_gap": metrics["Demographic_Parity"],
                "eo_gap": metrics["Equal_Opportunity"]
            })
    
    # Save outputs to the tables directory
    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame(summary_data).to_csv("results/tables/summary_results.csv", index=False)
    pd.DataFrame(detailed_data).to_csv("results/tables/detailed_fairness.csv", index=False)
    print("Evaluation complete. Results saved to results/tables/")

if __name__ == "__main__":
    run_evaluation()
