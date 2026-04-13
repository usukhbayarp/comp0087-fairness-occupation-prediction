import numpy as np
import pandas as pd

def compute_group_metrics(df, group_label):
    """
    Variables:
    - df: DataFrame derived from JSONL
    - group_label: "M" or "F" for the demographic group of interest

    Output:
    - Computes selection rate, TPR, and FPR for a specific demographic group,
      for each occupation using One-vs-Rest.
    """
    results = {}
    occupations = df['label_true'].unique()
    
    # filter DataFrame for the specified demographic group
    group_df = df[df['gender'] == group_label]
    
    for occ in occupations:
        # selection rate for Demographic Parity
        selection_rate = (group_df['label_pred'] == occ).mean() if len(group_df) > 0 else np.nan
        
        # --- One-vs-Rest for this occupation ---
        # Positives: samples whose true label IS this occupation
        actual_pos = group_df[group_df['label_true'] == occ]
        # Negatives: samples whose true label is NOT this occupation
        actual_neg = group_df[group_df['label_true'] != occ]
        
        # TPR = P(pred == occ | true == occ)  (sensitivity / recall)
        tpr = (actual_pos['label_pred'] == occ).mean() if len(actual_pos) > 0 else np.nan
        
        # FPR = P(pred == occ | true != occ)  (false alarm rate)
        fpr = (actual_neg['label_pred'] == occ).mean() if len(actual_neg) > 0 else np.nan
        
        # store results for the occupation
        results[occ] = {"selection_rate": selection_rate, "tpr": tpr, "fpr": fpr}
    
    return results

def compute_fairness_gaps(df):
    """
    Variables:
    - df: DataFrame derived from JSONL

    Output:
    - Per-occupation fairness gaps:
        Demographic_Parity  – |selection_rate_M − selection_rate_F|
        EO_TPR_Gap          – |TPR_M − TPR_F|  (equal-opportunity component)
        EO_FPR_Gap          – |FPR_M − FPR_F|  (false-positive component)
    """
    m_metrics = compute_group_metrics(df, "M")
    f_metrics = compute_group_metrics(df, "F")
    
    gaps = {}    
    for occ in m_metrics.keys():
        dp_gap = abs(m_metrics[occ]['selection_rate'] - f_metrics[occ]['selection_rate'])
        tpr_gap = abs(m_metrics[occ]['tpr'] - f_metrics[occ]['tpr'])
        fpr_gap = abs(m_metrics[occ]['fpr'] - f_metrics[occ]['fpr'])
        
        gaps[occ] = {
            "Demographic_Parity": dp_gap,
            "EO_TPR_Gap": tpr_gap,
            "EO_FPR_Gap": fpr_gap,
        }
        
    return gaps


# -- BOOTSTRAPPING ------------------------

# Group the minority occupations so they get sampled in the bootstrap
M_DOMINATED = ["surgeon", "architect", "software_engineer", "composer", "comedian"]
F_DOMINATED = ["nurse", "model", "dietitian"]
BALANCED = ["professor", "physician", "attorney", "photographer", "journalist", 
            "psychologist", "teacher", "dentist", "painter", "filmmaker", "poet", "accountant"]

def calculate_pooled_eo(indices, df):
    # 1. Create the resampled slice
    sample_df = df.iloc[indices]
    category_gaps = []
    
    # 2. Calculate gaps for our 3 hard-coded pools
    for occupations in [M_DOMINATED, F_DOMINATED, BALANCED]:
        group_df = sample_df[sample_df['label_true'].isin(occupations)]
        if group_df.empty: continue
        
        metrics = {}
        for g in ['M', 'F']:
            sub = group_df[group_df['gender'] == g]
            if len(sub) == 0: continue
            
            tpr = (sub['label_true'] == sub['label_pred']).sum() / len(sub)
            # Simplified FPR: Predicted an occupation in this group incorrectly
            fp = (sub['label_pred'].isin(occupations)) & (sub['label_pred'] != sub['label_true'])
            fpr = fp.sum() / len(sub)
            
            metrics[g] = {'tpr': tpr, 'fpr': fpr}

        if 'M' in metrics and 'F' in metrics:
            tpr_gap = abs(metrics['M']['tpr'] - metrics['F']['tpr'])
            fpr_gap = abs(metrics['M']['fpr'] - metrics['F']['fpr'])
            category_gaps.append(max(tpr_gap, fpr_gap))
    
    return np.mean(category_gaps)

    

