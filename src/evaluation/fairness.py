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
    """
    This function is WRONG! E.g. one v all approach even in the pooling
    """
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
            # INCORRECT FPR CALCULATION
            fp = (sub['label_pred'].isin(occupations)) & (sub['label_pred'] != sub['label_true'])
            fpr = fp.sum() / len(sub)
            
            metrics[g] = {'tpr': tpr, 'fpr': fpr}

        if 'M' in metrics and 'F' in metrics:
            tpr_gap = abs(metrics['M']['tpr'] - metrics['F']['tpr'])
            fpr_gap = abs(metrics['M']['fpr'] - metrics['F']['fpr'])
            category_gaps.append(max(tpr_gap, fpr_gap))
    
    return np.mean(category_gaps)

# no pooling
""""
def calculate_exact_eo_diff(indices, df):
    # 1. Resample the data
    sample_df = df.iloc[indices]
    
    # 2. Compute the per-occupation fairness gaps exactly as evaluate.py does
    fairness = compute_fairness_gaps(sample_df)
    
    # 3. Extract and average the gaps
    tpr_gaps = [v["EO_TPR_Gap"] for v in fairness.values()]
    fpr_gaps = [v["EO_FPR_Gap"] for v in fairness.values()]
    
    avg_tpr_gap = np.nanmean(tpr_gaps)
    avg_fpr_gap = np.nanmean(fpr_gaps)
    
    # 4. Return the standard EO difference
    return max(avg_tpr_gap, avg_fpr_gap)
"""
    


def calculate_exact_eo_diff(indices, df):
    # 1. Create a copy to avoid warnings and enable fast pre-calculations
    sample_df = df.iloc[indices].copy()
    sample_df['is_correct'] = (sample_df['label_true'] == sample_df['label_pred'])
    
    # 2. Pre-calculate gender totals (for FPR denominator)
    gender_totals = sample_df['gender'].value_counts()
    if len(gender_totals) < 2:
        return np.nan 

    # 3. TPR calculation: Aggregating is now much faster without the lambda
    pos_stats = sample_df.groupby(['label_true', 'gender']).agg(
        actual_pos=('id', 'count'), # Counts how many people are actually in that category.
        correct=('is_correct', 'sum') # Counts how many people were correctly predicted.
    ).unstack('gender') # Reshapes the table to have genders as columns, rows are 

    # 4. FPR calculation
    # how many times the model predicted a specific category for each gender
    preds_stats = sample_df.groupby(['label_pred', 'gender']).size().unstack('gender') 
    
    # 5. Compute Rates
    # Aligning Corrected/Predicted/Positives (DF vs DF alignment is automatic)
    tpr = pos_stats['correct'] / pos_stats['actual_pos'] # True Positive Rate: Correct predictions / Actual positives
    
    false_positives = preds_stats.reindex(pos_stats.index).fillna(0) - pos_stats['correct'].fillna(0)
    
    # FIX: Use rsub to correctly align gender_totals (Series) to columns of actual_pos (DF)
    actual_negatives = pos_stats['actual_pos'].fillna(0).rsub(gender_totals, axis=1)
    
    fpr = false_positives / actual_negatives

    # 6. Calculate Gaps (|M - F|)
    tpr_gaps = (tpr['M'] - tpr['F']).abs()
    fpr_gaps = (fpr['M'] - fpr['F']).abs()

    # 7. Final Averages (ignores NaNs for missing jobs/genders)
    avg_tpr_gap = np.nanmean(tpr_gaps)
    avg_fpr_gap = np.nanmean(fpr_gaps)

    return max(avg_tpr_gap, avg_fpr_gap)