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


def calculate_exact_eo_diff(indices, df):
    """
    Pooled bootstrap statistic for EO diff.
 
    Units of analysis:
      - M_DOMINATED occupations  → pooled into a single unit "male_dom"
      - F_DOMINATED occupations  → pooled into a single unit "female_dom"
      - BALANCED occupations     → each treated as its own unit
 
    Within each unit the one-vs-rest TPR and FPR are computed across
    all pooled rows, giving stable estimates for the minority-gender
    cells that caused BCa to break down per-occupation.
 
    Returns max(avg_tpr_gap, avg_fpr_gap) over all units.
    """
    # 1. Resample and mark correct predictions
    sample_df = df.iloc[indices].copy()
    sample_df['is_correct'] = (sample_df['label_true'] == sample_df['label_pred'])
 
    if sample_df['gender'].nunique() < 2:
        return np.nan
 
    # 2. Map every occupation to its analysis unit.
    #    Pooled groups get a shared key; balanced occupations keep their own name.
    occ_to_unit = {}
    for occ in M_DOMINATED:
        occ_to_unit[occ] = 'male_dom'
    for occ in F_DOMINATED:
        occ_to_unit[occ] = 'female_dom'
    for occ in BALANCED:
        occ_to_unit[occ] = occ
 
    sample_df['unit_true'] = sample_df['label_true'].map(occ_to_unit)
    sample_df['unit_pred'] = sample_df['label_pred'].map(occ_to_unit)
 
    # Drop any rows whose occupation isn't in our lists (shouldn't happen, but safe)
    sample_df = sample_df.dropna(subset=['unit_true', 'unit_pred'])
 
    # 3. actual_pos[unit, gender] and correct[unit, gender]
    pos_stats = (
        sample_df
        .groupby(['unit_true', 'gender'])
        .agg(actual_pos=('is_correct', 'count'),
             correct=('is_correct', 'sum'))
        .unstack('gender')
    )
    # columns: MultiIndex [(actual_pos|correct), (M|F)]
 
    # 4. predicted[unit, gender] — how many times each unit was predicted per gender
    predicted = (
        sample_df
        .groupby(['unit_pred', 'gender'])
        .size()
        .unstack('gender', fill_value=0)
        .reindex(pos_stats.index, fill_value=0)
    )
 
    # 5. Build the four components, same logic as the per-occupation version
    actual_pos = pos_stats['actual_pos'].fillna(0)
    correct    = pos_stats['correct'].fillna(0)
 
    gender_totals = sample_df['gender'].value_counts()
    actual_neg = actual_pos.rsub(gender_totals, axis='columns')
 
    false_pos = (predicted - correct).clip(lower=0)
 
    # 6. Rates — 0 denominators become NaN and are skipped by nanmean
    tpr = correct   / actual_pos.replace(0, np.nan)
    fpr = false_pos / actual_neg.replace(0, np.nan)
 
    # 7. Per-unit absolute gaps then average
    tpr_gaps = (tpr['M'] - tpr['F']).abs()
    fpr_gaps = (fpr['M'] - fpr['F']).abs()
 
    avg_tpr_gap = np.nanmean(tpr_gaps)
    avg_fpr_gap = np.nanmean(fpr_gaps)
 
    return max(avg_tpr_gap, avg_fpr_gap)
