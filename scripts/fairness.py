import numpy as np
import pandas as pd

def compute_group_metrics(df, group_label):
    """
    Variables:
- df: DataFrame derived from JSONL
- group_label: "M" or "F" for the demographic group of interest

    Output:
- Computes TPR and selection rate for a specific demographic group, for each occupation.
- Implementation follows the One-vs-Rest recommendation.
    """
    results = {}
    occupations = df['label_true'].unique()
    
    # filter DataFrame for the specified demographic group
    group_df = df[df['gender'] == group_label]
    
    for occ in occupations:
        # selection rate for Demographic Parity
        selection_rate = (group_df['label_pred'] == occ).mean()
        
        # true positive rate for Equalized Odds
        actual_occ = group_df[group_df['label_true'] == occ]
        tpr = (actual_occ['label_pred'] == occ).mean() if len(actual_occ) > 0 else 0
        
        # store results for the occupation
        results[occ] = {"selection_rate": selection_rate, "tpr": tpr}
    
    return results



def compute_fairness_gaps(df):
    """
    Variables:
- df: DataFrame derived from JSONL

    Output:
- DP Difference and EO Difference per occupation.
    """
    m_metrics = compute_group_metrics(df, "M")
    f_metrics = compute_group_metrics(df, "F")
    
    gaps = {}    
    for occ in m_metrics.keys():
        dp_gap = abs(m_metrics[occ]['selection_rate'] - f_metrics[occ]['selection_rate'])
        eo_gap = abs(m_metrics[occ]['tpr'] - f_metrics[occ]['tpr'])
        
        gaps[occ] = {"Demographic_Parity": dp_gap, "Equal_Opportunity": eo_gap}
        
    return gaps

