import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from matplotlib.patheffects import withStroke

plt.style.use('ggplot')
sns.set_palette("Set1")

def parse_model_info(name):
    """Parse a model filename into structured metadata for plotting.

    Extracts model size (e.g. '160m', '1.4b'), masking condition
    (Masked/Unmasked), training method (Zero-shot/Few-shot/Fine-tuned/
    Baseline), and a short human-readable label.

    Returns a Series: [Size, Size_Num, Condition, Method, Label].
    """
    name_lower = name.lower()
    size_match = re.search(r'(\d+(?:\.\d+)?[mb])', name_lower)
    size_str = size_match.group(1) if size_match else 'Other'
    condition = 'Masked' if 'masked' in name_lower else 'Unmasked'
    
    if 'zeroshot' in name_lower: method = 'Zero-shot'
    elif 'fewshot' in name_lower: method = 'Few-shot'
    elif any(m in name_lower for m in ['finetuned', 'full', 'qlora', '_ft']): method = 'Fine-tuned'
    else: method = 'Baseline'
        
    short_label = f"{size_str} {method} {'(M)' if condition == 'Masked' else '(U)'}"
    if 'qlora' in name_lower: short_label = f"{size_str} QLoRA (U)"
    if 'roberta' in name_lower: short_label = "RoBERTa FT"
    if 'distilbert' in name_lower: short_label = "DistilBERT FT"
    
    size_num = 0
    if 'm' in size_str: size_num = float(size_str.replace('m', ''))
    elif 'b' in size_str: size_num = float(size_str.replace('b', '')) * 1000
    
    return pd.Series([size_str, size_num, condition, method, short_label])

def plot_pareto(df):
    """Pareto Frontier: Macro-F1 vs. Equalized Odds Gap (max of TPR/FPR gaps)."""
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.scatterplot(data=df, x="eo_diff", y="macro_f1", hue="Method", style="Condition", s=250, alpha=0.9, ax=ax)

    MANUAL_OFFSETS = {
        '1.4b QLoRA (U)': (50, 40), '1.4b Fine-tuned (M)': (-100, 10), '1.4b Fine-tuned (U)': (0, 30),
        'RoBERTa FT': (-30, -80), 'DistilBERT FT': (70, 50), '410m Fine-tuned (M)': (-80, -30),
        '410m Fine-tuned (U)': (0, 20), '160m Fine-tuned (U)': (60, 20), '160m Fine-tuned (M)': (0, 40),
        '1.4b Zero-shot (U)': (70, 30), '1.4b Zero-shot (M)': (-70, -30), '1.4b Few-shot (U)': (-80, 40),
        '1.4b Few-shot (M)': (0, 60), '410m Zero-shot (U)': (80, 0), '410m Zero-shot (M)': (-80, 0),
        '410m Few-shot (U)': (-30, 50), '410m Few-shot (M)': (60, -50), '160m Zero-shot (U)': (60, 40),
        '160m Zero-shot (M)': (50, -40), '160m Few-shot (U)': (-60, -50), '160m Few-shot (M)': (-60, 50),
    }

    for i, row in df.iterrows():
        ax.annotate(row['Label'], xy=(row['eo_diff'], row['macro_f1']), xytext=MANUAL_OFFSETS.get(row['Label'], (25, 25)),
                    textcoords='offset points', ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=1.0, alpha=0.6),
                    path_effects=[withStroke(linewidth=2, foreground="w")])

    ax.set_title("Pareto Frontier: Macro-F1 vs. Equalized Odds Gap", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Average Equalized Odds Gap (Lower = Fairer)")
    ax.set_ylabel("Macro-F1 Score (Higher = Better)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/pareto_frontier.png', dpi=300)
    plt.close()

def plot_job_bias(detailed_df):
    """Per-occupation bias comparison (TPR + FPR gaps) for Pythia 1.4B fine-tuned."""
    target_u = "pythia_1.4b_finetuned"
    target_m = "pythia_1.4b_masked_full"               
    df_job = detailed_df[detailed_df['model_name'].isin([target_u, target_m])].copy()
    df_job['Condition'] = df_job['model_name'].apply(lambda x: 'Unmasked' if x == target_u else 'Masked')
    
    if df_job.empty: return
    
    # Compute a combined EO gap for ranking (max of TPR and FPR gap)
    df_job['eo_gap'] = df_job[['eo_tpr_gap', 'eo_fpr_gap']].max(axis=1)
    
    top_jobs = df_job[df_job['Condition'] == 'Unmasked'].sort_values('eo_gap', ascending=False).head(12)
    top_names = top_jobs['occupation'].tolist()
    
    # Melt TPR/FPR gaps into a long format for grouped bars
    plot_df = df_job[df_job['occupation'].isin(top_names)].copy()
    plot_df['occupation'] = pd.Categorical(plot_df['occupation'], categories=top_names[::-1], ordered=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    for ax, gap_col, title in zip(
        axes,
        ['eo_tpr_gap', 'eo_fpr_gap'],
        ['TPR Gap (Equal Opportunity)', 'FPR Gap'],
    ):
        sns.barplot(
            data=plot_df, y='occupation', x=gap_col, hue='Condition',
            palette={'Unmasked': '#F8766D', 'Masked': '#00BFC4'}, ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Gap')
    
    fig.suptitle("Occupation-Specific Bias: Pythia 1.4B Fine-tuned", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/job_bias_comparison.png', dpi=300)
    plt.close()

def plot_scaling(df):
    """Scaling curves for performance and both fairness components."""
    scaling_df = df[df['Size_Num'] > 0].sort_values('Size_Num')
    
    # --- Performance ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scaling_df, x="Size_Num", y='macro_f1', hue="Method", style="Condition", markers=True, markersize=10)
    plt.xscale('log')
    plt.title("Performance Scaling Law", fontweight='bold')
    plt.ylabel("Macro-F1 Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/scaling_performance.png', dpi=300)
    plt.close()
    
    # --- Combined EO gap ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scaling_df, x="Size_Num", y='eo_diff', hue="Method", style="Condition", markers=True, markersize=10)
    plt.xscale('log')
    plt.title("Fairness Scaling Law (Equalized Odds Gap)", fontweight='bold')
    plt.ylabel("Average EO Gap (max of TPR/FPR)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/scaling_fairness.png', dpi=300)
    plt.close()
    
    # --- TPR gap vs FPR gap breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for ax, col, label in zip(axes, ['avg_tpr_gap', 'avg_fpr_gap'], ['TPR Gap', 'FPR Gap']):
        sns.lineplot(data=scaling_df, x="Size_Num", y=col, hue="Method", style="Condition", markers=True, markersize=10, ax=ax)
        ax.set_xscale('log')
        ax.set_title(f"Scaling: {label}", fontweight='bold')
        ax.set_ylabel(label)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/scaling_fairness_breakdown.png', dpi=300)
    plt.close()

def plot_all_correlations(summary_df, detailed_df, gender_df):
    all_models = summary_df['model_name'].unique()
    pairs = []
    
    for u in [m for m in all_models if 'masked' not in m.lower()]:
        m_cand = next((m for m in all_models if 'masked' in m.lower() and 
                       os.path.basename(m).replace('masked_', '').replace('_masked', '') == os.path.basename(u)), None)
        if m_cand: 
            pairs.append((u, m_cand))

    for u_model, m_model in pairs:
        clean_id = os.path.basename(u_model).replace('pythia_', '').replace('_full', '_FT').replace('_zeroshot', '_ZS').replace('_fewshot', '_FS')
        
        df_u = detailed_df[detailed_df['model_name'] == u_model].merge(gender_df, on='occupation')
        df_m = detailed_df[detailed_df['model_name'] == m_model].merge(gender_df, on='occupation')
        
        if df_u.empty or df_m.empty: continue

        # Compute combined EO gap for correlation plots
        df_u['eo_gap'] = df_u[['eo_tpr_gap', 'eo_fpr_gap']].max(axis=1)
        df_m['eo_gap'] = df_m[['eo_tpr_gap', 'eo_fpr_gap']].max(axis=1)

        plt.figure(figsize=(10, 7))
        sns.regplot(data=df_u, x='F_frac', y='eo_gap', label='Unmasked', scatter_kws={'s':80}, line_kws={'linestyle':'--'})
        sns.regplot(data=df_m, x='F_frac', y='eo_gap', label='Masked', scatter_kws={'s':80})
        plt.title(f"Bias Amplification: Gender Ratio vs. EO Gap ({clean_id})", fontweight='bold')
        plt.xlabel("Percentage of Women in Profession (Training Data)")
        plt.ylabel("Equalized Odds Gap")
        plt.legend()
        plt.savefig(f'results/figures/Correlation Plots/Amplification/{clean_id}.png', dpi=300)
        plt.close()

        df_delta = df_u.copy()
        df_delta['eo_improvement'] = df_u['eo_gap'] - df_m['eo_gap']
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df_delta, x='F_frac', y='eo_improvement', s=100, color='purple')
        
        for i, row in df_delta.iterrows():
            if abs(row['eo_improvement']) > 0.015 or row['F_frac'] > 0.8 or row['F_frac'] < 0.2:
                plt.text(row['F_frac']+0.005, row['eo_improvement'], row['occupation'], fontsize=9)
                
        plt.axhline(0, color='black', lw=1, ls='--')
        plt.title(f"Mitigation Success: {clean_id}", fontweight='bold')
        plt.xlabel("Percentage of Women in Profession")
        plt.ylabel("Reduction in Bias (Unmasked EO − Masked EO)")
        plt.tight_layout()
        plt.savefig(f'results/figures/Correlation Plots/Delta/{clean_id}.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/figures/Correlation Plots/Amplification", exist_ok=True)
    os.makedirs("results/figures/Correlation Plots/Delta", exist_ok=True)
    
    # Load from updated paths
    summary_df = pd.read_csv('results/tables/summary_results.csv')
    detailed_df = pd.read_csv('results/tables/detailed_fairness.csv')
    gender_df = pd.read_csv('data/stats/occupation_gender_breakdown.csv')
    
    summary_df[['Size', 'Size_Num', 'Condition', 'Method', 'Label']] = summary_df['model_name'].apply(parse_model_info)

    print("Generating figures into results/figures/...")
    plot_pareto(summary_df)
    plot_job_bias(detailed_df)
    plot_scaling(summary_df)
    plot_all_correlations(summary_df, detailed_df, gender_df)
    print("All images generated successfully!")
