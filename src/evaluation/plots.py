import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from matplotlib.patheffects import withStroke

# Set the style and color palette for all matplotlib/seaborn plots
plt.style.use('ggplot')
sns.set_palette("Set1")

def parse_model_info(name):
    """Parse a model filename into structured metadata for plotting.

    This function takes a raw model string (e.g., 'pythia_1.4b_finetuned_masked')
    and extracts key characteristics needed for visualization and grouping:
    1. Model size (e.g., '160m', '1.4b')
    2. Masking condition (whether names/pronouns were masked during training)
    3. Training method (Zero-shot, Few-shot, Fine-tuned, or Baseline)
    4. A concise human-readable label for plot legends.

    Args:
        name (str): The raw model filename or identifier.

    Returns:
        pd.Series: A series containing [Size, Size_Num, Condition, Method, Label].
                   Size_Num is the numerical representation of parameters in millions.
    """
    name_lower = name.lower()
    
    # Extract model size using regex (looking for numbers followed by 'm' or 'b')
    size_match = re.search(r'(\d+(?:\.\d+)?[mb])', name_lower)
    size_str = size_match.group(1) if size_match else 'Other'
    
    # Determine if the model was trained on dataset with masking applied
    condition = 'Masked' if 'masked' in name_lower else 'Unmasked'
    
    # Determine the evaluation/training methodology used
    if 'zeroshot' in name_lower: method = 'Zero-shot'
    elif 'fewshot' in name_lower: method = 'Few-shot'
    elif any(m in name_lower for m in ['finetuned', 'full', 'qlora', '_ft']): method = 'Fine-tuned'
    else: method = 'Baseline'
        
    # Construct a concise label for plot annotations
    short_label = f"{size_str} {method} {'(M)' if condition == 'Masked' else '(U)'}"
    
    # Override generic labels for specific known baseline/fine-tuned architectures
    if 'qlora' in name_lower: short_label = f"{size_str} QLoRA (U)"
    if 'roberta' in name_lower: short_label = "RoBERTa FT"
    if 'distilbert' in name_lower: short_label = "DistilBERT FT"
    
    # Convert size string to a numerical value (in millions) for scaling plots
    size_num = 0
    if 'm' in size_str: size_num = float(size_str.replace('m', ''))
    elif 'b' in size_str: size_num = float(size_str.replace('b', '')) * 1000
    
    return pd.Series([size_str, size_num, condition, method, short_label])

def plot_pareto(df):
    """Plot the Pareto Frontier for Macro-F1 vs. Equalized Odds Gap.

    Visualizes the trade-off between model performance (Macro-F1) and fairness 
    (Equalized Odds Gap). Ideal models fall in the top-left quadrant (high F1, low gap).
    
    Args:
        df (pd.DataFrame): DataFrame containing summary results with 
                           'eo_diff' (fairness gap), 'macro_f1' (performance score),
                           and parsed model metadata.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Scatter plot mapping performance and fairness, grouped by Method and Condition
    sns.scatterplot(data=df, x="eo_diff", y="macro_f1", hue="Method", style="Condition", s=250, alpha=0.9, ax=ax)

    # Dictionary mapping model labels to specific (x, y) offset coordinates 
    # for text annotations to prevent label overlapping on the plot.
    MANUAL_OFFSETS = {
        '1.4b QLoRA (U)': (50, 40), '1.4b Fine-tuned (M)': (-100, 10), '1.4b Fine-tuned (U)': (0, 30),
        'RoBERTa FT': (-30, -80), 'DistilBERT FT': (70, 50), '410m Fine-tuned (M)': (-80, -30),
        '410m Fine-tuned (U)': (0, 20), '160m Fine-tuned (U)': (60, 20), '160m Fine-tuned (M)': (0, 40),
        '1.4b Zero-shot (U)': (70, 30), '1.4b Zero-shot (M)': (-70, -30), '1.4b Few-shot (U)': (-80, 40),
        '1.4b Few-shot (M)': (0, 60), '410m Zero-shot (U)': (80, 0), '410m Zero-shot (M)': (-80, 0),
        '410m Few-shot (U)': (-30, 50), '410m Few-shot (M)': (60, -50), '160m Zero-shot (U)': (60, 40),
        '160m Zero-shot (M)': (50, -40), '160m Few-shot (U)': (-60, -50), '160m Few-shot (M)': (-60, 50),
    }

    # Annotate each point with its short label, using arrows pointing to the data point
    for i, row in df.iterrows():
        ax.annotate(row['Label'], xy=(row['eo_diff'], row['macro_f1']), 
                    xytext=MANUAL_OFFSETS.get(row['Label'], (25, 25)),
                    textcoords='offset points', ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=1.0, alpha=0.6),
                    path_effects=[withStroke(linewidth=2, foreground="w")])

    # Configure axes, title, and legend
    ax.set_title("Pareto Frontier: Macro-F1 vs. Equalized Odds Gap", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Average Equalized Odds Gap (Lower = Fairer)")
    ax.set_ylabel("Macro-F1 Score (Higher = Better)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/figures/pareto_frontier.png', dpi=300)
    plt.close()

def plot_job_bias(detailed_df):
    """Plot per-occupation bias comparison for a specific pair of models.

    Focuses on the Pythia 1.4B fine-tuned models (masked vs unmasked) to 
    visualize which specific occupations suffer the highest True Positive Rate (TPR) 
    and False Positive Rate (FPR) gaps by gender.
    
    Args:
        detailed_df (pd.DataFrame): DataFrame containing per-occupation fairness metrics.
    """
    # Define the models to compare (unmasked baseline vs masked intervention)
    target_u = "pythia_1.4b_finetuned"
    target_m = "pythia_1.4b_masked_full"               
    df_job = detailed_df[detailed_df['model_name'].isin([target_u, target_m])].copy()
    
    # Label conditions based on model name
    df_job['Condition'] = df_job['model_name'].apply(lambda x: 'Unmasked' if x == target_u else 'Masked')
    
    if df_job.empty: return
    
    # Compute a combined Equalized Odds (EO) gap for ranking occupations.
    # EO gap is defined here as the maximum of either the TPR gap or the FPR gap.
    df_job['eo_gap'] = df_job[['eo_tpr_gap', 'eo_fpr_gap']].max(axis=1)
    
    # Select the top 12 occupations with the highest bias in the unmasked model
    top_jobs = df_job[df_job['Condition'] == 'Unmasked'].sort_values('eo_gap', ascending=False).head(12)
    top_names = top_jobs['occupation'].tolist()
    
    # Filter and format for plotting
    plot_df = df_job[df_job['occupation'].isin(top_names)].copy()
    plot_df['occupation'] = pd.Categorical(plot_df['occupation'], categories=top_names[::-1], ordered=True)
    
    # Create side-by-side bar charts for TPR gap and FPR gap
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
    """Plot scaling laws: model size vs. performance and fairness.

    Generates three separate plots to visualize how scaling model parameters:
    1. Impacts Macro-F1 (Performance)
    2. Impacts the combined EO gap (Fairness)
    3. Differentially impacts TPR and FPR gaps separately.
    
    Args:
        df (pd.DataFrame): DataFrame containing summary metrics and parsed size data.
    """
    # Filter valid numerical sizes and sort ascending
    scaling_df = df[df['Size_Num'] > 0].sort_values('Size_Num')
    
    # --- Performance Scaling Plot ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scaling_df, x="Size_Num", y='macro_f1', hue="Method", style="Condition", markers=True, markersize=10)
    plt.xscale('log')  # Log scale since parameter counts vary exponentially
    plt.title("Performance Scaling Law", fontweight='bold')
    plt.ylabel("Macro-F1 Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/scaling_performance.png', dpi=300)
    plt.close()
    
    # --- Fairness Scaling Plot (Combined EO gap) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=scaling_df, x="Size_Num", y='eo_diff', hue="Method", style="Condition", markers=True, markersize=10)
    plt.xscale('log')
    plt.title("Fairness Scaling Law (Equalized Odds Gap)", fontweight='bold')
    plt.ylabel("Average EO Gap (max of TPR/FPR)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/scaling_fairness.png', dpi=300)
    plt.close()
    
    # --- Breakdown: TPR gap and FPR gap scaling ---
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
    """Plot correlation between occupation gender ratio and model bias.

    Iterates through pairs of matched models (Unmasked vs. Masked versions)
    to generate two types of scatter plots:
    1. Amplification Plot: How training data gender skew amplifies prediction gaps.
    2. Delta Plot: The reduction in bias achieved by the masking intervention relative
       to the gender fraction of the occupation.
       
    Args:
        summary_df (pd.DataFrame): Unique models to extract unmasked/masked pairs.
        detailed_df (pd.DataFrame): Per-occupation stats containing gap metrics.
        gender_df (pd.DataFrame): Demographics containing the fraction of females ('F_frac') per job.
    """
    all_models = summary_df['model_name'].unique()
    pairs = []
    
    # Automatically identify pairs of models: (Unmasked, Masked equivalent)
    for u in [m for m in all_models if 'masked' not in m.lower()]:
        # Search for a matching counterpart that has 'masked' in the name
        m_cand = next((m for m in all_models if 'masked' in m.lower() and 
                       os.path.basename(m).replace('masked_', '').replace('_masked', '') == os.path.basename(u)), None)
        if m_cand: 
            pairs.append((u, m_cand))

    for u_model, m_model in pairs:
        # Create a clean ID for titles and filenames
        clean_id = os.path.basename(u_model).replace('pythia_', '').replace('_full', '_FT').replace('_zeroshot', '_ZS').replace('_fewshot', '_FS')
        
        # Merge task performance data with real-world gender statistics
        df_u = detailed_df[detailed_df['model_name'] == u_model].merge(gender_df, on='occupation')
        df_m = detailed_df[detailed_df['model_name'] == m_model].merge(gender_df, on='occupation')
        
        if df_u.empty or df_m.empty: continue

        # Compute combined EO gap (max of TPR and FPR gap) for both models
        df_u['eo_gap'] = df_u[['eo_tpr_gap', 'eo_fpr_gap']].max(axis=1)
        df_m['eo_gap'] = df_m[['eo_tpr_gap', 'eo_fpr_gap']].max(axis=1)

        # 1. Bias Amplification Scatter Plot
        plt.figure(figsize=(10, 7))
        sns.regplot(data=df_u, x='F_frac', y='eo_gap', label='Unmasked', scatter_kws={'s':80}, line_kws={'linestyle':'--'})
        sns.regplot(data=df_m, x='F_frac', y='eo_gap', label='Masked', scatter_kws={'s':80})
        plt.title(f"Bias Amplification: Gender Ratio vs. EO Gap ({clean_id})", fontweight='bold')
        plt.xlabel("Percentage of Women in Profession (Training Data)")
        plt.ylabel("Equalized Odds Gap")
        plt.legend()
        plt.savefig(f'results/figures/Correlation Plots/Amplification/{clean_id}.png', dpi=300)
        plt.close()

        # 2. Mitigation Success (Delta) Scatter Plot
        df_delta = df_u.copy()
        # Positive values mean unmasked was historically more biased than masked (an improvement)
        df_delta['eo_improvement'] = df_u['eo_gap'] - df_m['eo_gap']
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df_delta, x='F_frac', y='eo_improvement', s=100, color='purple')
        
        # Annotate significant data points for better readability
        # Shows names for large improvements/degradations or heavily skewed jobs
        for i, row in df_delta.iterrows():
            if abs(row['eo_improvement']) > 0.015 or row['F_frac'] > 0.8 or row['F_frac'] < 0.2:
                plt.text(row['F_frac']+0.005, row['eo_improvement'], row['occupation'], fontsize=9)
                
        plt.axhline(0, color='black', lw=1, ls='--')  # Baseline of zero improvement
        plt.title(f"Mitigation Success: {clean_id}", fontweight='bold')
        plt.xlabel("Percentage of Women in Profession")
        plt.ylabel("Reduction in Bias (Unmasked EO − Masked EO)")
        plt.tight_layout()
        plt.savefig(f'results/figures/Correlation Plots/Delta/{clean_id}.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    # Ensure all output directories exist prior to saving plots
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/figures/Correlation Plots/Amplification", exist_ok=True)
    os.makedirs("results/figures/Correlation Plots/Delta", exist_ok=True)
    
    # Load required dataframes from tables generated during evaluation
    summary_df = pd.read_csv('results/tables/summary_results.csv')
    detailed_df = pd.read_csv('results/tables/detailed_fairness.csv')
    gender_df = pd.read_csv('data/stats/occupation_gender_breakdown.csv')
    
    # Apply parsing logic to populate model parameters (Size, Condition, etc.)
    summary_df[['Size', 'Size_Num', 'Condition', 'Method', 'Label']] = summary_df['model_name'].apply(parse_model_info)

    # Generate all plots sequentially
    print("Generating figures into results/figures/...")
    plot_pareto(summary_df)
    plot_job_bias(detailed_df)
    plot_scaling(summary_df)
    plot_all_correlations(summary_df, detailed_df, gender_df)
    print("All images generated successfully!")
