import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_detailed_plots():
    summary_df = pd.read_csv("summary_results.csv")
    detailed_df = pd.read_csv("detailed_fairness.csv")

    # 1. Pareto Frontier (Performance vs Fairness Trade-off)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=summary_df, x="avg_eo_gap", y="macro_f1", hue="model_name", s=100)
    plt.title("Pareto Frontier: Macro-F1 vs. Average Equalized Odds Gap")
    plt.xlabel("Average EO Gap (Lower = Fairer)")
    plt.ylabel("Macro-F1 (Higher = Better)")
    plt.grid(True)
    plt.savefig("pareto_frontier.png")

    # 2. Heatmap: Which occupations are most biased per model?
    # Pivot the detailed data for a heatmap
    heatmap_data = detailed_df.pivot(index="occupation", columns="model", values="eo_gap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd")
    plt.title("EO Gap Per Occupation Across Models")
    plt.savefig("occupation_bias_heatmap.png")

if __name__ == "__main__":
    generate_detailed_plots()