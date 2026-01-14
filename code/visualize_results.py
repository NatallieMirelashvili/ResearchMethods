import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import ListedColormap

SUMMARY_FILE = "statistical_reports/model_performance_summary.csv"
SIG_FILE = "statistical_reports/statistical_significance_report.csv"
OUTPUT_DIR = "paper_figures"

def create_visualizations():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(SUMMARY_FILE) or not os.path.exists(SIG_FILE):
        print("Error: Input files missing.")
        return

    df_perf = pd.read_csv(SUMMARY_FILE)
    df_sig = pd.read_csv(SIG_FILE)

    # --- 1. Model Performance Chart ---
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    df_perf = df_perf.sort_values('Mean_Macro_F1', ascending=False)
    colors = sns.color_palette("muted")
    
    bars = plt.bar(df_perf['Model'], df_perf['Mean_Macro_F1'], 
                   yerr=df_perf['Std_Dev'], capsize=8, 
                   color=colors,
                   edgecolor='black', alpha=0.85)

    plt.title('Model Performance Comparison (Macro F1)', fontsize=16, pad=25, fontweight='bold')
    plt.ylabel('Mean Macro F1 Score', fontsize=13, fontweight='bold')
    plt.xlabel('Architectures & Baselines', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.1) 
    plt.xticks(rotation=15)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        error = df_perf['Std_Dev'].iloc[i]
        plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.03,
                 f'{height:.2f}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=11, color='black')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/performance_comparison.png", dpi=300)

    # --- 2. Statistical Significance Heatmap (Green/Red) ---
    models = ['vit_tiny', 'resnet50', 'efficientnet_b0']
    sig_matrix = pd.DataFrame(np.nan, index=models, columns=models)
    
    for _, row in df_sig.iterrows():
        m1, m2 = row['Comparison'].split(' vs ')
        if m1 in models and m2 in models:
            val = 1 if row['Significant'] else 0
            sig_matrix.loc[m1, m2] = val
            sig_matrix.loc[m2, m1] = val

    plt.figure(figsize=(8, 6))
    
    # Custom Map: 0 = Red (Not Significant), 1 = Green (Significant)
    custom_cmap = ListedColormap(['#ff4d4d', '#2eb82e'])
    mask = np.eye(len(models))
    
    sns.heatmap(sig_matrix, annot=True, cmap=custom_cmap, cbar=False,
                xticklabels=models, yticklabels=models,
                mask=mask, linewidths=3, linecolor='white',
                vmin=0, vmax=1,
                annot_kws={"size": 16, "weight": "bold", "color": "white"})
    
    plt.title('Pairwise Statistical Significance\n(Green = Significant, Red = Not Significant)', 
              fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Model A', fontweight='bold')
    plt.xlabel('Model B', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/significance_heatmap.png", dpi=300)
    print(f"Exported updated visualizations to: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_visualizations()