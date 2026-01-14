import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
from sklearn.metrics import f1_score

INPUT_DIR = "results_analysis"  
OUTPUT_DIR = "statistical_reports"

def calculate_baselines(y_true):
    unique_classes = np.unique(y_true)
    n_samples = len(y_true)
    
    counts = np.bincount(y_true)
    majority_class = np.argmax(counts)
    y_pred_majority = np.full(n_samples, majority_class)
    
    y_pred_random = np.random.choice(unique_classes, size=n_samples)
    
    return {
        "Majority_F1": f1_score(y_true, y_pred_majority, average='macro', zero_division=0),
        "Random_F1": f1_score(y_true, y_pred_random, average='macro', zero_division=0)
    }

def perform_statistical_analysis(n_bootstraps=20):
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Folder '{INPUT_DIR}' not found. Did you run test.py?")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_outputs.npz")]
    
    if not files:
        print(f"❌ No .npz files found in {INPUT_DIR}")
        return

    model_scores = {}
    performance_summary = []
    sample_y_true = None

    for f in files:
        data = np.load(os.path.join(INPUT_DIR, f))
        y_true, y_pred = data['y_true'], data['y_pred']
        sample_y_true = y_true 
        model_name = f.replace("_outputs.npz", "")

        boot_f1 = []
        for i in range(n_bootstraps):
            yt_res, yp_res = resample(y_true, y_pred, random_state=i)
            boot_f1.append(f1_score(yt_res, yp_res, average='macro', zero_division=0))
        
        scores_array = np.array(boot_f1)
        model_scores[model_name] = scores_array
        performance_summary.append({
            "Model": model_name,
            "Mean_Macro_F1": np.mean(scores_array),
            "Std_Dev": np.std(scores_array)
        })

    if sample_y_true is not None:
        base = calculate_baselines(sample_y_true)
        performance_summary.append({"Model": "Baseline_Majority", "Mean_Macro_F1": base["Majority_F1"], "Std_Dev": 0.0})
        performance_summary.append({"Model": "Baseline_Random", "Mean_Macro_F1": base["Random_F1"], "Std_Dev": 0.0})

    pd.DataFrame(performance_summary).to_csv(os.path.join(OUTPUT_DIR, "model_performance_summary.csv"), index=False)

    model_names = list(model_scores.keys())
    p_values, comparisons = [], []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            _, p = wilcoxon(model_scores[m1], model_scores[m2])
            p_values.append(p)
            comparisons.append((m1, m2))

    if p_values:
        reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')
        sig_report = []
        for idx, (m1, m2) in enumerate(comparisons):
            sig_report.append({
                "Comparison": f"{m1} vs {m2}",
                "Corrected_P_Value": p_corrected[idx],
                "Significant": reject[idx]
            })
        pd.DataFrame(sig_report).to_csv(os.path.join(OUTPUT_DIR, "statistical_significance_report.csv"), index=False)

    print(f"✅ Analysis complete! Results saved in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    perform_statistical_analysis()