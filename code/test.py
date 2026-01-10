import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
import os

# Import necessary project modules
from train import SatelliteClassifier 
from build_model import BuildModel

checkpoints = [
    "checkpoints/vit_tiny-best.ckpt", 
    "checkpoints/resnet50-best.ckpt", 
    "checkpoints/efficientnet_b0-best.ckpt"
]
DATAMODULE_FILE = "datamodule.pt"
CONFIG_PATH = "configurations/models_config.yaml"

def analyze_model(checkpoint_path, datamodule_path="datamodule.pt", builder=None):
    """
    Loads a trained model, runs inference, calculates metrics, and saves results.
    Updated for 6 classes (30-min intervals) with a Legend.
    """
    print(f"\nProcessing model: {checkpoint_path}")
    
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the trained model from checkpoint
    try:
        filename = os.path.basename(checkpoint_path) 
        arch_name = filename.split('-')[0]
        backbone = builder.build(arch_name)
        
        model = SatelliteClassifier.load_from_checkpoint(checkpoint_path, model=backbone)
        model.to(device)
        model.eval() 
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        return

    # 3. Load the pre-processed DataModule
    print(f"Loading data from {datamodule_path}...")
    if not os.path.exists(datamodule_path):
        print(f"Error: {datamodule_path} not found.")
        return

    try:
        dm = torch.load(datamodule_path, weights_only=False)
    except Exception as e:
        print(f"Error loading DataModule: {e}")
        return

    try:
        test_loader = dm.test_dataloader()
        print("✅ DataModule is already set up.")
    except Exception:
        print("⚠️ DataModule needs setup. Running dm.setup()...")
        dm.setup()
        test_loader = dm.test_dataloader()

    # 4. Run Inference
    print("Running inference on Test Set...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 5. Calculate Metrics
    print("Calculating metrics...")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    model_name = os.path.basename(checkpoint_path).replace('.ckpt', '')

    print("-" * 40)
    print(f"📊 Evaluation Results: {model_name}")
    print("-" * 40)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Acc:      {balanced_acc:.4f}")
    print(f"F1 Score (Macro):  {f1:.4f}")
    print("-" * 40)

    # 6. Save Metrics to CSV
    metrics_dict = {
        "Metric": ["Accuracy", "Balanced Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"],
        "Value": [acc, balanced_acc, precision, recall, f1]
    }
    pd.DataFrame(metrics_dict).to_csv(f"metrics_{model_name}.csv", index=False)
    print(f"✅ Metrics saved to CSV")

    # --- 7. Generate and Save Confusion Matrix with Legend ---
    print("Generating Confusion Matrix...")

    class_labels = [0, 1, 2, 3, 4, 5]
    time_labels = [
        "09:00 - 09:29",
        "09:30 - 09:59",
        "10:00 - 10:29",
        "10:30 - 10:59",
        "11:00 - 11:29",
        "11:30 - 11:59"
    ]
    
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    plt.figure(figsize=(14, 8)) 
    

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=time_labels, 
                yticklabels=time_labels,
                cbar=True)
    
    plt.title(f'Confusion Matrix - {model_name}\n(Time Prediction)', fontsize=15)
    plt.xlabel('Predicted Time Interval', fontsize=12)
    plt.ylabel('True Time Interval', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    legend_text = "Class Mapping:\n" + "\n".join([f"Class {i}: {t}" for i, t in zip(class_labels, time_labels)])
    
    plt.gcf().text(0.82, 0.5, legend_text, fontsize=11, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                   verticalalignment='center')

    plt.subplots_adjust(right=0.8) 
    
    plot_filename = f"confusion_matrix_{model_name}.png"
    plt.savefig(plot_filename)
    plt.close() 
    print(f"✅ Confusion Matrix saved to '{plot_filename}'")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
    else:
        print("Initializing Model Builder...")
        builder = BuildModel(CONFIG_PATH)

        print("Starting batch analysis...")
        for ckpt in checkpoints:
            if os.path.exists(ckpt):
                analyze_model(ckpt, DATAMODULE_FILE, builder)
            else:
                print(f"⚠️ Warning: Checkpoint file not found: {ckpt}")
                
        print("\nAnalysis complete.")