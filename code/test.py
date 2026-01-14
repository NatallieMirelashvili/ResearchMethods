import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
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
RESULTS_DIR = "results_analysis"

def analyze_model(checkpoint_path, datamodule_path, builder):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_name = os.path.basename(checkpoint_path).split('-')[0]
    print(f"\n--- Analyzing: {model_name} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        backbone = builder.build(model_name)
        model = SatelliteClassifier.load_from_checkpoint(checkpoint_path, model=backbone)
        model.to(device).eval()
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return


    if not os.path.exists(datamodule_path):
        print(f"❌ Data file not found at: {datamodule_path}")
        return
    dm = torch.load(datamodule_path, weights_only=False)
    dm.setup()
    test_loader = dm.test_dataloader()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            preds = torch.argmax(model(x.to(device)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average='macro')
    }
    pd.DataFrame([metrics]).to_csv(f"{RESULTS_DIR}/metrics_{model_name}.csv", index=False)

# --- Confusion Matrix Design (Section 4.1) ---
    plt.figure(figsize=(10, 8))
    
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(
        cm, 
        annot=True,     
        fmt='d',         
        cmap='Blues',    
        cbar=True,      
        xticklabels=[f"Bin {i}" for i in range(cm.shape[1])], 
        yticklabels=[f"Bin {i}" for i in range(cm.shape[0])]  
    )
    
    plt.title(f'Confusion Matrix - {model_name}\n(Spectral-Temporal Classification)', fontsize=14, pad=20)
    plt.ylabel('Actual Acquisition Time (Ground Truth)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Acquisition Time (Model Output)', fontsize=12, fontweight='bold')

    plt.annotate('Diagonal elements represent correct predictions.\nOff-diagonal elements show systematic confusions.', 
                 xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10, style='italic')

    plt.tight_layout() 
    plt.savefig(f"{RESULTS_DIR}/cm_{model_name}.png", dpi=300) 
    plt.close()

    np.savez(f"{RESULTS_DIR}/{model_name}_outputs.npz", y_true=y_true, y_pred=y_pred)
    print(f"✅ Finished {model_name}")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: Config file not found at {CONFIG_PATH}")
    else:
        builder = BuildModel(CONFIG_PATH)
        for ckpt in checkpoints:
            if os.path.exists(ckpt):
                analyze_model(ckpt, DATAMODULE_FILE, builder)
            else:
                print(f"⚠️ Checkpoint missing: {ckpt}")