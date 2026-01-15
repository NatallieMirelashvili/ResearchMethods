import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from preprocess import run_preprocessing_pipeline, SatelliteDataModule, BigEarthNetDataset
from build_model import BuildModel    
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall

import json
from datetime import datetime
import pandas as pd

USER_NAME = "avivyuv"

pkl_relative_path = 'bigearthnet_df.pkl'
PKL_CHUNKS_DIR = "out_chunks"
DATAMODULE_PATH = f"/home/{USER_NAME}/bigearthnet_v2/ResearchMethods/datamodule.pt"
config_path = "configurations/models_config.yaml"

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def list_chunk_pkls(chunks_dir: str):
    # helper to list all chunk .pkl files
    return [
        os.path.join(chunks_dir, f)
        for f in sorted(os.listdir(chunks_dir))
        if f.startswith("chunk_") and f.endswith(".pkl")
    ]

def try_add_head_dropout(model: nn.Module, p: float = 0.2) -> bool:
    """
    Best-effort: wrap a common classification head (fc/classifier/head) with Dropout.
    Returns True if a head was modified, False otherwise.
    """
    candidate_attrs = ["classifier", "fc", "head"]
    for attr in candidate_attrs:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Linear):
                setattr(model, attr, nn.Sequential(nn.Dropout(p), head))
                return True
            if isinstance(head, nn.Sequential):
                # Prepend dropout
                setattr(model, attr, nn.Sequential(nn.Dropout(p), *list(head.children())))
                return True
    return False


# class SatelliteClassifier(pl.LightningModule):
    # def __init__(self, model, lr=1e-4):
    #     super().__init__()
    #     self.model = model
    #     self.lr = lr
    #     self.criterion = nn.CrossEntropyLoss()
    #     self.save_hyperparameters(ignore=['model']) 
    #     self.test_preds = []
    #     self.test_labels = []

    # def forward(self, x):
    #     return self.model(x)

    # def training_step(self, batch, batch_idx):

    #     # x - pictures
    #     # y - labels
    #     x, y = batch
    #     # what is logits?
    #     logits = self(x) 
    #     loss = self.criterion(logits, y) 
        
    #     # calculate accuracy
    #     preds = torch.argmax(logits, dim=1)
    #     acc = (preds == y).float().mean()
        
    #     # Log to progress bar
    #     self.log('train_loss', loss, prog_bar=True)
    #     self.log('train_acc', acc, prog_bar=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = self.criterion(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     acc = (preds == y).float().mean()
        
    #     self.log('val_loss', loss, prog_bar=True)
    #     self.log('val_acc', acc, prog_bar=True)
    #     return loss
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     preds = torch.argmax(logits, dim=1)

    #     self.test_preds.append(preds.cpu())
    #     self.test_labels.append(y.cpu())

    #     acc = (preds == y).float().mean()
    #     self.log('test_acc', acc)


    # def configure_optimizers(self):
    #     return optim.AdamW(self.parameters(), lr=self.lr)

class SatelliteClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        lr: float = 1e-4,
        num_classes: int = 6,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        # CrossEntropyLoss supports label_smoothing in modern PyTorch.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.save_hyperparameters(ignore=['model'])

        # --- Metrics (epoch-level) ---
        # "acc" here is standard multiclass accuracy.
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        # Balanced accuracy ≈ macro recall for multiclass.
        self.val_bal_acc = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_bal_acc = MulticlassRecall(num_classes=num_classes, average="macro")

        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update + log metrics
        self.train_acc.update(preds, y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)
        self.val_bal_acc.update(preds, y)
        self.val_f1_macro.update(preds, y)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_bal_acc', self.val_bal_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1_macro', self.val_f1_macro, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_acc.update(preds, y)
        self.test_bal_acc.update(preds, y)
        self.test_f1_macro.update(preds, y)

        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_bal_acc', self.test_bal_acc, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.test_f1_macro, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    # 1. Load DataModule
    print("--- Setting up Data ---")
    if os.path.exists(DATAMODULE_PATH):
        print(f"♻️ Loading existing DataModule from {DATAMODULE_PATH}...")
        dm = torch.load(DATAMODULE_PATH, weights_only=False)
    else:
        print("--- Setting up Data (Full Pipeline) ---")
        dm = run_preprocessing_pipeline(pkl_path=pkl_relative_path, batch_size=32)
        # chunk
        # chunk_paths = list_chunk_pkls(PKL_CHUNKS_DIR)
        # dm = run_preprocessing_pipeline(pkl_paths=chunk_paths, batch_size=32)
        if dm is not None:
            torch.save(dm, DATAMODULE_PATH)
            print("✅ DataModule saved to disk.")
        else:
            print("❌ DataModule creation failed. Exiting.")
            return


    # 2. Build Model Builder
    print("\n--- Setting up Model Builder ---")

    if not Path(config_path).exists():
        print(f"Error: {config_path} not found. Please create it.")
        return
        
    builder = BuildModel(config_path)

    results = []
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3. Train Multiple Models
    models_to_train = ["resnet50", "efficientnet_b0", "vit_tiny"] 
    
    for model_name in models_to_train:
        print(f"\n" + "="*50)
        print(f"🚀 Starting training for: {model_name}")
        print("="*50)

        # 1. Build Raw Model    
        try:
            raw_model = builder.build(model_name)
        except Exception as e:
            print(f"Failed to build {model_name}: {e}")
            continue
        
        # --- Unambiguous identity print (addresses “order confusion”) ---
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"Built model type: {type(raw_model).__name__} | params: {n_params/1e6:.2f}M")

        # --- Per-model hyperparams / regularization ---
        lr = 3e-4
        weight_decay = 0.0
        label_smoothing = 0.0

        if model_name == "efficientnet_b0":
            weight_decay = 1e-2
            label_smoothing = 0.1
            modified = try_add_head_dropout(raw_model, p=0.2)
            print(f"Regularization for {model_name}: wd={weight_decay}, label_smoothing={label_smoothing}, head_dropout={modified}")

        # 2. Wrap in Lightning Module
        system = SatelliteClassifier(model=raw_model, lr=lr, weight_decay=weight_decay, label_smoothing=label_smoothing, num_classes=6)

        model_ckpt_dir = f'checkpoints/{model_name}/'
        Path(model_ckpt_dir).mkdir(parents=True, exist_ok=True)

        # 3. Setup Checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor='val_bal_acc',
            mode='max',
            dirpath=model_ckpt_dir,
            filename=f'{model_name}-best',
            save_top_k=1,
            save_last=True,
            verbose=True
        )

        # 4. Setup Trainer
        trainer = pl.Trainer(
            max_epochs=5,          
            accelerator="auto",   
            devices=1,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10
        )

        # 5. Training
        last_ckpt = Path(model_ckpt_dir) / "last.ckpt"
        resume_path = str(last_ckpt) if last_ckpt.exists() else None
        
        if resume_path:
            print(f"🔄 Resuming {model_name} from last checkpoint...")

        trainer.fit(system, datamodule=dm, ckpt_path=resume_path)
        
        # --- Test best checkpoint ---
        print(f"\n🧪 Testing best checkpoint for: {model_name}")
        test_results = trainer.test(system, datamodule=dm, ckpt_path="best")
        r = test_results[0] if test_results and isinstance(test_results, list) else {}

        # --- Collect summary ---
        entry = {
            "run_tag": run_tag,
            "model_name": model_name,
            "built_model_type": type(raw_model).__name__,
            "n_params": int(sum(p.numel() for p in raw_model.parameters())),
            "best_model_path": checkpoint_callback.best_model_path,
            "best_val_bal_acc": _safe_float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None,
            "test_acc": _safe_float(r.get("test_acc")),
            "test_bal_acc": _safe_float(r.get("test_bal_acc")),
            "test_f1_macro": _safe_float(r.get("test_f1_macro")),
        }
        results.append(entry)

        print(
            f"✅ {model_name} | best_val_bal_acc={entry['best_val_bal_acc'] if entry['best_val_bal_acc'] is not None else float('nan'):.4f} | "
            f"test_acc={entry['test_acc'] if entry['test_acc'] is not None else float('nan'):.4f} | "
            f"test_bal_acc={entry['test_bal_acc'] if entry['test_bal_acc'] is not None else float('nan'):.4f} | "
            f"test_f1_macro={entry['test_f1_macro'] if entry['test_f1_macro'] is not None else float('nan'):.4f}"
        )

    # --- Final summary table ---
    df = pd.DataFrame(results)

    # Order columns for readability
    col_order = [
        "run_tag", "model_name", "built_model_type", "n_params",
        "best_val_bal_acc", "test_acc", "test_bal_acc", "test_f1_macro",
        "best_model_path"
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Sort by best validation balanced accuracy (desc)
    if "best_val_bal_acc" in df.columns:
        df = df.sort_values("best_val_bal_acc", ascending=False)

    print("\n================== FINAL RESULTS ==================")
    print(df.to_string(index=False))

    # --- Save artifacts ---
    csv_path = out_dir / f"model_results_{run_tag}.csv"
    json_path = out_dir / f"model_results_{run_tag}.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to:\n- {csv_path}\n- {json_path}")


if __name__ == "__main__":
    main()