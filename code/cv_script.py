#!/usr/bin/env python3
"""
cv_script.py (drop-in)

5-fold Stratified CV for time-of-day (30-min bins, 09:00–11:59) using the exact same
BigEarthNetDataset preprocessing path (normalization + Resize(224,224)) as in preprocess.py.

Key points:
- Fixes ViT crash by ensuring all inputs are 224x224 (via BigEarthNetDataset).
- Uses StratifiedKFold on the same label formula you use in training.
- Checkpoints based on val_bal_acc (macro recall).
- Evaluates each fold on its validation split via trainer.test(..., dataloaders=val_loader, ckpt_path="best")
- Writes per-fold and summary results to results_cv/.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import StratifiedKFold

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall

# Your project modules
from preprocess import BigEarthNetDataset
from build_model import BuildModel


# ----------------------------
# Helpers
# ----------------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def compute_label_from_timestr(t: str) -> int:
    """
    Maps time_str "HHMMSS" to 30-minute bin label in [0..5] for 09:00–11:59.
    label = (hour-9)*2 + (minute>=30)
    """
    t = str(t).zfill(6)
    hour = int(t[:2])
    minute = int(t[2:4])
    return (hour - 9) * 2 + (1 if minute >= 30 else 0)


def load_and_filter_df(pkl_path: str) -> pd.DataFrame:
    """
    Loads bigearthnet_df.pkl and keeps only times 09:00–11:59.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL not found: {pkl_path}")

    df = pd.read_pickle(pkl_path).copy()

    # Filter to 09–11 (inclusive) by hour
    df["hour_temp"] = df["time_str"].apply(lambda x: int(str(x).zfill(6)[:2]))
    df = df[df["hour_temp"].between(9, 11)].reset_index(drop=True)
    df = df.drop(columns=["hour_temp"])

    # Compute labels and keep only valid bins [0..5]
    y = df["time_str"].apply(compute_label_from_timestr).to_numpy()
    ok = (y >= 0) & (y < 6)
    df = df.loc[ok].reset_index(drop=True)

    return df


# ----------------------------
# Lightning Module (self-contained)
# ----------------------------
class SatelliteClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
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

        self.save_hyperparameters(ignore=["model"])

        # Accuracy
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        # Balanced accuracy ≈ macro recall in multiclass
        self.val_bal_acc = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_bal_acc = MulticlassRecall(num_classes=num_classes, average="macro")

        # Macro-F1
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc.update(preds, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, y)
        self.val_bal_acc.update(preds, y)
        self.val_f1_macro.update(preds, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_bal_acc", self.val_bal_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1_macro", self.val_f1_macro, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_acc.update(preds, y)
        self.test_bal_acc.update(preds, y)
        self.test_f1_macro.update(preds, y)

        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_bal_acc", self.test_bal_acc, on_step=False, on_epoch=True)
        self.log("test_f1_macro", self.test_f1_macro, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ----------------------------
# Main CV routine
# ----------------------------
def run_cv(
    pkl_path: str,
    config_path: str,
    model_name: str = "vit_tiny",
    n_splits: int = 5,
    seed: int = 42,
    max_epochs: int = 5,
    batch_size: int = 32,
    num_workers: int = 2,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    out_dir: str = "results_cv",
) -> Dict[str, Any]:
    pl.seed_everything(seed, workers=True)

    df = load_and_filter_df(pkl_path)
    y = df["time_str"].apply(compute_label_from_timestr).to_numpy()

    # Dataset uses SAME preprocessing logic (normalization + Resize(224,224))
    full_ds = BigEarthNetDataset(df, transform=None)

    builder = BuildModel(config_path)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows: List[Dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(df)), y), start=1):
        print(f"\n==================== Fold {fold}/{n_splits} ====================")

        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)

        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        raw_model = builder.build(model_name)

        # Model identity print (avoid “order confusion”)
        n_params = sum(p.numel() for p in raw_model.parameters())
        print(f"Built model type: {type(raw_model).__name__} | params: {n_params/1e6:.2f}M")

        system = SatelliteClassifier(
            model=raw_model,
            lr=lr,
            num_classes=6,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )

        ckpt_dir = out_dir_p / "checkpoints" / model_name / f"fold_{fold}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_bal_acc",
            mode="max",
            dirpath=str(ckpt_dir),
            filename=f"{model_name}-best",
            save_top_k=1,
            save_last=False,
            verbose=True,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            enable_checkpointing=True,
        )

        trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader)

        print(f"\n🧪 Testing best checkpoint on fold-{fold} validation split")
        test_results = trainer.test(system, dataloaders=val_loader, ckpt_path="best")
        r = test_results[0] if test_results and isinstance(test_results, list) else {}

        row = {
            "run_tag": run_tag,
            "fold": fold,
            "model_name": model_name,
            "built_model_type": type(raw_model).__name__,
            "n_params": int(n_params),
            "best_model_path": checkpoint_callback.best_model_path,
            "best_val_bal_acc_ckpt_score": _safe_float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None,
            # These keys come from the test() logs
            "val_as_test_acc": _safe_float(r.get("test_acc")),
            "val_as_test_bal_acc": _safe_float(r.get("test_bal_acc")),
            "val_as_test_f1_macro": _safe_float(r.get("test_f1_macro")),
        }
        fold_rows.append(row)

        print(
            f"✅ Fold {fold} | "
            f"best_val_bal_acc_ckpt_score={row['best_val_bal_acc_ckpt_score'] if row['best_val_bal_acc_ckpt_score'] is not None else float('nan'):.4f} | "
            f"val_as_test_bal_acc={row['val_as_test_bal_acc'] if row['val_as_test_bal_acc'] is not None else float('nan'):.4f} | "
            f"val_as_test_f1_macro={row['val_as_test_f1_macro'] if row['val_as_test_f1_macro'] is not None else float('nan'):.4f}"
        )

    # Summary
    df_folds = pd.DataFrame(fold_rows)

    summary = {}
    for k in ["val_as_test_acc", "val_as_test_bal_acc", "val_as_test_f1_macro"]:
        if k in df_folds.columns:
            vals = pd.to_numeric(df_folds[k], errors="coerce").dropna().to_numpy()
            summary[k] = {
                "mean": float(np.mean(vals)) if len(vals) else None,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else None,
                "min": float(np.min(vals)) if len(vals) else None,
                "max": float(np.max(vals)) if len(vals) else None,
            }

    print("\n================== CV RESULTS (per fold) ==================")
    print(df_folds.to_string(index=False))

    print("\n================== CV SUMMARY ==================")
    for metric, stats in summary.items():
        print(
            f"{metric}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )

    # Save artifacts
    csv_path = out_dir_p / f"cv_folds_{model_name}_{run_tag}.csv"
    json_path = out_dir_p / f"cv_folds_{model_name}_{run_tag}.json"
    sum_path = out_dir_p / f"cv_summary_{model_name}_{run_tag}.json"

    df_folds.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fold_rows, f, indent=2)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved CV artifacts to:\n- {csv_path}\n- {json_path}\n- {sum_path}")

    return {"folds": fold_rows, "summary": summary}


def main():
    # Minimal “drop-in” config (edit as needed)
    PKL_PATH = "bigearthnet_df.pkl"
    CONFIG_PATH = "configurations/models_config.yaml"
    MODEL_NAME = "efficientnet_b0"

    run_cv(
        pkl_path=PKL_PATH,
        config_path=CONFIG_PATH,
        model_name=MODEL_NAME,
        n_splits=5,
        seed=42,
        max_epochs=5,
        batch_size=32,
        num_workers=2,  # avoids your warning about too many workers
        lr=3e-4,
        weight_decay=0.0,
        label_smoothing=0.0,
        out_dir="results_cv",
    )


if __name__ == "__main__":
    main()
