import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from preprocess import run_preprocessing_pipeline  
from build_model import BuildModel             

# pkl_relative_path = 'bigearthnet_df.pkl'
PKL_CHUNKS_DIR = "out_chunks"
DATAMODULE_PATH = "datamodule.pt"
config_path = "configurations/models_config.yaml"

def list_chunk_pkls(chunks_dir: str):
    # helper to list all chunk .pkl files
    return [
        os.path.join(chunks_dir, f)
        for f in sorted(os.listdir(chunks_dir))
        if f.startswith("chunk_") and f.endswith(".pkl")
    ]

class SatelliteClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=['model']) 
        self.test_preds = []
        self.test_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        # x - pictures
        # y - labels
        x, y = batch
        # what is logits?
        logits = self(x) 
        loss = self.criterion(logits, y) 
        
        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log to progress bar
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        self.test_preds.append(preds.cpu())
        self.test_labels.append(y.cpu())

        acc = (preds == y).float().mean()
        self.log('test_acc', acc)


    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)


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
        # dm = run_preprocessing_pipeline(pkl_path=pkl_relative_path, batch_size=32)
        chunk_paths = list_chunk_pkls(PKL_CHUNKS_DIR)
        dm = run_preprocessing_pipeline(pkl_paths=chunk_paths, batch_size=32)
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
        
        # 2. Wrap in Lightning Module
        system = SatelliteClassifier(model=raw_model, lr=3e-4)

        model_ckpt_dir = f'checkpoints/{model_name}/'

        # 3. Setup Checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
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
        
if __name__ == "__main__":
    main()