import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from preprocess import run_preprocessing_pipeline  
from build_model import BuildModel             


class SatelliteClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=['model']) 

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
    dm = run_preprocessing_pipeline(pkl_path='/dt/shabtaia/DT_Satellite/satellite_image_data/BigEarthNet-S2/bigearthnet_df.pkl', batch_size=32)
    
    if dm is None:
        print("DataModule could not be loaded. Exiting.")
        return


    print("💾 Saving DataModule to disk...")
    saved_path = "/dt/shabtaia/DT_Satellite/satellite_image_data/BigEarthNet-S2/datamodule.pt"
    torch.save(dm, saved_path) 
    print("✅ DataModule saved to ", saved_path)

    # 2. Build Model Builder
    print("\n--- Setting up Model Builder ---")
    config_path = "ResearchMethods/configurations/models_config.yaml"
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

        # 3. Setup Checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath='checkpoints/',
            filename=f'{model_name}-best',
            save_top_k=1,
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
        trainer.fit(system, datamodule=dm)
        
if __name__ == "__main__":
    main()