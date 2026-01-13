import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import os

pkl_relative_path = 'bigearthnet_df.pkl'

# --- Dataset Class ---
class BigEarthNetDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.resize = T.Resize((224, 224), antialias=True)  # Resize to (224, 224) for ViT compatibility
        
    def __len__(self):
        return len(self.dataframe)

    # This function retrieves a processed image and its label
    def __getitem__(self, idx):
        # Get image numpy array from dataframe
        img_np = self.dataframe.iloc[idx]['image_matrix']
        
        # Normalization:
        # 1. Convert to float32 
        # 2. Scale to [0, 1] by dividing by 10000
        img_np = img_np.astype(np.float32) / 10000.0

        # Clipping to ensure values are within [0, 1] (noise handling)
        img_np = np.clip(img_np, 0, 1) 
        
        # Convert to torch tensor, structure: (Channels, Height, Width)
        image_tensor = torch.from_numpy(img_np)
        image_tensor = self.resize(image_tensor)

        # Apply augmentations if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # --- NEW LABEL LOGIC ---
        # Extract hour and minute to calculate 30-minute intervals
        time_str = self.dataframe.iloc[idx]['time_str'] # Format "HHMMSS"
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        
        # Formula: (Hour - 9) * 2 + (1 if minute >= 30 else 0)
        # Example 09:15 -> (9-9)*2 + 0 = 0
        # Example 09:45 -> (9-9)*2 + 1 = 1
        # Example 11:45 -> (11-9)*2 + 1 = 5
        label_id = (hour - 9) * 2 + (1 if minute >= 30 else 0)
        
        return image_tensor, torch.tensor(label_id, dtype=torch.long)

# --- 2. Data Module ---
class SatelliteDataModule(LightningDataModule):
    def __init__(self, df, batch_size=32, test_size=0.3):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.test_size = test_size
        self.transform = None 

    def setup(self, stage=None):
        # Run only once before training
        
        # --- NEW STRATIFICATION LOGIC ---
        # Calculate labels for the entire dataset to ensure balanced split
        labels = []
        for t in self.df['time_str']:
            hour = int(t[:2])
            minute = int(t[2:4])
            lbl = (hour - 9) * 2 + (1 if minute >= 30 else 0)
            labels.append(lbl)
        
        # Split into datasets
        indices = list(range(len(self.df)))
        
        # Ensure stratified split based on the new 30-minute labels
        train_idxs, test_idxs = train_test_split(
            indices, 
            test_size=self.test_size, 
            stratify=labels, 
            random_state=42
        )

        # Create Dataset objects
        full_dataset = BigEarthNetDataset(self.df, transform=self.transform)
        self.train_dataset = Subset(full_dataset, train_idxs)
        self.test_dataset = Subset(full_dataset, test_idxs)

        print(f"Split Summary: Train={len(self.train_dataset)}, Test={len(self.test_dataset)}")

    def train_dataloader(self):
        # Shuffle training data
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


# Main function to run preprocessing and return DataModule
def run_preprocessing_pipeline(pkl_path=None, pkl_paths=None, batch_size=32):
    # pkl_paths: list of chunk files (recommended)
    # pkl_path: single file (backward compatible)

    if pkl_paths is None:
        # fall back to single-file behavior
        pkl_paths = [pkl_path or 'bigearthnet_df.pkl']

    print(f"Loading {len(pkl_paths)} DataFrame chunk(s)...")

    dfs = []
    for path in pkl_paths:
        try:
            df = pd.read_pickle(path)
        except FileNotFoundError:
            print(f"Error: Could not find {path}")
            return None

        # --- SAME FILTERING LOGIC ---
        # Filter data to keep only hours between 09:00 and 11:59
        df['hour_temp'] = df['time_str'].apply(lambda x: int(x[:2]))
        df = df[df['hour_temp'].between(9, 11)].reset_index(drop=True)
        df = df.drop(columns=['hour_temp'])

        if len(df) > 0:
            dfs.append(df)

    if len(dfs) == 0:
        print("Error: No data found in the specified time range (09-12).")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Filtered Dataset size: {len(df)} images")

    print("Initializing DataModule...")
    data_module = SatelliteDataModule(df, batch_size=batch_size)
    data_module.setup()

    return data_module

# --- Run Preprocessing Pipeline and Inspect (Test) ---
if __name__ == "__main__":
    print("--- Running Preprocessing Pipeline ---")
    # Returns a DataModule ready for training/testing
    dm = run_preprocessing_pipeline(pkl_path=pkl_relative_path, batch_size=32)
    
    if dm is not None:
        torch.save(dm, "datamodule.pt")
        print("✅ DataModule saved successfully as datamodule.pt")
    else:
        print("❌ DataModule creation failed.")

# # --- Run Preprocessing Pipeline and Inspect (Test) ---
# if __name__ == "__main__":
#     print("--- Running Preprocessing Pipeline ---")

#     chunks_dir = "out_chunks"
#     if os.path.isdir(chunks_dir):
#         chunk_paths = [
#             os.path.join(chunks_dir, f)
#             for f in sorted(os.listdir(chunks_dir))
#             if f.startswith("chunk_") and f.endswith(".pkl")
#         ]
#     else:
#         chunk_paths = []

#     # Prefer chunks if available, otherwise fall back to single pkl
#     if len(chunk_paths) > 0:
#         print(f"Found {len(chunk_paths)} chunk file(s) in {chunks_dir}. Using chunks.")
#         dm = run_preprocessing_pipeline(pkl_paths=chunk_paths, batch_size=32)
#     else:
#         print(f"No chunk files found. Falling back to {pkl_relative_path}.")
#         dm = run_preprocessing_pipeline(pkl_path=pkl_relative_path, batch_size=32)

#     if dm is not None:
#         torch.save(dm, "datamodule.pt")
#         print("✅ DataModule saved successfully as datamodule.pt")
#     else:
#         print("❌ DataModule creation failed.")
