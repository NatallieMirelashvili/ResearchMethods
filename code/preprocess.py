import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Dataset Class ---
class BigEarthNetDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    # This function retrives a processed image and its label
    def __getitem__(self, idx):
        # get image numpy array (12, 120, 120) from dataframe
        img_np = self.dataframe.iloc[idx]['image_matrix']
        
        # Normalization:
        # 1. Convert to float32 
        # 2. scale to [0, 1] by dividing by 10000
        img_np = img_np.astype(np.float32) / 10000.0

        # Clipping to ensure values are within [0, 1] - some values might exceed 1 after scaling (noise)
        img_np = np.clip(img_np, 0, 1) # each value exeeds 1 will be set to 1, values below 0 set to 0
        
        # Convert to torch tensor, the structure saved as (Channels, Height, Width)
        image_tensor = torch.from_numpy(img_np)

        # Appy any augmentations if provided as transform function
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Extract hour label from 'time_str'
        time_str = self.dataframe.iloc[idx]['time_str']
        hour_label = int(time_str[:2]) # extract hour as integer (0-23)
        
        # Return image tensor and hour label
        return image_tensor, torch.tensor(hour_label, dtype=torch.long)

# --- 2. Data Module ---
# Defines the location of data, train - test splitting, batch size, and data transforms
class SatelliteDataModule(LightningDataModule):
    def __init__(self, df, batch_size=32, test_size=0.3):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.test_size = test_size
        self.transform = None #for future augmentations

    def setup(self, stage=None):
        # Run only once before training
        # Extract hour labels for stratification
        labels = [int(t[:2]) for t in self.df['time_str']]
        
        # Split into datasets
        indices = list(range(len(self.df)))
        
        # Ensure stratified split based on hour labels, keep same distribution of hours in train and test
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
        # We must shuffle training data to ensure randomness in each epoch
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        # We do not shuffle validation and test data to ensure consistent evaluation
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

# Main function to run preprocessing and return DataModule
def run_preprocessing_pipeline(pkl_path='bigearthnet_toy_data.pkl', batch_size=16):
    print("Loading DataFrame...")
    try:
        # Load the preprocessed DataFrame from pickle file
        df = pd.read_pickle(pkl_path)
    except FileNotFoundError:
        print(f"Error: Could not find {pkl_path}")
        return None

    # Can add filtering logic here if needed 
    # For example: limit to a maximum of X samples per hour
    # Currently running on all available data
    
    print("Initializing DataModule...")
    data_module = SatelliteDataModule(df, batch_size=batch_size)
    data_module.setup()
    
    return data_module

# --- Run Preprocessing Pipeline and Inspect (Test) ---
if __name__ == "__main__":
    # Returns a DataModule ready for training/testing
    dm = run_preprocessing_pipeline()
    
    if dm:
        # Fetch one batch for inspection
        train_loader = dm.train_dataloader()
        images, labels = next(iter(train_loader))
        
        print("\n--- Batch Inspection ---")
        print(f"Images Shape: {images.shape}") 
        print(f"Labels Shape: {labels.shape}") 
        print(f"Labels Example: {labels}")
        
        # Check pixel value range (normalization)
        print(f"Max pixel value: {images.max():.4f} (Should be <= 1.0)")
        print(f"Min pixel value: {images.min():.4f} (Should be >= 0.0)")