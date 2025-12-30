from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split





class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (DataFrame): DataFrame containing the images and acquisition dates.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform
        
        
    def get_time_of_day(self, date_str):
        # Extract the hour from the date string
        hour = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").time().hour

        # Assign a time of day category based on the hour
        if 9 <= hour < 10:
            return 0  # Morning
        elif 10 <= hour < 11:
            return 1  # Noon
        else:
            return 2  # Evening
        
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    # Assuming the image data is stored as a numpy array in the DataFrame
        img_as_np = self.dataframe.iloc[idx, 0]
        acquisition_date = self.dataframe.iloc[idx, 1]

        # Parse the time from the acquisition_date string
        time_of_day = self.get_time_of_day(acquisition_date)

        # Convert numpy array image to a supported dtype before conversion to tensor
        img_as_np = img_as_np.astype(np.float32)  # Convert to float32 first

        # Now, convert numpy array image to tensor
        image = torch.tensor(img_as_np, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, time_of_day



class DataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=32, mode='train'):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode

    def setup(self, stage=None):
        # Generate a list of time_of_day for all samples in the dataset
        times_of_day = [self.dataset.get_time_of_day(date) for date in self.dataset.dataframe.iloc[:, 1]]

        if self.mode == 'train':
            # Stratified split of the dataset indices
            indices = list(range(len(self.dataset)))
            train_indices, test_indices = train_test_split(indices, test_size=0.3, stratify=times_of_day, random_state=42)

            # Creating PyTorch datasets for training and testing
            self.train_dataset = Subset(self.dataset, train_indices)
            self.test_dataset = Subset(self.dataset, test_indices)
        elif self.mode == 'test':
            self.test_dataset = self.dataset

    def train_dataloader(self):
        if self.mode == 'train':
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=5)
        else:
            # This should not be called in 'test' mode, raise an exception or return None
            raise RuntimeError("Train dataloader called in test mode")
        
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
    
class ScaleTransform:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, pic):
        # Convert to numpy array
        img_array = np.array(pic)
        # Scale the image array to 0-1 range
        img_array = (img_array - self.min_val) / (self.max_val - self.min_val)
        # Ensure the array values are within the bounds after scaling
        img_array = np.clip(img_array, 0, 1)
        # Convert back to PIL Image
        return Image.fromarray(img_array.astype('float32'))

def data_preprocess(mode='test', batch_size=128):


    transform = transforms.Compose([
        transforms.ToPILImage(),  # Ensure the input image is a PIL Image
        ScaleTransform(0, 10000),  # Custom scaling to 0-1 range
        transforms.Resize((120, 120)),  # Resize to the target dimensions
        transforms.ToTensor(),  # Convert the image to a tensor and scale pixel values to 0-1
    ])

    df = pd.read_pickle('df_120.pkl')
    # Convert 'acquisition_date' to datetime without specifying format
    df['acquisition_date_time'] = pd.to_datetime(df['acquisition_date'], errors='coerce')

    # Define the desired number of samples to keep for each time range
    desired_samples_per_hour = 71248

    # Filter the DataFrame to retain only the desired number of samples for each time range
    df = pd.concat([
        df[(df['acquisition_date_time'].dt.hour == hour)][:desired_samples_per_hour]
        for hour in range(9, 12)
    ])
    df = df[['image_data', 'acquisition_date']]
    
    # Create the dataset
    dataset = CustomImageDataset(df, transform=transform)
    # Initialize counts for each time of day class
    data_module = DataModule(dataset, batch_size=batch_size, mode=mode)
    return data_module



