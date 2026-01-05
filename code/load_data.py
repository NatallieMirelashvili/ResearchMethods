import os
import rasterio
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
from tqdm import tqdm

# Update this path if needed
BASE_DIR = r'/dt/shabtaia/DT_Satellite/satellite_image_data/BigEarthNet-S2'

BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

def parse_timestamp_from_name(folder_name): 
    try:
        parts = folder_name.split('_')
        # Works for both Tile name and Patch name as the timestamp is always at index 2
        if len(parts) > 2:
            return parts[2] 
    except:
        return None
    return None

def read_patch(patch_folder_path, patch_name):
    # Extract timestamp
    timestamp_str = parse_timestamp_from_name(patch_name)
    if timestamp_str is None:
        return None

    # Extract time only (HHMMSS)
    time_str = timestamp_str[9:]
    
    bands_data = []
    try:
        for band in BAND_NAMES:
            # Construct filename: PatchName_BandName.tif
            tif_filename = f"{patch_name}_{band}.tif"
            tif_path = os.path.join(patch_folder_path, tif_filename)
            
            if not os.path.exists(tif_path):
                return None

            with rasterio.open(tif_path) as src:
                # Read and resize to 120x120
                band_img = src.read(
                    1,
                    out_shape=(120, 120),
                    resampling=Resampling.bilinear
                )
                bands_data.append(band_img)
        
        # Stack bands into (12, 120, 120)
        return {
            'patch_name': patch_name,
            'time_str': time_str,  
            'image_matrix': np.stack(bands_data, axis=0) 
        }

    except Exception as e:
        # print(f"Error in {patch_name}: {e}")
        return None

# --- Main Processing Loop ---
all_data = []

if not os.path.exists(BASE_DIR):
    print(f"Error: Directory not found at {BASE_DIR}")
else:
    print(f"Scanning Base Directory: {BASE_DIR}")
    
    # 1. Get all Tile folders
    tile_folders = [f for f in os.scandir(BASE_DIR) if f.is_dir()]
    print(f"Found {len(tile_folders)} Tiles. Scanning for Patches inside...")

    # 2. Loop over Tiles
    for tile in tqdm(tile_folders, desc="Processing Tiles"):
        
        # 3. Get all Patches inside the current Tile
        patch_folders = [p for p in os.scandir(tile.path) if p.is_dir()]
        
        # 4. Loop over Patches
        for patch in patch_folders:
            patch_data = read_patch(patch.path, patch.name)
            if patch_data:
                all_data.append(patch_data)

    # --- Save and Inspect ---
    print("\nCreating DataFrame...")
    df = pd.DataFrame(all_data)

    output_filename = 'bigearthnet_df.pkl'
    df.to_pickle(output_filename)



    # Debug prints - disable before running large scale on cluster!
    print(f"Saved to {output_filename}")

    # --- Data Inspection ---
    print("\n" + "="*30)
    print("DataFrame Inspection")
    print("="*30)

    print(f"Total patches loaded: {len(df)}")

    if not df.empty:
        print("\nFirst 5 rows:")
        print(df[['patch_name', 'time_str']].head())

        first_matrix_shape = df.iloc[0]['image_matrix'].shape
        print(f"\nShape of the first image matrix: {first_matrix_shape}")
        
        if first_matrix_shape == (12, 120, 120):
            print("✅ Success! Image shape is correct.")
        else:
            print(f"⚠️ Warning! Expected (12, 120, 120) but got {first_matrix_shape}")
    else:
        print("❌ DataFrame is empty. Check the path again.")


# import matplotlib.pyplot as plt

# def show_satellite_image(img_matrix, title="Satellite Image"):
#     """
#    Visualizes a satellite image using the Red, Green, and Blue bands.
#     Assumes img_matrix has shape (bands, height, width) with bands ordered as: 
#     0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red), ...
#     3 (Red), 2 (Green), and 1 (Blue) are used
#     """

#     # 0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red), ...
    

#     r = img_matrix[3] # Red
#     g = img_matrix[2] # Green
#     b = img_matrix[1] # Blue
    
#     rgb_image = np.stack([r, g, b], axis=-1)
    
#     rgb_image = rgb_image.astype(float)
#     rgb_image = rgb_image / rgb_image.max()
    
#     plt.imshow(rgb_image)
#     plt.title(title)
#     plt.axis('off') 

# print("Generating visualization...")
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# first_img = df.iloc[0]['image_matrix']
# first_name = df.iloc[0]['patch_name']
# show_satellite_image(first_img, title=f"First Patch\n{first_name[-10:]}")


# plt.subplot(1, 2, 2)
# last_img = df.iloc[-1]['image_matrix']
# last_name = df.iloc[-1]['patch_name']
# show_satellite_image(last_img, title=f"Last Patch\n{last_name[-10:]}")

# plt.tight_layout()
# plt.show()