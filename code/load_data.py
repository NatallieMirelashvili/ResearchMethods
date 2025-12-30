# import os
# import rasterio
# from rasterio.enums import Resampling
# import pandas as pd
# import numpy as np
# from tqdm import tqdm


# # BASE_DIR = '/dt/shabtaia/DT_Satellite/satellite_image_data/BigEarthNet-S2' 
# BASE_DIR = r'C:\Users\natal\OneDrive\תואר שני\experiment\ResearchMethods\data\BigEarthNet-S2-toy'
# BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# # --- DEBUG SNIPPET START ---
# print("\n=== DEBUGGING FILENAMES ===")
# # בודק את התיקייה הראשונה שהוא מוצא
# test_dirs = [d for d in os.scandir(BASE_DIR) if d.is_dir()]
# if test_dirs:
#     first_folder = test_dirs[0]
#     print(f"Checking folder: {first_folder.name}")
#     print("Files found inside:")
#     files = os.listdir(first_folder.path)
#     for f in files[:5]: # מדפיס רק את ה-5 הראשונים
#         print(f" - {f}")
# else:
#     print("No folders found in BASE_DIR!")
# print("===========================\n")
# # --- DEBUG SNIPPET END ---

# def parse_timestamp_from_name(folder_name): # time + date
#     try:
#         parts = folder_name.split('_')
#         if len(parts) > 2:
#             return parts[2] 
#     except:
#         return None
#     return None

# def read_patch(patch_folder_path, patch_name):
#     timestamp_str = parse_timestamp_from_name(patch_name)
#     if timestamp_str is None:
#         return None

#     time_str = timestamp_str[9:]
    
  

#     # read all bands
#     bands_data = []
#     try:
#         for band in BAND_NAMES:
#             tif_filename = f"{patch_name}_{band}.tif"
#             tif_path = os.path.join(patch_folder_path, tif_filename)
            
#             # if the file does not exsist (happend sometimes if using clusters with less downloading)
#             if not os.path.exists(tif_path):
#                 return None
#                 # # נסה אופציה ב' - לפעמים השם הוא רק B01.tif בלי הפרפיקס הארוך
#                 # # תלוי איך חילצו את התיקיות בקלאסטר
#                 # tif_path = os.path.join(patch_folder_path, f"{band}.tif") 
#                 # if not os.path.exists(tif_path):
#                 #     return None # אם חסר ערוץ, מדלגים על הפאץ'

#             with rasterio.open(tif_path) as src:
#                 # קריאה ושינוי גודל ל-120x120
#                 band_img = src.read(
#                     1,
#                     out_shape=(120, 120),
#                     resampling=Resampling.bilinear
#                 )
#                 bands_data.append(band_img)
        
#         return {
#             'patch_name': patch_name,
#             'time_str': time_str,  # time label (only time with no date)
#             'image_matrix': np.stack(bands_data, axis=0) # (12, 120, 120)
#         }

#     except Exception as e:
#         print(f"Error in {patch_name}: {e}")
#         return None

# # main processing loop
# all_data = []

# print(f"Scanning directory: {BASE_DIR}")
# with os.scandir(BASE_DIR) as entries:
#     # Beware of size! 
#     folder_entries = [entry for entry in entries if entry.is_dir()]

# print(f"Found {len(folder_entries)} folders. Processing...")

# for entry in tqdm(folder_entries):
#     patch_data = read_patch(entry.path, entry.name)
    
#     if patch_data:
#         all_data.append(patch_data)

# # sameple output saving
# print("Saving extracted data...")
# df = pd.DataFrame(all_data)

# # --- Data Inspection (Add to the end of the file) ---
# print("\n" + "="*30)
# print("DataFrame Inspection")
# print("="*30)

# # 1. How many rows were loaded?
# print(f"Total patches loaded: {len(df)}")

# # 2. Peek at the first 5 rows (only key columns)
# if not df.empty:
#     print("\nFirst 5 rows:")
#     print(df[['patch_name', 'time_str']].head())

#     # 3. Critical check: Is the matrix the correct size?
#     # We expect to see (12, 120, 120)
#     first_matrix_shape = df.iloc[0]['image_matrix'].shape
#     print(f"\nShape of the first image matrix: {first_matrix_shape}")
    
#     if first_matrix_shape == (12, 120, 120):
#         print("✅ Success! Image shape is correct.")
#     else:
#         print(f"⚠️ Warning! Expected (12, 120, 120) but got {first_matrix_shape}")
# else:
#     print("❌ DataFrame is empty. Check the path again.")

# df.to_pickle('bigearthnet_data_from_filenames.pkl')
# print("Done.")

import os
import rasterio
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
from tqdm import tqdm

# Update this path if needed
BASE_DIR = r'C:\Users\natal\OneDrive\תואר שני\experiment\ResearchMethods\data\BigEarthNet-S2-toy'

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

    output_filename = 'bigearthnet_toy_data.pkl'
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


import matplotlib.pyplot as plt

def show_satellite_image(img_matrix, title="Satellite Image"):
    """
   Visualizes a satellite image using the Red, Green, and Blue bands.
    Assumes img_matrix has shape (bands, height, width) with bands ordered as: 
    0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red), ...
    3 (Red), 2 (Green), and 1 (Blue) are used
    """

    # 0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red), ...
    

    r = img_matrix[3] # Red
    g = img_matrix[2] # Green
    b = img_matrix[1] # Blue
    
    rgb_image = np.stack([r, g, b], axis=-1)
    
    rgb_image = rgb_image.astype(float)
    rgb_image = rgb_image / rgb_image.max()
    
    plt.imshow(rgb_image)
    plt.title(title)
    plt.axis('off') 

print("Generating visualization...")
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
first_img = df.iloc[0]['image_matrix']
first_name = df.iloc[0]['patch_name']
show_satellite_image(first_img, title=f"First Patch\n{first_name[-10:]}")


plt.subplot(1, 2, 2)
last_img = df.iloc[-1]['image_matrix']
last_name = df.iloc[-1]['patch_name']
show_satellite_image(last_img, title=f"Last Patch\n{last_name[-10:]}")

plt.tight_layout()
plt.show()