import os
import rasterio
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
from tqdm import tqdm

# Base dir of the original dataset
# BASE_DIR = r'/dt/shabtaia/DT_Satellite/satellite_image_data/BigEarthNet-S2'
BASE_DIR = r'/home/avivyuv/bigearthnet_v2/ResearchMethods/data/BigEarthNet-S2-115-tiles-50-patches'  # adjust as needed
# List of band names in BigEarthNet-S2
BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
# relative for running location in ResearchMethods/ directory, save the chunks here
OUT_DIR = "out_chunks"
os.makedirs(OUT_DIR, exist_ok=True)
BATCH_SIZE = 500

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

def get_next_chunk_id(out_dir: str) -> int:
    # chunk_0000.pkl, chunk_0001.pkl, ...
    # Implements a minimal resume mechanism based on already-saved chunk files.
    ids = []
    for fn in os.listdir(out_dir):
        if fn.startswith("chunk_") and fn.endswith(".pkl"):
            try:
                ids.append(int(fn[len("chunk_"):len("chunk_")+4]))
            except:
                pass
    return (max(ids) + 1) if ids else 0

def main_patches():
    # resume mechanism logic
    chunk_id = get_next_chunk_id(OUT_DIR)
    already_saved = chunk_id * BATCH_SIZE
    skip_loaded = already_saved  # number of *successful* patches to skip

    print(f"Resuming: found {chunk_id} chunk(s) already saved.")
    print(f"Will skip {skip_loaded} successfully-loaded patches (approx resume).")

    batch = []
    loaded_ok = 0

    tile_folders = sorted([f for f in os.scandir(BASE_DIR) if f.is_dir()], key=lambda e: e.name)

    for tile in tqdm(tile_folders, desc="Processing Tiles"):
        patch_folders = sorted([p for p in os.scandir(tile.path) if p.is_dir()], key=lambda e: e.name)

        for patch in patch_folders:
            d = read_patch(patch.path, patch.name)
            if not d:
                continue

            # skip already-processed successes
            if skip_loaded > 0:
                skip_loaded -= 1
                continue

            batch.append(d)
            loaded_ok += 1

            if len(batch) >= BATCH_SIZE:
                df = pd.DataFrame(batch)
                out_path = os.path.join(OUT_DIR, f"chunk_{chunk_id:04d}.pkl")
                df.to_pickle(out_path)
                batch.clear()
                chunk_id += 1
                print(f"Saved {out_path}")

    # save leftovers
    if batch:
        df = pd.DataFrame(batch)
        out_path = os.path.join(OUT_DIR, f"chunk_{chunk_id:04d}.pkl")
        df.to_pickle(out_path)
        batch.clear()
        print(f"Saved {out_path}")

    print(f"Done. Total successfully loaded patches: {loaded_ok}")
    # Before every thing was saved to a big bigearthnet_df.pkl.
    # Now we have multiple chunk_XXXX.pkl files in out_chunks/ directory.
    # Each chunk file contains a DataFrame with columns:
    #   patch_name
    #   time_str
    #   image_matrix (numpy array (12,120,120))
    # That is the equivalent content to your old df, just split across multiple files.


def main_full_dataframe():
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

if __name__ == "__main__":
    main_full_dataframe()