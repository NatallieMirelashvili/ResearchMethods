import os
import rasterio
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
from tqdm import tqdm


BASE_DIR = '/dt/shabtaia/DT_Satellite/satellite_image_data/BigEarthNet-S2' 

# רשימת הערוצים (ודא שזה תואם את מה שיש בתוך התיקיות, לפעמים בגרסאות מסוימות יש רק 10)
BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

def parse_timestamp_from_name(folder_name):
    """
    מחלץ את התאריך והשעה משם התיקייה
    מבנה: S2B_MSIL2A_20180525T094029_...
    אנחנו רוצים את החלק השלישי (אינדקס 2)
    """
    try:
        parts = folder_name.split('_')
        if len(parts) > 2:
            # מחזיר מחרוזת בסגנון: 20180525T094029
            return parts[2] 
    except:
        return None
    return None

def read_patch(patch_folder_path, patch_name):
    # 1. חילוץ Timestamp משם התיקייה (פעולה מתמטית מהירה בלי לקרוא קובץ)
    timestamp_str = parse_timestamp_from_name(patch_name)
    
    if timestamp_str is None:
        return None

    # 2. קריאת ערוצי התמונה
    bands_data = []
    try:
        for band in BAND_NAMES:
            # הרכבת שם קובץ ה-TIF: שםהתיקייה_שםהערוץ.tif
            tif_filename = f"{patch_name}_{band}.tif"
            tif_path = os.path.join(patch_folder_path, tif_filename)
            
            # אם הקובץ לא קיים (קורה לפעמים בקלאסטרים עם הורדות חלקיות)
            if not os.path.exists(tif_path):
                # נסה אופציה ב' - לפעמים השם הוא רק B01.tif בלי הפרפיקס הארוך
                # תלוי איך חילצו את התיקיות בקלאסטר
                tif_path = os.path.join(patch_folder_path, f"{band}.tif") 
                if not os.path.exists(tif_path):
                    return None # אם חסר ערוץ, מדלגים על הפאץ'

            with rasterio.open(tif_path) as src:
                # קריאה ושינוי גודל ל-120x120
                band_img = src.read(
                    1,
                    out_shape=(120, 120),
                    resampling=Resampling.bilinear
                )
                bands_data.append(band_img)
        
        # בניית המילון להחזרה
        return {
            'patch_name': patch_name,
            'timestamp': timestamp_str,  # המחרוזת שחילצנו
            'image_matrix': np.stack(bands_data, axis=0) # (12, 120, 120)
        }

    except Exception as e:
        # print(f"Error in {patch_name}: {e}")
        return None

# --- ריצה ראשית ---
all_data = []

print(f"Scanning directory: {BASE_DIR}")
# שימוש ב-scandir ליעילות מקסימלית בקלאסטר
with os.scandir(BASE_DIR) as entries:
    # לוקחים דוגמית או את הכל (זהירות עם הזיכרון!)
    folder_entries = [entry for entry in entries if entry.is_dir()]

print(f"Found {len(folder_entries)} folders. Processing...")

for entry in tqdm(folder_entries):
    patch_data = read_patch(entry.path, entry.name)
    
    if patch_data:
        all_data.append(patch_data)

# שמירה
print("Saving extracted data...")
df = pd.DataFrame(all_data)
# מומלץ לשמור שם ייחודי כדי לא לדרוס ריצות קודמות
df.to_pickle('bigearthnet_data_from_filenames.pkl')
print("Done.")