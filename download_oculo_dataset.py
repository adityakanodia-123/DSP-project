import os
import shutil
import time
import pandas as pd

# Suppress HF symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download

# 1. Prepare target directories
output_dir = "raw_dataset"
normal_dir = os.path.join(output_dir, "normal")
rd_dir = os.path.join(output_dir, "retinal_detachment")

os.makedirs(normal_dir, exist_ok=True)
os.makedirs(rd_dir, exist_ok=True)

print("Fetching dataset metadata (data.csv) from Hugging Face: SankaraEyeHospital/Oculo...", flush=True)
try:
    csv_path = hf_hub_download(repo_id="SankaraEyeHospital/Oculo", filename="data.csv", repo_type="dataset")
    df = pd.read_csv(csv_path)
    print(f"Metadata loaded successfully! Total records in dataset: {len(df)}", flush=True)
except Exception as e:
    print(f"Error fetching data.csv: {e}", flush=True)
    exit(1)

# Categorize dataset records
# Retinal Detachment: retinal_detachment == 1
rd_df = df[df['retinal_detachment'] == 1]

# Normal / Healthy: diagnosis == 0 and all pathology flags == 0
pathology_cols = [
    'vitreous_dot_echo', 'abnormal_contour', 'membranous_echo',
    'posterior_vitreous_detachment', 'retinal_detachment',
    'choroidal_detachment', 'mass_lesion', 'phthisis'
]
normal_df = df[(df['diagnosis'] == 0) & (df[pathology_cols] == 0).all(axis=1)]

print(f"Found {len(rd_df)} Retinal Detachment records and {len(normal_df)} Normal/Healthy records.", flush=True)

# Target limit per class (set to 100 as requested, or set to None to extract all)
TARGET_COUNT = 100 

rd_subset = rd_df.head(TARGET_COUNT) if TARGET_COUNT else rd_df
normal_subset = normal_df.head(TARGET_COUNT) if TARGET_COUNT else normal_df

def download_and_save_images(records, target_folder, category_label, prefix):
    count = 0
    total = len(records)
    print(f"\nProcessing {total} {category_label} images into '{target_folder}'...", flush=True)
    
    for idx, row in records.iterrows():
        img_id = int(row['image_id'])
        img_filename = f"images/{img_id}.png"
        count += 1
        out_filepath = os.path.join(target_folder, f"{prefix}_{count:04d}.png")
        
        # Download or retrieve from local cache
        try:
            local_cached_path = hf_hub_download(
                repo_id="SankaraEyeHospital/Oculo",
                filename=img_filename,
                repo_type="dataset"
            )
            # Copy image to target directory
            shutil.copy(local_cached_path, out_filepath)
            print(f" [{count}/{total}] Saved {out_filepath}", flush=True)
        except Exception as e:
            print(f" [{count}/{total}] Warning: Could not download {img_filename} - {e}", flush=True)
            time.sleep(1)

    return count

# Save Retinal Detachment images
rd_saved = download_and_save_images(rd_subset, rd_dir, "Retinal Detachment", "rd")

# Save Normal images
normal_saved = download_and_save_images(normal_subset, normal_dir, "Normal/Healthy", "normal")

print("\n" + "="*50, flush=True)
print("EXTRACTION COMPLETED SUCCESSFULLY", flush=True)
print(f"Normal / Healthy scans collected:    {normal_saved}", flush=True)
print(f"Retinal Detachment scans collected: {rd_saved}", flush=True)
print(f"Files saved in:                     {os.path.abspath(output_dir)}", flush=True)
print("="*50, flush=True)
