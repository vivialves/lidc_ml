import os
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm
import cv2

# ---------------- CONFIG ----------------
PATCH_SIZE = 512  # change as needed
STRIDE = 256       # overlap between patches
SAVE_DIR = '/home/vivianea/projects/BrainInnov/data/LIDC_patches'
DIAGNOSIS_FILE = '/home/vivianea/projects/BrainInnov/data/LIDC/tcia-diagnosis-data-2012-04-20.xls'
LIDC_DATA_DIR = '/home/vivianea/projects/BrainInnov/data/LIDC'
# ---------------------------------------

# Create save directories
os.makedirs(os.path.join(SAVE_DIR, 'cancer'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'non-cancer'), exist_ok=True)

# ---------- Step 0: Load diagnosis file ----------
diagnosis_df = pd.read_excel(DIAGNOSIS_FILE)
diagnosis_df = diagnosis_df[['TCIA Patient ID', 'Diagnosis']]
diagnosis_df = diagnosis_df.rename(columns={'TCIA Patient ID': 'patient_id'})
diagnosis_df['patient_id'] = diagnosis_df['patient_id'].astype(str).str.zfill(4)
diagnosis_dict = dict(zip(diagnosis_df['patient_id'], diagnosis_df['Diagnosis']))

def load_scan(dicom_dir):
    slices = []
    for root, _, files in os.walk(dicom_dir):
        for f in files:
            if f.endswith('.dcm'):
                path = os.path.join(root, f)
                try:
                    dcm = pydicom.dcmread(path)
                    if hasattr(dcm, 'ImagePositionPatient') and hasattr(dcm, 'pixel_array'):
                        slices.append(dcm)
                except:
                    continue
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices

def normalize_image(img):
    img = np.clip(img, -1000, 400)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return (img * 255).astype(np.uint8)

# ---------- Step 1: Build path dataframe ----------
df_path = pd.DataFrame(columns=['filename', 'dirname'])
for dirname, _, filenames in os.walk(LIDC_DATA_DIR):
    if filenames:
        df_path.loc[len(df_path)] = {'filename': filenames, 'dirname': dirname}

def contains_dcm(file_list):
    return any(f.endswith('.dcm') for f in file_list) if isinstance(file_list, list) else False

df_fil = df_path[df_path['filename'].apply(contains_dcm)]
print(f"Found {len(df_fil)} DICOM directories")

# ---------- Step 2: Process each patient ----------
for _, row in tqdm(df_fil.iterrows(), total=len(df_fil)):
    filenames = row['filename']
    dirname = row['dirname']

    parent_dir = Path(dirname).parent
    dicom_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    dicom_dir = None
    for d in dicom_dirs:
        if any(f.endswith('.dcm') for f in os.listdir(d)):
            dicom_dir = d
            break

    if not dicom_dir:
        continue

    patient_id = Path(dirname).parents[1].name
    pid = str(patient_id).zfill(4)
    diagnosis = diagnosis_dict.get(pid)

    label = None
    if diagnosis == 0 or diagnosis == 1:
        label = 'non-cancer'
    elif diagnosis == 2 or diagnosis == 3:
        label = 'cancer'
    else:
        continue
 
    slices = load_scan(dicom_dir)
    if not slices:
        continue

    for i, slice in enumerate(slices):
        try:
            img = slice.pixel_array.astype(np.int16)
            img = normalize_image(img)

            # Convert grayscale to 3 channels for PIL compatibility
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            h, w, _ = img_rgb.shape
            for y in range(0, h - PATCH_SIZE + 1, STRIDE):
                for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                    patch = img_rgb[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    patch_filename = f"{patient_id}_slice{i:03d}_y{y}_x{x}.png"
                    patch_path = os.path.join(SAVE_DIR, label, patch_filename)
                    Image.fromarray(patch).save(patch_path)

        except Exception as e:
            print(f"Error processing slice {i} of {patient_id}: {e}")
            continue

print("✅ Done: Patches saved and labeled by diagnosis!")