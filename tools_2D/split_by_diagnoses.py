import os
import shutil
from pathlib import Path
import pandas as pd
import pydicom
from tqdm import tqdm

# ---------------- CONFIG ----------------
SAVE_DIR = '/home/etudiant/Projets/Viviane/LIDC-ML/data/LIDC_classes_dcm'
DIAGNOSIS_FILE = '/home/etudiant/Projets/Viviane/LIDC-ML/data/LIDC/tcia-diagnosis-data-2012-04-20.xls'
LIDC_DATA_DIR = '/home/etudiant/Projets/Viviane/LIDC-ML/LIDC'
# ----------------------------------------

# Create output directories
os.makedirs(os.path.join(SAVE_DIR, 'cancer'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'non-cancer'), exist_ok=True)

# ---------- Step 0: Load diagnosis file ----------
diagnosis_df = pd.read_excel(DIAGNOSIS_FILE)
diagnosis_df = diagnosis_df[['TCIA Patient ID', 'Diagnosis']]
diagnosis_df = diagnosis_df.rename(columns={'TCIA Patient ID': 'patient_id'})
diagnosis_df['patient_id'] = diagnosis_df['patient_id'].astype(str).str.zfill(4)
diagnosis_dict = dict(zip(diagnosis_df['patient_id'], diagnosis_df['Diagnosis']))

def is_dicom_file(file_path):
    try:
        _ = pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False

# ---------- Step 1: Walk through LIDC directories ----------
print("🔍 Scanning directories...")
dicom_files = []
for root, _, files in os.walk(LIDC_DATA_DIR):
    for f in files:
        if f.endswith('.dcm'):
            full_path = os.path.join(root, f)
            if is_dicom_file(full_path):
                dicom_files.append(full_path)

print(f"📂 Found {len(dicom_files)} DICOM files")

# ---------- Step 2: Process and copy files ----------
for dcm_path in tqdm(dicom_files, desc="📦 Sorting DICOMs"):
    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)

        patient_id = str(dcm.PatientID).zfill(4)
        diagnosis = diagnosis_dict.get(patient_id)

        if diagnosis in [0, 1]:
            label = 'non-cancer'
        elif diagnosis in [2, 3]:
            label = 'cancer'
        else:
            continue  # skip unknown or missing labels

        new_filename = f"{patient_id}_{Path(dcm_path).stem}.dcm"
        save_path = os.path.join(SAVE_DIR, label, new_filename)

        shutil.copy2(dcm_path, save_path)  # copy with metadata

    except Exception as e:
        print(f"⚠️ Error processing {dcm_path}: {e}")
        continue

print("✅ Done: DICOMs split and saved with original metadata.")