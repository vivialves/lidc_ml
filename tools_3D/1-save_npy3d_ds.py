#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Save Numpy Dataset from Dicom ------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

import os
import random
import torch
import pydicom
import csv

import SimpleITK as sitk
import numpy as np

from monai.transforms import Resize

from sklearn.model_selection import train_test_split


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Constants -------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------


IMAGE_SIZE = (256, 256, 256)
SEED = 42
DICOM_DIR = "/home/vivianea/projects/BrainInnov/data/LIDC_classes_dcm"
SAVE_DIR = "/home/vivianea/projects/BrainInnov/data/npy_3D_splitted"
VAL_RATIO = 0.2
TEST_RATIO = 0.2

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------  Processing ----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------


# Set seed for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)



def build_patient_dict(base_dcm_dir):
    class_dict = {"cancer": {}, "non-cancer": {}}
    for cls in ["cancer", "non-cancer"]:
        cls_path = os.path.join(base_dcm_dir, cls)
        for root, _, files in os.walk(cls_path):
            for fname in files:
                if fname.endswith(".dcm"):
                    pid = fname.split("_")[0]
                    if pid not in class_dict[cls]:
                        class_dict[cls][pid] = []
                    class_dict[cls][pid].append(os.path.join(root, fname))

    print(f"✅ Patient grouping complete.")
    print(f"🧠 Classes: {list(class_dict.keys())}")
    print(f"📦 Total patients: {sum(len(pids) for pids in class_dict.values())}")
    return class_dict

def split_by_patient(class_dict):
    train, val, test = [], [], []
    for label_name, pid_dict in class_dict.items():
        label = 1 if label_name == "cancer" else 0
        pids = list(pid_dict.keys())
        random.shuffle(pids)
        train_p, test_p = train_test_split(pids, test_size=TEST_RATIO, random_state=SEED)
        train_p, val_p = train_test_split(train_p, test_size=VAL_RATIO / (1 - TEST_RATIO), random_state=SEED)

        for pid in train_p:
            train.append((pid_dict[pid], label))
        for pid in val_p:
            val.append((pid_dict[pid], label))
        for pid in test_p:
            test.append((pid_dict[pid], label))

    return train, val, test


def resize_volume(volume, target_shape):
    resize = Resize(spatial_size=target_shape, mode="trilinear")
    resized = resize(volume)
    if isinstance(resized, np.ndarray):
        return resized
    elif hasattr(resized, "numpy"):
        return resized.numpy()
    else:
        return np.asarray(resized)

def normalize_volume(volume):
    volume = np.clip(volume, -1000, 150)
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)
    return volume.astype(np.float32)

def load_dicom_volume(dcm_paths, target_size=IMAGE_SIZE, min_slices=3):
    slices = []
    for path in dcm_paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=False)
            if hasattr(ds, 'InstanceNumber'):
                slices.append((ds.InstanceNumber, path))
        except Exception:
            print(f"DICOM read failed: {path} | {e}")
            continue

    if len(slices) < min_slices:
        return None

    slices.sort(key=lambda x: x[0])
    sorted_paths = [p for _, p in slices]

    volume = []
    for path in sorted_paths:
        try:
            img = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(img)[0]  # (H, W)
            expected_shape = (512, 512) # This is the expected shape form he dicom images, because there is different shapes that don`t correspond the right folder`
            if array.shape == expected_shape:  
                volume.append(array)
        except Exception as e:
            continue

    if len(volume) < min_slices:
        return None
    
    volume = np.stack(volume, axis=0)  # (D, H, W)
    volume = np.transpose(volume, (1, 2, 0))  # (H, W, D)
    volume = normalize_volume(volume)
        
    volume = np.expand_dims(volume, axis = 0)  # Add channel for Resize
    volume = resize_volume(volume, target_size)
    return volume.astype(np.float32)


def save_dataset(split_list, split_folder):
    skipped = []

    index_file = os.path.join(SAVE_DIR, f"{split_folder}_index.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Write header once
    with open(index_file, "w", newline="") as f_index:
        writer = csv.writer(f_index)
        writer.writerow(["class", "label", "filename", "folder", "patient_id"])
    for volume, label in split_list:
        pid = extract_patient_id(volume[0])  # Extract from path

        try:
            vol = load_dicom_volume(volume)
        except Exception as e:
            print(f"⚠️ Failed to load volume for {pid}: {e}")
            skipped.append(pid)
            continue

        if vol is None:
            print(f"⚠️ Skipped: volume is None for PID {pid}")
            skipped.append(pid)
            continue

        # Set output dir and save .npy file
        class_folder = "cancer" if label == 1 else "non-cancer"
        out_dir = os.path.join(SAVE_DIR, split_folder, class_folder)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{pid}_org.npy")
        np.save(out_path, vol)

        # Write to index file
        with open(index_file, "a", newline="") as f_index:
            writer = csv.writer(f_index)
            writer.writerow([class_folder, label, out_path, split_folder, pid])

    print(f"✅ Done: {split_folder} | Skipped: {len(skipped)} volumes")



def extract_patient_id(path):
    filename = os.path.basename(path)
    return filename.split("_")[0]

if __name__ == "__main__":
    class_dict = build_patient_dict(DICOM_DIR)

    splits = split_by_patient(class_dict)
    

    for idx, split in enumerate(splits):
        if idx == 0:
            save_dataset(split_list=split,split_folder='train')
        elif idx == 1:
            save_dataset(split_list=split, split_folder='val')
        else:
            save_dataset(split_list=split, split_folder='test')
