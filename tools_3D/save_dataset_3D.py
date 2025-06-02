import os
import random
from collections import defaultdict
import numpy as np
import pydicom
import SimpleITK as sitk
from monai.transforms import (
    RandGaussianNoised, RandAdjustContrastd, RandGaussianSmoothd,
    Rand3DElasticd, RandBiasFieldd, RandCropByPosNegLabeld, Resized, 
    RandFlipd, RandAffined, Compose, Resize
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

# Constants
DICOM_DIR = "/home/vivianea/projects/BrainInnov/data/LIDC_classes_dcm"
SAVE_DIR = "/home/vivianea/projects/BrainInnov/data/npy3d_split"
AUG_PER_CLASS = {"train": 3, "val": 3, "test": 3}
SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.2
IMAGE_SIZE = (224, 224, 224)

random.seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# Augmentation pipeline
augment_transform = Compose([
    # RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),  # Flip in all directions
    RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image"], spatial_axis=[2], prob=0.5),
    RandAffined(
        keys=["image"],
        rotate_range=(0.1, 0.1, 0.1),
        shear_range=(0.05, 0.05, 0.05),
        translate_range=(5, 5, 5),
        scale_range=(0.1, 0.1, 0.1),
        prob=0.3,
        spatial_size=IMAGE_SIZE,
        padding_mode="border"
    ),
    RandBiasFieldd(keys=["image"], prob=0.3),  # Simulate MR bias fields
    RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),  # Add noise
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),  # Vary contrast
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
    Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode="trilinear")
])

def is_dicom_segmentation(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return hasattr(ds, "Modality") and ds.Modality == "SEG"
    except Exception:
        return False

def has_pixel_data(path):
    try:
        ds = pydicom.dcmread(path)
        return hasattr(ds, "pixel_array")
    except Exception:
        return False

def get_patient_id_from_filename(filename):
    # Assuming filename is like "LIDC-IDRI-0068_1-001.dcm"
    return filename.split("_")[0]  # Returns "LIDC-IDRI-0068"

def collect_by_patient(dicom_dir):
    class_dict = defaultdict(lambda: defaultdict(list))
    log_file = os.path.join(SAVE_DIR, "bad_dicom_files.txt")
    skipped_patient_ids = set()

    with open(log_file, "w") as log:
        for cls in os.listdir(dicom_dir):
            cls_path = os.path.join(dicom_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            for root, _, files in os.walk(cls_path):
                for fname in files:
                    if not fname.endswith(".dcm"):
                        continue

                    path = os.path.join(root, fname)

                    if is_dicom_segmentation(path):
                        log.write(f"Skipped segmentation file: {path}\n")
                        continue
                    if not has_pixel_data(path):
                        log.write(f"Missing pixel data: {path}\n")
                        continue

                    try:
                        pid = get_patient_id_from_filename(fname)
                        class_dict[cls][pid].append(path)
                    except Exception as e:
                        log.write(f"Bad DICOM: {path} | Error: {str(e)}\n")

        # Filter out patients with too few slices
        for cls in list(class_dict.keys()):
            for pid in list(class_dict[cls].keys()):
                if len(class_dict[cls][pid]) < 3:
                    log.write(f"Skipping patient {pid} (class: {cls}) — only {len(class_dict[cls][pid])} slices\n")
                    skipped_patient_ids.add(pid)
                    del class_dict[cls][pid]

    print(f"✅ Patient grouping complete.")
    print(f"🧠 Classes: {list(class_dict.keys())}")
    print(f"📦 Total patients: {sum(len(pids) for pids in class_dict.values())}")
    print(f"⚠️ Skipped patients with < 3 slices: {len(skipped_patient_ids)}")

    return class_dict

def split_patient_ids(class_dict, test_ratio=0.2, val_ratio=0.1, seed=42):
    split_dict = {}

    for label, patient_ids in class_dict.items():
        patient_ids = list(patient_ids)
        total = len(patient_ids)
        print(f"\nLabel '{label}' has {total} patients")

        if total < 3:
            print(f"⚠️ Not enough patients to split '{label}' — assigning all to training.")
            split_dict[label] = {
                'train': patient_ids,
                'val': [],
                'test': []
            }
            continue

        try:
            # Split into train_val and test
            train_val_pids, test_pids = train_test_split(
                patient_ids,
                test_size=test_ratio,
                random_state=seed
            )
            print(f"  - After test split: train_val={len(train_val_pids)}, test={len(test_pids)}")

            # Split train_val into train and val
            val_ratio_adjusted = val_ratio / (1 - test_ratio)
            if len(train_val_pids) < 2:
                print(f"⚠️ Too few patients to further split train_val into train and val — skipping val split.")
                train_pids, val_pids = train_val_pids, []
            else:
                train_pids, val_pids = train_test_split(
                    train_val_pids,
                    test_size=val_ratio_adjusted,
                    random_state=seed
                )

            print(f"  - Final splits: train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")

            split_dict[label] = {
                'train': train_pids,
                'val': val_pids,
                'test': test_pids
            }

        except ValueError as e:
            print(f"❌ Split failed for label '{label}': {e}")
            print("Assigning all patients to training set.")
            split_dict[label] = {
                'train': patient_ids,
                'val': [],
                'test': []
            }

    return split_dict

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
    volume = np.clip(volume, -1000, 400)
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    else:
        volume = np.zeros_like(volume)
    return volume.astype(np.float32)

def load_dicom_volume(paths, target_shape=IMAGE_SIZE):
    try:
        slices = []
        for path in paths:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "InstanceNumber"):
                slices.append((ds.InstanceNumber, path))

        if len(slices) < 3:
            return None

        slices.sort(key=lambda x: x[0])
        sorted_paths = [p for _, p in slices]

        arrays = []
        with open(os.path.join(SAVE_DIR, "Slice_shape.txt"), "a") as f:
            for path in sorted_paths:
                img = sitk.ReadImage(path)
                arr = sitk.GetArrayFromImage(img)[0]  # Extract 2D slice
                expected_shape = (512, 512)
                if arr.shape == expected_shape:
                    shape_info = f"Slice shape from {path}: {arr.shape}"
                    f.write(shape_info + "\n")
                    arrays.append(arr)

        volume = np.stack(arrays, axis=0)  # (D, H, W)
        volume = np.transpose(volume, (1, 2, 0))  # (H, W, D)

        volume = normalize_volume(volume)
        
        volume = np.expand_dims(volume, axis = 0)  # Add channel for Resize
        volume = resize_volume(volume, target_shape)
        volume = volume[0]  # Remove channel dim
        
        return volume.astype(np.float32)
    except Exception as e:
        print(f"❌ load_dicom_volume error: {e}")
        return None

def apply_transform_and_save(split_cls, split_list, class_dict_path, augment=False):
    skipped = []
    split_data = []
    for name_folder, list_pid in split_list.items():
        label = 1 if split_cls == "cancer" else 0
        index_file = os.path.join(SAVE_DIR, f"{name_folder}_index.csv")
        with open(index_file, "a", newline="") as f_index:
            writer = csv.writer(f_index)
            if label == 0:
                writer.writerow(["class", "label","filename", "folder", "patient_id"])
            for pid in tqdm(list_pid, desc=f"Processing = {split_cls} - {name_folder}"):
                if pid in class_dict_path[split_cls]:
                    paths = class_dict_path[split_cls][pid]
                    split_data.append((paths, split_cls, pid))
                else:
                    print(f"⚠️ Warning: PID {pid} not found in class_dict[{split_cls}]")
                    continue

                paths = class_dict_path[split_cls][pid]
                vol = load_dicom_volume(paths)
                if vol is None:
                    skipped.append(pid)
                    continue

                # Build output folder path
                out_dir = os.path.join(SAVE_DIR, name_folder, split_cls)
                os.makedirs(out_dir, exist_ok=True)

                # Save the main volume
                out_path = os.path.join(out_dir, f"{pid}_org.npy")
                np.save(out_path, vol)
                f_index.write(f"{split_cls},{label},{out_path},{name_folder},{pid}\n")

                # Save augmentations if needed
                if augment and AUG_PER_CLASS[name_folder] > 0:
                    for i in range(AUG_PER_CLASS[name_folder]):
                        vol = vol.astype(np.float32)
                        if vol.ndim == 3:
                            vol = vol[np.newaxis, ...]  # (C, D, H, W)
                        elif vol.ndim == 4:
                            pass  # already correct
                        else:
                            raise ValueError(f"Unexpected volume shape: {vol.shape}")
                        # print(vol.shape)
                        aug = augment_transform({"image": vol})["image"]
                        aug_path = os.path.join(out_dir, f"{pid}_aug{i+1}.npy")
                        np.save(aug_path, aug)
                        f_index.write(f"{split_cls},{label},{aug_path},{name_folder},{pid}\n")

        # Save skipped list
        skipped_path = os.path.join(SAVE_DIR, f"skipped_{split_cls}_{name_folder}.txt")
        with open(skipped_path, "w") as f:
            for pid in skipped:
                f.write(pid + "\n")


if __name__ == "__main__":
    class_dict = collect_by_patient(DICOM_DIR)

    splits = split_patient_ids(class_dict)

    for split_cls, split_list_pid in splits.items():
        apply_transform_and_save(split_cls, split_list_pid, class_dict, augment=True)
