#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Save Numpy Dataset from Dicom ------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

import os
import csv
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm

from monai.transforms import (
    Compose, Resize, LoadImaged, RepeatChanneld, ScaleIntensity, ResizeWithPadOrCropd, ToTensord,
    RandGaussianNoise, RandAdjustContrast, RandGaussianSmooth, Rand3DElasticd, RandBiasField, 
    RandCropByPosNegLabeld, Resized, RandFlip, RandAffine, Compose, Resize, RandRotate, RandZoom
)


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Constants -------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------


IMAGE_SIZE = (256, 256, 256)

PATH_TRAIN = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/train'
PATH_TEST = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/test'
PATH_VAL = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/val'

CSV_TRAIN = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/train_index.csv'
CSV_TEST = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/test_index.csv'
CSV_VAL = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/val_index.csv'

SAVE_DIR = "/home/vivianea/projects/BrainInnov/data/npy_3D_augmented"

NUM_AUGMENTATION = 5

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------  Processing -------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def get_transforms():
    return Compose([
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandFlip(spatial_axis=2, prob=0.5),
        RandRotate(range_x=0.2, range_y=0.2, range_z=0.2, prob=0.4),
        RandAffine(
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(4, 4, 4),  # slightly reduced to avoid artifacts
            scale_range=(0.05, 0.05, 0.05),  # less scale distortion
            prob=0.3
        ),
        RandBiasField(coeff_range=(0.0, 0.05), prob=0.2),
        RandAdjustContrast(prob=0.2, gamma=(0.8, 1.3)),
        RandZoom(min_zoom=0.95, max_zoom=1.05, prob=0.2),
        Resize(spatial_size=IMAGE_SIZE, mode="trilinear")
    ])


def apply_augmentation(npy_dir, csv_path, num_aug, split_folder):
    df = pd.read_csv(csv_path)
    transform = get_transforms()
    os.makedirs(SAVE_DIR, exist_ok=True)

    # CSV to track saved files
    index_file = os.path.join(SAVE_DIR, f"{split_folder}_index.csv")
    with open(index_file, "w", newline="") as f_index:
        writer = csv.writer(f_index)
        writer.writerow(["class", "label", "filename", "folder", "patient_id"])

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting {split_folder}"):
        label = int(row["label"])
        class_name = "cancer" if label == 1 else "non-cancer"
        file_path = os.path.join(npy_dir, class_name, os.path.basename(row["filename"]))
        pid = row["patient_id"]

        try:
            volume = np.load(file_path).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Skipping {pid}: {e}")
            continue

        if volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)  # (1, D, H, W)

        volume_tensor = torch.from_numpy(volume)

        # Save original version (optional)
        out_dir = os.path.join(SAVE_DIR, split_folder, class_name)
        os.makedirs(out_dir, exist_ok=True)

        orig_path = os.path.join(out_dir, f"{pid}_org.npy")
        np.save(orig_path, volume)
        with open(index_file, "a", newline="") as f_index:
            writer = csv.writer(f_index)
            writer.writerow([class_name, label, orig_path, split_folder, pid])

        # Save augmented versions
        for n in range(num_aug):
            augmented = transform(volume_tensor.clone())
            aug_np = augmented.numpy()
            aug_path = os.path.join(out_dir, f"{pid}_aug{n+1}.npy")
            np.save(aug_path, aug_np)

            with open(index_file, "a", newline="") as f_index:
                writer = csv.writer(f_index)
                writer.writerow([class_name, label, aug_path, split_folder, f"{pid}_aug{n+1}"])

    print(f"✅ Done with {split_folder}. All augmented files saved.")

# ---- EXECUTION ----
if __name__ == "__main__":
    splits = [
        ("train", PATH_TRAIN, CSV_TRAIN),
        ("val", PATH_VAL, CSV_VAL),
        ("test", PATH_TEST, CSV_TEST),
    ]

    for split_folder, npy_dir, csv_path in splits:
        apply_augmentation(npy_dir, csv_path, NUM_AUGMENTATION, split_folder)






