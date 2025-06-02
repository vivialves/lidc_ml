import os
import random
from collections import defaultdict
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, RandFlip, RandRotate, RandGaussianNoise
)
from tqdm import tqdm

# --- Configuration ---
DICOM_DIR = "/home/vivianea/projects/BrainInnov/data/LIDC_classes_dcm"
SAVE_DIR = "/home/vivianea/projects/BrainInnov/data/npy_balanced"
IMAGE_SIZE = (512, 512)
VAL_RATIO = 0.2
TEST_RATIO = 0.2
SEED = 42

AUG_PER_CLASS = {
    "cancer": 1,       # limit augmentations for cancer (majority)
    "non-cancer": 2    # allow more for minority class
}

random.seed(SEED)
np.random.seed(SEED)

# --- Transforms ---
base_transform = Compose([
    LoadImage(image_only=True, reader="PydicomReader"),
    EnsureChannelFirst(),
    Resize(spatial_size=IMAGE_SIZE)
])

augment_transform = Compose([
    RandFlip(prob=0.5, spatial_axis=0),
    RandRotate(range_x=0.2, prob=0.5),
    RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
    Resize(spatial_size=IMAGE_SIZE)
])

# --- Helper functions ---
def get_patient_id(dicom_path):
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    return ds.PatientID

def collect_by_patient(dicom_dir):
    class_dict = defaultdict(lambda: defaultdict(list))
    for cls in os.listdir(dicom_dir):
        cls_path = os.path.join(dicom_dir, cls)
        for file in os.listdir(cls_path):
            if file.endswith(".dcm"):
                path = os.path.join(cls_path, file)
                pid = get_patient_id(path)
                class_dict[cls][pid].append(path)
    return class_dict

def split_patient_ids(class_dict):
    splits = {"train": [], "val": [], "test": []}
    for cls, patients in class_dict.items():
        pids = list(patients.keys())
        random.shuffle(pids)  # Shuffle patient IDs before splitting
        train_val, test = train_test_split(pids, test_size=TEST_RATIO, random_state=SEED)
        train, val = train_test_split(train_val, test_size=VAL_RATIO / (1 - TEST_RATIO), random_state=SEED)

        for split_name, split_ids in zip(["train", "val", "test"], [train, val, test]):
            for pid in split_ids:
                for path in patients[pid]:
                    splits[split_name].append((path, cls))
    return splits

def apply_transform_and_save(samples, split_name, augment=False):
    images, labels = [], []
    for path, cls in tqdm(samples, desc=f"Processing {split_name}"):
        img = base_transform(path)
        images.append(img.numpy())
        labels.append(1 if cls == "cancer" else 0)

        if augment:
            for _ in range(AUG_PER_CLASS.get(cls, 0)):
                aug_img = augment_transform(img)
                images.append(aug_img.numpy())
                labels.append(1 if cls == "cancer" else 0)

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Shuffle before saving
    images, labels = shuffle(images, labels, random_state=SEED)
    np.savez_compressed(
        os.path.join(SAVE_DIR, f"{split_name}.npz"),
        X=np.array(images),
        y=np.array(labels)
    )

# --- Run the pipeline ---
class_dict = collect_by_patient(DICOM_DIR)
splits = split_patient_ids(class_dict)
apply_transform_and_save(splits["train"], "train", augment=True)
apply_transform_and_save(splits["val"], "val", augment=False)
apply_transform_and_save(splits["test"], "test", augment=False)