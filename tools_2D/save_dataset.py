import os
import random
import torch
from collections import defaultdict
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, RandFlip, RandRotate, RandGaussianNoise
)
from tqdm import tqdm
import csv

# --- Configuration ---
DICOM_DIR = "/home/vivianea/projects/BrainInnov/data/LIDC_classes_dcm"
SAVE_DIR = "/home/vivianea/projects/BrainInnov/data/npy_balanced_split"
IMAGE_SIZE = (512, 512)
VAL_RATIO = 0.2
TEST_RATIO = 0.2
SEED = 42
LOW_VARIANCE_THRESHOLD = 10.0 

AUG_PER_CLASS = {
    "cancer": 0,
    "non-cancer": 0
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

def has_pixel_data(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=False)
        return hasattr(ds, "PixelData")
    except Exception:
        return False
    
def is_low_variance(image_array, threshold=LOW_VARIANCE_THRESHOLD):
    # Assuming image is (C, H, W), we flatten to compute variance
    return np.var(image_array) < threshold

def is_dicom_segmentation(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        modality = getattr(ds, "Modality", "")
        sop_class_uid = getattr(ds, "SOPClassUID", "")
        if modality == "SEG":
            return True
        if sop_class_uid.startswith("1.2.840.10008.5.1.4.1.1.66"):
            return True
        return False
    except Exception:
        return False

def get_patient_id(dicom_path):
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    return ds.PatientID

def collect_by_patient(dicom_dir):
    BAD_FILES_LOG = os.path.join(SAVE_DIR, "bad_dicom_files.txt")
    os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure SAVE_DIR exists for log file
    class_dict = defaultdict(lambda: defaultdict(list))

    with open(BAD_FILES_LOG, "a") as log:   # open once per function call
        for cls in os.listdir(dicom_dir):
            cls_path = os.path.join(dicom_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for file in os.listdir(cls_path):
                if file.endswith(".dcm"):
                    path = os.path.join(cls_path, file)

                    # Skip segmentation files
                    if is_dicom_segmentation(path):
                        log.write(f"Skipped segmentation file: {path}\n")
                        continue

                    if not has_pixel_data(path):
                        log.write(f"Missing pixel data: {path}\n")
                        continue  # skip bad file

                    pid = get_patient_id(path)
                    class_dict[cls][pid].append(path)
    return class_dict

def split_patient_ids(class_dict):
    splits = {"train": [], "val": [], "test": []}
    for cls, patients in class_dict.items():
        pids = list(patients.keys())
        random.shuffle(pids)  # Shuffle here
        train_val, test = train_test_split(pids, test_size=TEST_RATIO, random_state=SEED)
        train, val = train_test_split(train_val, test_size=VAL_RATIO / (1 - TEST_RATIO), random_state=SEED)

        for split_name, split_ids in zip(["train", "val", "test"], [train, val, test]):
            for pid in split_ids:
                for path in patients[pid]:
                    splits[split_name].append((path, cls))
    return splits

def normalize_image(img):
    img = np.clip(img, -1000, 150)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return (img * 255).astype(np.uint8)


def apply_transform_and_save(samples, split_name, augment=False):
    base_split_path = os.path.join(SAVE_DIR, split_name)
    os.makedirs(base_split_path, exist_ok=True)

    index_file = os.path.join(SAVE_DIR, f"{split_name}_index.csv")
    patient_image_counter = defaultdict(int)  # Track how many images we've saved per patient

    skipped_path = os.path.join(SAVE_DIR, "skipped_files.txt")
    with open(index_file, "w", newline="") as f_index, open(skipped_path, "a") as sf:
        writer = csv.writer(f_index)
        writer.writerow(["filename", "label", "class", "patient_id"])

        for path, cls in tqdm(samples, desc=f"Processing {split_name}"):
            label = 1 if cls == "cancer" else 0
            patient_id = get_patient_id(path)
            img_index = patient_image_counter[patient_id]
            patient_image_counter[patient_id] += 1

            class_dir = os.path.join(base_split_path, cls)
            os.makedirs(class_dir, exist_ok=True)
            try:
                img = base_transform(path)
                img = normalize_image(img).astype(np.float32)
                if is_low_variance(img):
                    # print(f"Skipped low-variance image: {path}")
                    sf.write(f"Low variance: {path}\n")
                    continue
            except Exception as e:
                print(f"Skipping file {path} due to base_transform error: {e}")
                # with open(os.path.join(SAVE_DIR, "skipped_files.txt"), "a") as sf:
                    # sf.write(f"{path}\n")
                sf.write(f"Transform error: {path}\n")
                continue

            # Save original image
            filename = f"{patient_id}_{img_index}_orig.npy"
            filepath = os.path.join(class_dir, filename)
            np.save(filepath, img)
            writer.writerow([os.path.join(cls, filename), label, cls, patient_id])

            # Augmentations
            if augment:
                for j in range(AUG_PER_CLASS.get(cls, 0)):
                    try:
                        aug_img = augment_transform(img)
                    except Exception as e:
                        print(f"Skipping augmentation of file {path} due to augment_transform error: {e}")
                        continue
                    aug_filename = f"{patient_id}_{img_index}_aug{j}.npy"
                    aug_filepath = os.path.join(class_dir, aug_filename)
                    np.save(aug_filepath, aug_img)
                    writer.writerow([os.path.join(cls, aug_filename), label, cls, patient_id])


# --- Run the pipeline ---
class_dict = collect_by_patient(DICOM_DIR)
splits = split_patient_ids(class_dict)
apply_transform_and_save(splits["train"], "train", augment=True)
apply_transform_and_save(splits["val"], "val", augment=False)
apply_transform_and_save(splits["test"], "test", augment=False)
