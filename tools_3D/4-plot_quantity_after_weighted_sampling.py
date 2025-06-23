
#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Plot quantity after Weighted sampling ----------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

import os
import time
import random

import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from datetime import datetime
from tqdm import tqdm

from monai.transforms import (Compose, 
                              Resize, 
                              RandAdjustContrast, 
                              RandBiasField, 
                              RandFlip, 
                              RandAffine, 
                              RandRotate, 
                              RandZoom)

from monai.data import DataLoader, Dataset
from monai.data import SmartCacheDataset
from collections import Counter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  GPU Information --------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch is using GPU!")
    print(torch.cuda.get_device_name(0)) #prints the name of the GPU.
else:
    device = torch.device("cpu")
    print("PyTorch is using CPU.")


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Constants -------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

IMAGE_SIZE = (256, 128, 128)
BATCH_SIZE = 1

SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.2

PATH_TRAIN = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/train'
PATH_TEST = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/test'
PATH_VAL = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/val'

CSV_TRAIN = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/train_index.csv'
CSV_TEST = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/test_index.csv'
CSV_VAL = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/val_index.csv'

SAVE_DIR = "/home/vivianea/projects/BrainInnov/py_files_3D/augmented_quantity_weighted"
CLASS_MAP = {'cancer': 0, 'non-cancer': 1}
INDEX_TO_CLASS = {0: 'non-cancer', 1: 'cancer'}

NUM_AUG_PER_SAMPLE = 200

LOG_FILE = "training_log.txt"


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Preprocessing - Time----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def log_message(message):

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    with open(os.path.join(SAVE_DIR, LOG_FILE), "a") as f:
        f.write(message + "\n")


t1 = time.time()
starting_information = f"\n\n==== Sampling started at {datetime.now()} ====\n\n"
print(starting_information)
log_message(starting_information)

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Loading Dataset --------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------


class LIDCDataset3D(Dataset):
    def __init__(self, csv_path_idx, npy_dir, transform=None, seed=None):
        self.data = pd.read_csv(csv_path_idx)
        self.npy_dir = npy_dir
        self.transform = transform

        self.classes = sorted(self.data["label"].unique())
        self.class_to_idx = {0: "non-cancer", 1: "cancer"}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.seed = seed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.npy_dir, row['filename'])
        vol_image = np.load(file_path).astype(np.float32)  # e.g. (D, H, W)

        # Ensure shape is (C, D, H, W) with C=1 channel
        if vol_image.ndim == 3:
            vol_image = np.expand_dims(vol_image, axis=0)  # (1, D, H, W)

        vol_image = torch.from_numpy(vol_image)

        if self.transform:
            vol_image = self.transform(vol_image)

        label = int(row['label'])

        return vol_image, label

    
def get_transforms():
    return Compose([
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandFlip(spatial_axis=2, prob=0.5),
        RandRotate(range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5),
        RandAffine(
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.3
        ),
        RandBiasField(prob=0.3),
        RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.2),
        Resize(spatial_size=IMAGE_SIZE, mode="trilinear")
    ])

train_dataset = LIDCDataset3D(csv_path_idx=CSV_TRAIN, npy_dir=PATH_TRAIN, transform=get_transforms(), seed=SEED)
val_dataset = LIDCDataset3D(csv_path_idx=CSV_VAL, npy_dir=PATH_VAL, transform=get_transforms(), seed=SEED)
test_dataset = LIDCDataset3D(csv_path_idx=CSV_TEST, npy_dir=PATH_TEST, transform=get_transforms(), seed=SEED)

print(f"✅ Loaded: {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")


class FrozenAugmentedDataset(Dataset):
    def __init__(self, base_dataset):
        self.data = []
        for i in tqdm(range(len(base_dataset)), desc="Applying one-time augmentation"):
            x, y = base_dataset[i]
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
frozen_dataset_train = FrozenAugmentedDataset(train_dataset)
frozen_dataset_val = FrozenAugmentedDataset(val_dataset)
frozen_dataset_test = FrozenAugmentedDataset(test_dataset)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Weighted Sampler -------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

labels_train = [s[1] for s in train_dataset]  # Adjust if your dataset is not structured this way
class_counts = Counter(labels_train)
max_class_count = max(class_counts.values())
num_aug_per_sample = NUM_AUG_PER_SAMPLE

target_per_class = max_class_count * num_aug_per_sample
samples_weight = [target_per_class / class_counts[label_] for label_ in labels_train]
num_samples = len(samples_weight) * num_aug_per_sample
sampler_train = WeightedRandomSampler(samples_weight, num_samples=num_samples, replacement=True)

train_loader = DataLoader(frozen_dataset_train, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler_train, worker_init_fn=seed_worker, generator=g)

labels_val = [s[1] for s in val_dataset]  # Adjust if your dataset is not structured this way
class_counts = Counter(labels_train)
max_class_count = max(class_counts.values())
num_aug_per_sample = NUM_AUG_PER_SAMPLE

target_per_class = max_class_count * num_aug_per_sample
samples_weight = [target_per_class / class_counts[label_] for label_ in labels_val]
num_samples = len(samples_weight) * num_aug_per_sample
sampler_val = WeightedRandomSampler(samples_weight, num_samples=num_samples, replacement=True)

val_loader = DataLoader(frozen_dataset_val, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler_val, worker_init_fn=seed_worker, generator=g)

labels_test = [s[1] for s in val_dataset]
class_counts = Counter(labels_train)
max_class_count = max(class_counts.values())
num_aug_per_sample = NUM_AUG_PER_SAMPLE

target_per_class = max_class_count * num_aug_per_sample
samples_weight = [target_per_class / class_counts[label_] for label_ in labels_test]
num_samples = len(samples_weight) * num_aug_per_sample
sampler_test = WeightedRandomSampler(samples_weight, num_samples=num_samples, replacement=True)

test_loader = DataLoader(frozen_dataset_test, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler_test, worker_init_fn=seed_worker, generator=g)


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Weighted Sampler quantity samples ---------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------




# Collect labels from one full pass through the train_loader
sampled_labels = []

for i, (img, label) in enumerate(train_loader):
    sampled_labels.extend(label.cpu().numpy().astype(int))  # Convert to int (0 or 1)
    if i >= len(train_loader):  # Only loop once through the loader
        break

# Count label occurrences
label_counts = Counter(sampled_labels)

# Print result
print("Sampled label distribution from WeightedRandomSampler:")
for label in sorted(label_counts.keys()):
    print_count_labels = f"Class {label}: {label_counts[label]} samples"
    print(print_count_labels)
    log_message(print_count_labels)

# Optional: plot
plt.bar(label_counts.keys(), label_counts.values(), tick_label=["Non-cancer", "Cancer"])
plt.xlabel("Class")
plt.ylabel("Sample Count")
plt.title("Sample Distribution from WeightedRandomSampler")
filepath = os.path.join(SAVE_DIR, f"Samples_quantity.png")
plt.savefig(filepath)
plt.show()

elapsed = time.time() - t1
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)

summary_ = f"######## Training Finished in {int(hours)}h {int(minutes)}m {int(seconds)}s ###########"
print(summary_)
log_message(summary_)
