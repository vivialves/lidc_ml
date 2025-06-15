#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  CNN - Resnet - 3D CNN (Pytorch) ----------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------



import os
import time
import random
import cv2
import pydicom
import hashlib
import csv
import collections

import torch
import torchvision

import SimpleITK as sitk
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from PIL import Image

from monai.transforms import (
    Compose, Resize, LoadImaged, RepeatChanneld, ScaleIntensity, ResizeWithPadOrCropd, ToTensord,
    RandGaussianNoise, RandAdjustContrast, RandGaussianSmooth, Rand3DElasticd, RandBiasField, 
    RandCropByPosNegLabeld, Resized, RandFlip, RandAffine, Compose, Resize, RandRotate, RandZoom
)

from torch.amp import autocast, GradScaler
from monai.data import DataLoader, Dataset
from collections import Counter
from torchvision import models
from torchinfo import summary
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve



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

IMAGE_SIZE = (256, 256, 256)
BATCH_SIZE = 1
NUM_CHANNELS = 1
DEPTH = 128
NUM_CLASSES = 2
PATIENCE_COUNTER = 6
EPOCHS = 50
SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.2

DICOM_DIR = "/home/etudiant/Projets/Viviane/LIDC-ML/data/LIDC_classes_dcm"
PATH_MODEL = '/home/etudiant/Projets/Viviane/LIDC-ML/models/best_model_resnet_pytorch3D_architecture.pth'

SAVE_DIR = "/home/etudiant/Projets/Viviane/LIDC-ML/"
PATH_RESULTS = "/home/etudiant/Projets/Viviane/LIDC-ML/lidc_ml/py_files_3D/resnetPy/results"
CLASS_MAP = {'cancer': 0, 'non-cancer': 1}
INDEX_TO_CLASS = {0: 'non-cancer', 1: 'cancer'}

AUG_PER_CLASS = {"train": 0, "val": 0, "test": 0}

IMAGE_SIZE_SUMMARY = 256

NUM_AUG_PER_SAMPLE = 60

LOG_FILE = "training_log.txt"


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Preprocessing ----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

# Set seed for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

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


# Helper: Load and preprocess a 3D volume from a list of DICOM paths
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
            # print(f"Slice read failed in {path}: {e}")
            continue

    if len(volume) < min_slices:
        return None
    
    volume = np.stack(volume, axis=0)  # (D, H, W)
    volume = np.transpose(volume, (1, 2, 0))  # (H, W, D)
    volume = normalize_volume(volume)
        
    volume = np.expand_dims(volume, axis = 0)  # Add channel for Resize
    volume = resize_volume(volume, target_size)
    # volume = volume[0]  # Remove channel dim
    return volume.astype(np.float32)

# Helper: Build patient-to-path mapping
def build_patient_dict(base_dir):
    class_dict = {"cancer": {}, "non-cancer": {}}
    for cls in ["cancer", "non-cancer"]:
        cls_path = os.path.join(base_dir, cls)
        for root, _, files in os.walk(cls_path):
            for fname in files:
                if fname.endswith(".dcm"):
                    pid = fname.split("_")[0]
                    if pid not in class_dict[cls]:
                        class_dict[cls][pid] = []
                    class_dict[cls][pid].append(os.path.join(root, fname))
    return class_dict

def split_data(class_dict):
    train, val, test = [], [], []
    for label_name, pid_dict in class_dict.items():
        label = 1 if label_name == "cancer" else 0
        pids = list(pid_dict.keys())
        train_p, test_p = train_test_split(pids, test_size=TEST_RATIO, random_state=SEED)
        train_p, val_p = train_test_split(train_p, test_size=VAL_RATIO / (1 - TEST_RATIO), random_state=SEED)

        for pid in train_p:
            train.append((pid_dict[pid], label))
        for pid in val_p:
            val.append((pid_dict[pid], label))
        for pid in test_p:
            test.append((pid_dict[pid], label))

    return train, val, test


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Loading Dataset --------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------


class LIDCDataset3D(Dataset):
    def __init__(self, data_list, label_map, transform=None, seed=None):
        self.data_list = data_list  # [(paths, label), ...]
        self.label_map = label_map
        self.transform = transform

        self.class_to_idx = {0: "non-cancer", 1: "cancer"}

        self.seed = seed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.seed is not None:
            # Set a deterministic seed for this item
            seed = self.seed + idx
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        paths, label = self.data_list[idx]
        vol = load_dicom_volume(paths)
        if vol is None:
            return self.__getitem__((idx + 1) % len(self))  # fallback
        vol = torch.tensor(vol)
        if self.transform:
            vol = self.transform(vol)

        label = int(label)
        return vol, torch.tensor(label, dtype=torch.float32)
    
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

class_dict = build_patient_dict(DICOM_DIR)
train_data, val_data, test_data = split_data(class_dict)

train_dataset = LIDCDataset3D(train_data, label_map={"cancer": 1, "non-cancer": 0}, transform=get_transforms(), seed=SEED)
val_dataset = LIDCDataset3D(val_data, label_map={"cancer": 1, "non-cancer": 0}, transform=get_transforms(), seed=SEED)
test_dataset = LIDCDataset3D(test_data, label_map={"cancer": 1, "non-cancer": 0}, transform=get_transforms(), seed=SEED)

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

labels_train = [s[1] for s in train_dataset.data_list]  # Adjust if your dataset is not structured this way
class_counts = Counter(labels_train)
max_class_count = max(class_counts.values())
num_aug_per_sample = NUM_AUG_PER_SAMPLE

target_per_class = max_class_count * num_aug_per_sample
samples_weight = [target_per_class / class_counts[label_] for label_ in labels_train]
num_samples = len(samples_weight) * num_aug_per_sample
sampler_train = WeightedRandomSampler(samples_weight, num_samples=num_samples, replacement=True)

train_loader = DataLoader(frozen_dataset_train, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler_train, worker_init_fn=seed_worker, generator=g)

labels_val = [s[1] for s in val_dataset.data_list]  # Adjust if your dataset is not structured this way
class_counts = Counter(labels_train)
max_class_count = max(class_counts.values())
num_aug_per_sample = NUM_AUG_PER_SAMPLE

target_per_class = max_class_count * num_aug_per_sample
samples_weight = [target_per_class / class_counts[label_] for label_ in labels_val]
num_samples = len(samples_weight) * num_aug_per_sample
sampler_val = WeightedRandomSampler(samples_weight, num_samples=num_samples, replacement=True)

val_loader = DataLoader(frozen_dataset_val, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler_val, worker_init_fn=seed_worker, generator=g)

labels_test = [s[1] for s in val_dataset.data_list]
class_counts = Counter(labels_train)
max_class_count = max(class_counts.values())
num_aug_per_sample = NUM_AUG_PER_SAMPLE

target_per_class = max_class_count * num_aug_per_sample
samples_weight = [target_per_class / class_counts[label_] for label_ in labels_test]
num_samples = len(samples_weight) * num_aug_per_sample
sampler_test = WeightedRandomSampler(samples_weight, num_samples=num_samples, replacement=True)

test_loader = DataLoader(frozen_dataset_test, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler_test, worker_init_fn=seed_worker, generator=g)

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Size by classes in Train Dataset ---------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

classes = [cls for cls in train_dataset.class_to_idx.values()]

# Collect labels from your dataset
labels = [int(label) for _, label in train_dataset.data_list]

# Count occurrences
label_counts = Counter(labels)

# Define your index-to-class mapping manually if needed
index_to_class = INDEX_TO_CLASS  # adjust if different

# Print counts
print("\nTraining set counts:")
for idx, count in label_counts.items():
    print(f"Class: {index_to_class[idx]}, Count: {count}")

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Architecture -----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

from torchvision.models.video import r3d_18

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class Resnet3DClassifierPy(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet3DClassifierPy, self).__init__()
        self.backbone = r3d_18(weights="KINETICS400_V1")
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.attn = AttentionBlock(in_channels=512)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        self.dropout = nn.Dropout(0.5)
        x = self.backbone.layer4(x)
        x = self.attn(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        self.dropout = nn.Dropout(0.5)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Resnet3DClassifierPy() # Binary classification
model = model.to(device)  # If using GPU

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

summary(model, input_size=(BATCH_SIZE, NUM_CHANNELS, DEPTH, IMAGE_SIZE_SUMMARY, IMAGE_SIZE_SUMMARY))


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Training ---------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

def plot_augmented_volume(volume_tensor, label, index, epoch, batch_idx, save_dir="augmented_samples"):

    path_augmented = os.path.join(PATH_RESULTS, save_dir)
    os.makedirs(path_augmented, exist_ok=True)

    volume_np = volume_tensor.numpy()
    depth = volume_np.shape[1]  # Assuming [C, D, H, W]
    center_slice = depth // 2

    slice_img = volume_np[0, center_slice, :, :]  # [C, D, H, W] -> [H, W]

    plt.imshow(slice_img, cmap="gray")
    plt.title(f"Epoch {epoch+1} - Sample {index} - Label: {label.item()}")
    plt.axis("off")
    plt.tight_layout()
    filepath = os.path.join(path_augmented, f"epoch{epoch+1}_batch{batch_idx}_sample{index}_label{label.item()}.png")
    plt.savefig(filepath)
    plt.close()

def log_message(message):

    os.makedirs(PATH_RESULTS, exist_ok=True)
    
    with open(os.path.join(PATH_RESULTS, LOG_FILE), "a") as f:
        f.write(message + "\n")



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_classes, patience, path_model):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Metrics
    confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
    precision = MulticlassPrecision(num_classes=num_classes, average='macro')
    recall = MulticlassRecall(num_classes=num_classes, average='macro')
    f1_score = MulticlassF1Score(num_classes=num_classes, average='macro')

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = path_model

    scaler = GradScaler("cuda")

    # --- Gradient Accumulation Parameter ---
    # Set your desired effective batch size here.
    # For example, if your DataLoader's batch_size is 1 and you want an effective batch size of 4,
    # set accumulation_steps = 4.
    # physical_batch_size = train_loader.batch_size # Get this from your DataLoader
    # if physical_batch_size is None: # Handle cases where batch_size might be 1 by default (no explicit arg)
    #     physical_batch_size = 1
    # effective_batch_size = 4 # Or 8, 16, etc.
    # accumulation_steps = effective_batch_size // physical_batch_size
    # if accumulation_steps == 0: accumulation_steps = 1 # Ensure at least 1 step if effective_batch_size < physical_batch_size

    # A simpler way: just define the number of steps directly
    accumulation_steps = 4 # Example: Accumulate gradients over 4 mini-batches
    
    starting_information = (f"\n\n==== Training started at {datetime.now()} ====\n\n"
                            f"Using Gradient Accumulation with {accumulation_steps} steps.\n"
                            f"DataLoader batch size: {train_loader.batch_size}\n"
                            f"Effective batch size: {train_loader.batch_size * accumulation_steps if train_loader.batch_size else 'N/A (check DataLoader)'}\n")
    print(starting_information)
    log_message(starting_information)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        epoch_hashes = set()
        epoch_label_counter = Counter()

        confusion_matrix.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = torch.as_tensor(labels).to(device).long().view(-1)

            epoch_label_counter.update(labels.cpu().tolist())

            # Clear gradients
            # optimizer.zero_grad()

            with torch.no_grad():
                for img in inputs.cpu():
                    img_hash = hashlib.sha1(img.numpy().tobytes()).hexdigest()
                    epoch_hashes.add(img_hash)

            max_batches_to_save = 4  # How many batches you want to save from epoch 0
            max_samples_per_batch = 2  # How many samples per batch to save

            if epoch == 0 and batch_idx < max_batches_to_save:
                with torch.no_grad():
                    for i in range(min(max_samples_per_batch, inputs.size(0))):
                        sample_idx = batch_idx * max_samples_per_batch + i
                        plot_augmented_volume(inputs[i].cpu(), labels[i].cpu(), sample_idx, epoch, batch_idx)

            with autocast("cuda"):
            # feed foward
                outputs = model(inputs)

                # Compute loss using cross entropy
                # Assuming your criterion is suitable for outputs and labels shapes
                # For BCEWithLogitsLoss with 2 classes, outputs might be [B, 1] and labels [B, 1] (float)
                # Or for CrossEntropyLoss, outputs [B, num_classes] and labels [B] (long)
                # Your current line labels = labels.float().unsqueeze(1) suggests binary classification and outputs are also [B,1]
                # labels_for_loss = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float() # Ensure labels match output shape
                # loss = criterion(outputs, labels_for_loss)
                

                labels_for_loss = labels.view(-1).long() if labels.ndim > 1 else labels.long()
                loss = criterion(outputs, labels_for_loss)
            
                # Compute loss using cross entropy
                # labels = labels.float().unsqueeze(1)  # shape [B] -> [B, 1]
                # loss = criterion(outputs, labels)
            
            
            scaler.scale(loss).backward() # Scale loss and perform backward pass  # backpropagation
            
            # --- Gradient Accumulation: Step optimizer and update scaler only after N steps ---
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)        # Unscale gradients and perform optimizer step
                scaler.update()               # Update the scaler for the next iteration
                optimizer.zero_grad()

            # backpropagation
            # loss.backward()

            # Update weights
            # optimizer.step()

            # Update results
            train_loss += loss.item()

            if num_classes > 1: # Multiclass
                _, preds = torch.max(outputs, 1)
            else: # Binary
                preds = (outputs > 0).long().squeeze(1)

            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        if (batch_idx + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        # Evaluation part to print metrics for each epoch
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = torch.as_tensor(labels).to(device).long().view(-1)
                with autocast("cuda"):
                    outputs = model(inputs)
                    labels = labels.view(-1).long() if labels.ndim > 1 else labels.long()
                    loss = criterion(outputs, labels)
                val_loss += loss.item()


                if num_classes > 1:
                    _, preds = torch.max(outputs, 1)
                else:
                    preds = (outputs > 0).long().squeeze(1)

                    
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

                # For metrics
                all_preds.append(preds.cpu())
                all_labels.append(labels.long().cpu())

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct_val / total_val)

        
        # all_preds = torch.cat(all_preds)
        # all_labels = torch.cat(all_labels)

        all_preds = torch.cat(all_preds).view(-1)
        all_labels = torch.cat(all_labels).view(-1)

        
        precision_value = precision(all_preds, all_labels)
        recall_value = recall(all_preds, all_labels)
        f1_value = f1_score(all_preds, all_labels)
        conf_matrix = confusion_matrix(all_preds, all_labels)

        # all_preds_final = torch.as_tensor(all_preds, device=device)
        # all_labels_final = torch.as_tensor(all_labels, device=device)

        # precision_value = precision(all_preds_final, all_labels_final)
        # recall_value = recall(all_preds_final, all_labels_final)
        # f1_value = f1_score(all_preds_final, all_labels_final)
        # conf_matrix = confusion_matrix(all_preds_final, all_labels_final)

        
        epoch_log = (
            f"Epoch {epoch + 1}/{num_epochs}\n"
            f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.2f}%\n"
            f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]:.2f}%\n"
            f"Precision: {precision_value:.4f} | Recall: {recall_value:.4f} | F1 Score: {f1_value:.4f}\n"
            f"Current AMP scale: {scaler.get_scale()}\n"
            f"Unique augmented volumes seen in epoch {epoch + 1}: {len(epoch_hashes)}\n"
            f"Label distribution in training epoch: {epoch_label_counter}\n"
        )
        print(epoch_log)
        log_message(epoch_log)
        
        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reset patience counter
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'confusion_matrix': conf_matrix.cpu().numpy(), # Save the confusion matrix of the best model
                'scaler_state_dict': scaler.state_dict()
                }, best_model_path)
            print(f"Validation loss improved. Saving best model to {best_model_path}\n")
            log_message(f"Validation loss improved. Saving best model to {best_model_path}\n")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}\n")
            log_message(f"Validation loss did not improve. Patience: {patience_counter}/{patience}\n")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss for {patience} consecutive epochs.\n")
            log_message(f"Early stopping triggered after {epoch+1} epochs.\n")
            break # Exit the training loop

    print("\nTraining complete.")
    # Load the best model after training is complete (either by early stopping or max epochs)
    print(f"Loading best model from {best_model_path} for final metrics.")
    log_message("\nTraining complete.")
    log_message(f"Loading best model from {best_model_path} for final metrics.")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # You might not need to load optimizer state if you're just doing inference or final evaluation
    # --- Optional: Load optimizer state if resuming training ---
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Load optimizer state if resuming
    # --- Handle scaler state dict loading ---
    if 'scaler_state_dict' in checkpoint: # Check if the key exists
        scaler.load_state_dict(checkpoint['scaler_state_dict']) # --- AMP: Load scaler state ---
    else:
        print("Warning: 'scaler_state_dict' not found in checkpoint. This may be an older model or AMP was not used.")    
    final_conf_matrix = checkpoint['confusion_matrix'] # Retrieve the confusion matrix of the best model

    return train_losses, val_losses, train_accuracies, val_accuracies, conf_matrix, final_conf_matrix


t1 = time.time()
print(f"Using device: {device}")
train_losses, val_losses, train_accuracies, val_accuracies, conf_matrix, final_conf_matrix = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, num_classes=NUM_CLASSES, patience=PATIENCE_COUNTER, path_model=PATH_MODEL)

elapsed = time.time() - t1
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)

summary_ = f"######## Training Finished in {int(hours)}h {int(minutes)}m {int(seconds)}s ###########"
print(summary_)
log_message(summary_)


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Evaluating ---------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

#####################################################  Accuracy ###################################################################

model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)  # shape: [N, 1]
        probs = torch.sigmoid(outputs)  # Convert logits to probabilities
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.view(-1)  # Ensure shape match with predicted
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = f'Test Accuracy on {total} images: {(correct / total) * 100:.2f}%'
    print(accuracy)
    log_message(accuracy)


#####################################################  LOSS ###################################################################


def plot_training(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    filepath = os.path.join(PATH_RESULTS, f"Loss_Accuracy.png")
    plt.savefig(filepath)
    plt.show()

plot_training(train_losses, val_losses, train_accuracies, val_accuracies)

#####################################################  Confusion Matrix ###################################################################

cm = conf_matrix.cpu().numpy()
disp = ConfusionMatrixDisplay(cm, display_labels=classes,)
disp.plot()

plt.title("Confusion Matrix")
filepath = os.path.join(PATH_RESULTS, f"confusion_matrix.png")
plt.savefig(filepath, bbox_inches='tight')
plt.close()

y_true = []
y_probs = []  # Collect probabilities for class 1 (cancer)

#####################################################  ROC Curve ###################################################################

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:  # or test_loader
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # raw logits or softmax
        probs = torch.softmax(outputs, dim=1)[:, 1]

        y_true.extend(labels.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

# Compute AUC
auc = roc_auc_score(y_true, y_probs)

auc_score = f"AUC: {auc:.4f}"
print(auc_score)
log_message(auc_score)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
filepath = os.path.join(PATH_RESULTS, f"ROC_Curve.png")
plt.savefig(filepath)
plt.show()


#####################################################  Metrics ###################################################################

precision = MulticlassPrecision(num_classes=NUM_CLASSES, average=None)
recall = MulticlassRecall(num_classes=NUM_CLASSES, average=None)
f1_score = MulticlassF1Score(num_classes=NUM_CLASSES, average=None)


all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())


all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)


per_class_precision = precision(all_preds, all_labels)
per_class_recall = recall(all_preds, all_labels)
per_class_f1 = f1_score(all_preds, all_labels)


for i, name in enumerate(classes):
    metrics = f"Class {i}-{name}: Precision: {per_class_precision[i]:.2f}, Recall: {per_class_recall[i]:.2f}, F1-Score: {per_class_f1[i]:.2f}"
    print(metrics)
    log_message(metrics)


#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  GRAD CAM ---------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

try:
    # 2. Instantiate your model class
    # Pass the SAME in_channels and out_channels that you used during training.
    model = Resnet3DClassifierPy()
    print(f"Instantiated Resnet3D-Pytorch")

    # 3. Load the state_dict
    state_dict = torch.load(PATH_MODEL)
    print(f"Loaded state_dict from {PATH_MODEL}")

    # *** IMPORTANT CHECK: Handle 'module.' prefix if you used nn.DataParallel for training ***
    # If your model was trained using `nn.DataParallel`, the keys in the state_dict
    # will have a 'module.' prefix (e.g., 'module.model.features.0.weight').
    # You need to remove this prefix when loading into a single-GPU/CPU model.
    if list(state_dict.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from state_dict keys for DataParallel compatibility...")
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' prefix
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print("No 'module.' prefix found. Loading state_dict directly.")
        model.load_state_dict(state_dict)

    # 4. Set the model to evaluation mode
    model.eval()

    print("Model loaded successfully and set to evaluation mode.")

    # Now 'model' is a proper torch.nn.Module instance with loaded weights.
    # You can proceed with your Grad-CAM implementation.

except FileNotFoundError:
    print(f"Error: Model file not found at {PATH_MODEL}. Please double-check the path.")
except RuntimeError as e:
    print(f"Runtime Error during model loading (likely mismatch in model architecture or state_dict keys): {e}")
    print("Please ensure the `in_channels` and `out_channels` passed to DenseNet3DClassifier match what was used during training.")
    print("Also, confirm if the saved model was trained with `nn.DataParallel` (leading to 'module.' prefixes).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


def replace_relu_with_non_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU) and child.inplace:
            setattr(module, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu_with_non_inplace(child)

replace_relu_with_non_inplace(model)

import scipy.ndimage

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            print("Backward hook triggered")
            self.gradients = grad_out[0].detach().clone()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dim if missing
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad = True

        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output)

        loss = output[0, target_class]
        loss.backward()

        if self.gradients is None:
            raise RuntimeError("Backward hook did not capture gradients.")

        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()

        return cam, target_class

    def visualize(self, image_tensor, cam, predicted_class, lab):
    

        image_np = image_tensor.squeeze().cpu().numpy()
        cam_np = cam  # Already a NumPy array

        # Resize CAM to match input shape
        cam_resized = scipy.ndimage.zoom(
            cam_np,
            zoom=(
                image_np.shape[0] / cam_np.shape[0],
                image_np.shape[1] / cam_np.shape[1],
                image_np.shape[2] / cam_np.shape[2],
            ),
            order=1,
        )

        center = image_np.shape[0] // 2
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image_np[center], cmap='gray')
        ax[0].set_title("Original Slice")
        ax[1].imshow(image_np[center], cmap='gray')
        ax[1].imshow(cam_resized[center], cmap='jet', alpha=0.5)
        ax[1].set_title("Grad-CAM Overlay")
        plt.tight_layout()
        filepath = os.path.join(PATH_RESULTS, f"gradcam{predicted_class}-{lab}.png")
        plt.savefig(filepath)
        plt.show()

#####################################################  GRADCAM Cancer ###################################################################

# Choose target layer (last conv in ResNet_3D Pytorch CNN)
target_layer = model.backbone.layer4[-1]

# Initialize GradCAM
grad_cam = GradCAM3D(model, target_layer)

# Run on one sample
image, label = test_dataset[0]  # image should be a tensor
cam, predicted_class = grad_cam.generate(image)

# Show visualization
grad_cam.visualize(image, cam, predicted_class, lab='cancer')

#####################################################  GRADCAM Non-Cancer ###################################################################

# Choose target layer (last conv in ResNet_3D Pytorch CNN)
target_layer = model.backbone.layer4[-1]

# Initialize GradCAM
grad_cam = GradCAM3D(model, target_layer)

# Run on one sample
image, label = test_dataset[19]  # image should be a tensor
cam, predicted_class = grad_cam.generate(image)

# Show visualization
grad_cam.visualize(image, cam, predicted_class, lab='non-cancer')