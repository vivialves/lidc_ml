import os
import time
import random
import logging

import torch
import torch.nn as nn

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
from monai.data import CacheDataset
from collections import Counter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import plotly.io as pio


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

IMAGE_SIZE = (64, 64, 64)
BATCH_SIZE = 1
NUM_CLASSES = 2

SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.2

PATH_TRAIN = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/train'
PATH_TEST = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/test'
PATH_VAL = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/val'

CSV_TRAIN = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/train_index.csv'
CSV_TEST = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/test_index.csv'
CSV_VAL = '/home/vivianea/projects/BrainInnov/data/npy_3D_splitted/val_index.csv'

SAVE_DIR = "/home/vivianea/projects/BrainInnov/py_files_3D/3D-architecture/optuna"
CLASS_MAP = {'cancer': 0, 'non-cancer': 1}
INDEX_TO_CLASS = {0: 'non-cancer', 1: 'cancer'}

NUM_AUG_PER_SAMPLE = 1

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
        # RandRotate(range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5),
        # RandAffine(
        #    rotate_range=(0.1, 0.1, 0.1),
        #    translate_range=(5, 5, 5),
        #    scale_range=(0.1, 0.1, 0.1),
        #    prob=0.3
        #),
        # RandBiasField(prob=0.3),
        # RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),
        # RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.2),
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
#-------------------------------------------------  Architecture -----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

class SeAttBlock(nn.Module):
    def __init__(self, in_channels, trial: optuna.trial.Trial):
        super(SeAttBlock, self).__init__()
        # Tune the reduction ratio for the attention block
        # A smaller ratio means more channels in the intermediate layer,
        # potentially more expressive but also more parameters.
        # Common ratios are powers of 2.
        reduction_ratio = trial.suggest_categorical('attention_reduction_ratio', [2, 4, 8, 16])
        
        self.conv1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn


class ResidualSEBlock(nn.Module):
    # Pass trial to ResidualSeAttBlock as it contains an SeAttBlock
    def __init__(self, in_channels, out_channels_suggested, trial: optuna.trial.Trial, block_idx: int):
        super().__init__()
        # Use out_channels_suggested as the actual out_channels
        self.out_channels = out_channels_suggested

        self.conv1 = nn.Conv3d(in_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(self.out_channels)
        
        # Pass trial to SeAttBlock
        self.se = SeAttBlock(self.out_channels, trial)
        
        # Residual connection
        self.residual = nn.Conv3d(in_channels, self.out_channels, kernel_size=1) if in_channels != self.out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


class SE3DCNN(nn.Module):
    def __init__(self, trial: optuna.trial.Trial, num_classes=NUM_CLASSES):
        super().__init__()
        
        # --- Tune channel sizes for ResidualSEBlocks ---
        # Define ranges for channel sizes. They are usually powers of 2.
        # Make sure they increase or are reasonable for your input size.
        channels_l1 = trial.suggest_categorical('channels_layer1', [16, 32, 48])
        channels_l2 = trial.suggest_categorical('channels_layer2', [32, 64, 96])
        channels_l3 = trial.suggest_categorical('channels_layer3', [64, 128, 192])

        # Ensure that the in_channels for subsequent layers matches the out_channels of the previous
        self.layer1 = ResidualSEBlock(1, channels_l1, trial, block_idx=1)
        self.pool1 = nn.MaxPool3d(2)
        # Tune Dropout3d rates
        dropout3d_rate1 = trial.suggest_float('dropout3d_rate1', 0.1, 0.4, step=0.1)
        self.drop1 = nn.Dropout3d(dropout3d_rate1)

        self.layer2 = ResidualSEBlock(channels_l1, channels_l2, trial, block_idx=2)
        self.pool2 = nn.MaxPool3d(2)
        dropout3d_rate2 = trial.suggest_float('dropout3d_rate2', 0.1, 0.4, step=0.1)
        self.drop2 = nn.Dropout3d(dropout3d_rate2)

        self.layer3 = ResidualSEBlock(channels_l2, channels_l3, trial, block_idx=3)
        self.pool3 = nn.AdaptiveAvgPool3d(1) # Global average pooling, output size 1x1x1

        # --- Tune classifier hidden layer size ---
        # The input to the first linear layer will be channels_l3 (after global average pooling)
        classifier_hidden_size = trial.suggest_categorical('classifier_hidden_size', [32, 64, 128])
        
        # Tune dropout rate for the classifier
        classifier_dropout_rate = trial.suggest_float('classifier_dropout_rate', 0.2, 0.5, step=0.1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels_l3, classifier_hidden_size), # Input features based on last block's out_channels
            nn.ReLU(),
            nn.Dropout(classifier_dropout_rate),
            nn.Linear(classifier_hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.drop1(self.pool1(self.layer1(x)))
        x = self.drop2(self.pool2(self.layer2(x)))
        x = self.pool3(self.layer3(x)) # pool3 comes AFTER layer3, correctly applied
        x = self.classifier(x)
        return x

#---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------  Architecture -----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------


def objective(trial):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Hyperparameters to tune for training ---
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    # Batch sizes for 3D are often small due to VRAM limitations
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8]) 
    epochs = trial.suggest_int('epochs', 5, 20) # Keep low for initial testing
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    gradient_clipping_norm = trial.suggest_float('gradient_clipping_norm', 0.1, 5.0, step=0.1)

    # Get data loaders
    # train_loader, val_loader = get_data_loaders(batch_size, IMAGE_SIZE)

    # Initialize model with suggested hyperparameters
    model = SE3DCNN(
        trial=trial,
        num_classes=NUM_CLASSES # Fixed for your LIDC classification task
    ).to(device)

    # Initialize optimizer
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss function (CrossEntropyLoss is standard for multi-class classification from logits)
    criterion = nn.CrossEntropyLoss().to(device) 

    best_accuracy = 0.0

    print(f"Trial {trial.number} starting with params: {trial.params}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data) 
            
            loss = criterion(output, target)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        val_loss /= len(val_loader.dataset)
        accuracy = correct / total

        print(f"Trial {trial.number}, Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy={accuracy:.4f}")

        # Optuna Pruning
        trial.report(accuracy, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()

    return accuracy

# --- Run Optuna Study ---
if __name__ == '__main__':
    # --- IMPORTANT: Ensure your data directories and CSVs exist ---
    # You need to have 'data' directory with 'train.csv', 'val.csv'
    # and 'train_images_npy', 'val_images_npy' folders.
    # If not, create some dummy files for testing similar to the previous example,
    # but saving .npy files directly.
    
    # Example for creating dummy .npy data files and CSVs (remove for real data)
    # import pandas as pd
    # if not os.path.exists(PATH_TRAIN):
    #     os.makedirs(PATH_TRAIN)
    #     os.makedirs(PATH_VAL)
    #     # Create dummy train data
    #     train_data_list = []
    #     for i in range(20): # 20 dummy train samples
    #         filename = f'vol_{i:03d}.npy'
    #         dummy_img = np.random.rand(*IMAGE_SIZE).astype(np.float32)
    #         label = np.random.randint(0, 2) # Randomly assign 0 or 1
    #         np.save(os.path.join(PATH_TRAIN, filename), dummy_img)
    #         train_data_list.append({'filename': filename, 'label': label})
    #     pd.DataFrame(train_data_list).to_csv(CSV_TRAIN, index=False)

    #     # Create dummy val data
    #     val_data_list = []
    #     for i in range(5): # 5 dummy val samples
    #         filename = f'vol_{i:03d}.npy'
    #         dummy_img = np.random.rand(*IMAGE_SIZE).astype(np.float32)
    #         label = np.random.randint(0, 2)
    #         np.save(os.path.join(PATH_VAL, filename), dummy_img)
    #         val_data_list.append({'filename': filename, 'label': label})
    #     pd.DataFrame(val_data_list).to_csv(CSV_VAL, index=False)
    # print("Dummy data and CSVs created.")

    # storage_name = "sqlite:///db.sqlite3" # Store results in a SQLite database
    study = optuna.create_study(
        study_name="lidc_SE3DCNN_se_optimization",
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(),
        # storage=storage_name,
        load_if_exists=True # Continue from previous runs if db.sqlite3 exists
    )
    
    starting = "Starting Optuna optimization for SE3DCNN..."
    print(starting)
    log_message(starting)
    # For 3D models, n_trials and timeout should be chosen carefully.
    # 20 trials, 30 min timeout might be very short for a full run.
    # Start with fewer trials and epochs for quick testing.
    study.optimize(objective, n_trials=50, timeout=7200) # 50 trials, 2 hours timeout

    finish = "\nOptimization finished!"
    print(finish)
    log_message(finish)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    finish_trials = (f"Number of finished trials: {len(study.trials)}")
    print(finish_trials)
    log_message(finish_trials)

    pruned_trials = (f"Number of pruned trials: {len(pruned_trials)}")
    print(pruned_trials)
    log_message(pruned_trials)

    complete_trials = (f"Number of complete trials: {len(complete_trials)}")
    print(complete_trials)
    log_message(complete_trials)

    trial = study.best_trial
    best_trial_print = (f"\nBest trial:{trial}")
    print(best_trial_print)
    log_message(best_trial_print)

    print(f"  Value (validation accuracy): {trial.value:.4f}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        res = (f"    {key}: {value}")
        print(res)
        log_message(res)

    # Visualization (ensure plotly and kaleido are installed)
    try:
        import plotly.io as pio
        # Important for WSL/remote environments:
        # If your browser isn't opening, try setting to 'browser' explicitly
        # and ensure X server (for WSLg or VcXsrv) is running if needed.
        pio.renderers.default = "browser" 
        from optuna.visualization import (plot_optimization_history, 
                                          plot_param_importances, 
                                          plot_slice, 
                                          plot_contour, 
                                          plot_parallel_coordinate, 
                                          plot_edf,
                                          plot_intermediate_values)

        print("\nGenerating optimization history plot...")
        fig_history = plot_optimization_history(study)
        fig_history.write_html("optimization_history.html")
        fig_history.show()

        print("Generating parameter importances plot...")
        fig_importances = plot_param_importances(study)
        fig_importances.write_html('importances.html')
        fig_importances.show()

        print("Generating slice plot...")
        fig_slice = plot_slice(study)
        fig_slice.write_html('plot_slices.html')
        fig_slice.show()
        
        print("Generating plot contour...")
        fig_contour = plot_contour(study, params=['lr', 'batch_size'])
        fig_contour.write_html('plot_contour.html')
        fig_contour.show()
        
        print("Generating slice plot parallel coordinate ...")
        fig_parallel = plot_parallel_coordinate(study)
        fig_parallel.write_html('parallel_coordinates.html')
        fig_parallel.show()
        
        print("Generating edf ...")
        fig_edf = plot_edf(study)
        fig_edf.write_html('edf.html')
        fig_edf.show()
        
        print("Generating plot intermediate values ...")
        fig_intermediate = plot_intermediate_values(study) # Only if you report intermediate values in your objective function
        fig_intermediate.write_html('intermediate.html')
        fig_intermediate.show()
        

    except ImportError:
        print("\nPlotly or Kaleido are not installed. Skipping visualization.")
        print("Install them with: pip install plotly kaleido")
    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        print("Ensure your display environment is correctly configured (e.g., X server for WSL).")

elapsed = time.time() - t1
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)
summary_ = f"######## Training LR Finished in {int(hours)}h {int(minutes)}m {int(seconds)}s ###########"
print(summary_)
log_message(summary_)