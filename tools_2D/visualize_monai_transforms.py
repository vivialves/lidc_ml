import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pydicom # Keep this for general DICOM handling if needed, though MONAI's PydicomReader handles it.

from monai.transforms import (
    LoadImageD,           # Dictionary version
    EnsureChannelFirstD,  # Dictionary version
    ScaleIntensityD,      # Dictionary version
    ResizeD,              # Dictionary version
    RandRotate90D,        # Dictionary version
    RandFlipD,            # Dictionary version
    RandGaussianNoiseD,   # Dictionary version
    GaussianSharpend,
    RandGaussianSharpend,
    RandSpatialCropSamplesd,
    RandWeightedCropd,
    SpatialPadd,
    BorderPadd,
    DivisiblePadd,
    RandCoarseDropoutd,
    RandScaleCropd,
    RandGaussianNoised,
    Compose
)
from monai.data import DataLoader, Dataset
from monai.utils import first

# ---------------- CONFIG ----------------
# Adjust this path to your DICOM directory
DICOM_DIR = "/home/vivianea/projects/BrainInnov/data/LIDC_classes_balanced/cancer"
IMAGE_SIZE = 512
NUM_IMAGES = 3  # just to preview a few
PAD_SIZE = (10, 10)  # Spatial pad size
BORDER_PAD = (30, 50)  # pixels added to each side (y, x)
DIVISIBLE_BY = 128
# ----------------------------------------

# Step 1: Collect sample DICOM files and prepare data dictionaries
dicom_files = [os.path.join(DICOM_DIR, f) for f in os.listdir(DICOM_DIR) if f.endswith('.dcm')]
random.shuffle(dicom_files)
sample_files = dicom_files[:NUM_IMAGES]
print(sample_files)

# Data will be represented as a list of dictionaries, where each dict contains an 'image' key
# pointing to the DICOM file path.
data_dicts = [{"image": f} for f in sample_files]

# Step 2: Define a list of MONAI transforms to test visually using Dictionary Transforms
transform_steps = {
    # For dictionary transforms, you always need to specify the 'keys' argument.
    # In this case, our dictionary has an 'image' key.
    "Original": Compose([
    LoadImageD(keys="image", reader="PydicomReader")
    ]),
    "RandCoarseDropout": Compose([
        LoadImageD(keys="image", reader="PydicomReader"),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
        ResizeD(keys="image", spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        RandCoarseDropoutd(keys="image", holes=6, spatial_size=20, fill_value=0, prob=1.0),
    ]),
    "RandScaleCropd (±20%) to 192x192": Compose([
        LoadImageD(keys="image", reader="PydicomReader"),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
        RandScaleCropd(keys="image", roi_scale=(0.8, 0.8), max_roi_scale=(0.8, 1.2), random_center=True, random_size=False)
    ]),
    "SpatialPadd (to 256x256)": Compose([
        LoadImageD(keys="image", reader="PydicomReader"),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
        SpatialPadd(keys="image", spatial_size=PAD_SIZE)
    ]),
    "BorderPadd (30px y, 50px x)": Compose([
        LoadImageD(keys="image", reader="PydicomReader"),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
        BorderPadd(keys="image", spatial_border=BORDER_PAD)
    ]),
    "RandGaussianNoised": Compose([
        LoadImageD(keys="image", reader="PydicomReader"),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
        ResizeD(keys="image", spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        RandGaussianNoised(keys="image", prob=0.7, mean=0.5, std=0.5),
    ]),
    "RandGaussianSharpen": Compose([
        LoadImageD(keys="image", reader="PydicomReader"),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
        ResizeD(keys="image", spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        RandGaussianSharpend(keys="image", prob=1.0, alpha=(0.5, 1.5), sigma1_x=(0.1, 1.0), sigma2_x=(0.1, 1.0)),
    ]),
    "RandSpatialCropSamples": Compose([
        LoadImageD(keys="image", reader="PydicomReader"),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
        RandSpatialCropSamplesd(keys="image", roi_size=(IMAGE_SIZE//2, IMAGE_SIZE//2), num_samples=1),
    ])
}

# Step 3: Visualize each transformation
# We'll create a MONAI Dataset for each transform to demonstrate the typical workflow.
for data_dict_entry in data_dicts:
    original_filepath = data_dict_entry["image"]
    print(f"\n🖼️ Image: {os.path.basename(original_filepath)}")
    fig, axs = plt.subplots(1, len(transform_steps), figsize=(20, 4))
    fig.suptitle(f"Transformations on {os.path.basename(original_filepath)}", fontsize=14)

    for i, (name, transform) in enumerate(transform_steps.items()):
        test_ds = Dataset(data=[data_dict_entry], transform=transform)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

        transformed_batch = first(test_loader)
        img_data = transformed_batch["image"] # This will be (C, D, H, W) or (C, H, W)

        # IMPORTANT: Handle the dimensions for visualization
        # Remove batch dimension (always 1 here)
        img_data = img_data.squeeze(0) # Shape is now (C, D, H, W) or (C, H, W)

        # If it's 3D (C, D, H, W), select a central slice
        if img_data.ndim == 4: # Implies it's (C, D, H, W)
            # Take the middle slice along the depth (D) dimension
            slice_idx = img_data.shape[1] // 2
            img_np = img_data[0, slice_idx, :, :].cpu().numpy() # [0] for channel, [slice_idx] for depth
        elif img_data.ndim == 3: # Implies it's (C, H, W)
            img_np = img_data[0, :, :].cpu().numpy() # [0] for channel
        else:
            # Fallback for unexpected dimensions, or if it's already 2D
            img_np = img_data.cpu().numpy()
            if img_np.ndim != 2:
                print(f"Warning: Unexpected image dimensions {img_np.shape} for display in transform '{name}'. Attempting to reshape.")
                # This part might need further refinement depending on the actual input shape
                if img_np.ndim == 1:
                    # If it's truly 1D, reshape it into a square if possible
                    side = int(np.sqrt(img_np.shape[0]))
                    if side * side == img_np.shape[0]:
                        img_np = img_np.reshape((side, side))
                    else:
                        print(f"Error: Cannot reshape 1D array of size {img_np.shape[0]} into a square for display.")
                        img_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE)) # Fallback to black image
                # If it's already 2D, or we successfully reshaped, then imshow will work.

        axs[i].imshow(img_np, cmap='gray')
        axs[i].set_title(name)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

print("\n--- DICOM dictionary transform visualization complete. ---")