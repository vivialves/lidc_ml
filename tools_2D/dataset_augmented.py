import pandas as pd
import os
import shutil
import re
import torchvision.transforms as transforms

from tqdm import tqdm
from sklearn.model_selection import train_test_split


from PIL import Image


def augment_image(image_path, output_dir, num_augmentations=5):
    """
    Augments an image and saves the augmented versions.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save augmented images.
        num_augmentations (int): Number of augmented versions to create.
    """
    train_dir = os.path.join(os.path.realpath(output_dir), "train")
    test_dir = os.path.join(os.path.realpath(output_dir), "test")

    img = Image.open(image_path).convert("RGB")  # Ensure RGB

    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomInvert(),
        transforms.ToTensor(), #converts to tensor.
    ])

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(num_augmentations):
        augmented_img = transform(img)
        augmented_pil = transforms.ToPILImage()(augmented_img) #convert back to PIL to save.
        output_path = os.path.join(output_dir, f"{base_filename}_augmented_{i}.png")
        augmented_pil.save(output_path)


# Example Usage:

path = os.path.realpath('p2-mammography/train_original')
print(path)

input_image = os.path.join(os.path.realpath('p2-mammography'), "train_original/Density1_Benign/20588680.png")
output_directory = os.path.join(os.path.realpath('p2-mammography'), "train_augmented/Density1_Benign")
# os.makedirs(output_directory, exist_ok=True)


augment_image(input_image, output_directory, num_augmentations=36)