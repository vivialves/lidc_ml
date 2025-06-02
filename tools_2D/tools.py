import pandas as pd
import os
import shutil
import re
import torchvision.transforms as transforms

from tqdm import tqdm
from sklearn.model_selection import train_test_split


from PIL import Image


def create_csv(path_file, path_csv):

    df = pd.DataFrame(columns=['filename', 'dirname'])

    for dirname, _, filenames in os.walk(path_file):
        for filename in filenames:
            df.loc[len(df)] = {'filename': filename, 'dirname': dirname}
    df.to_csv(os.path.realpath(path_csv), index=False, sep=';')

path = os.path.abspath('data')
list_path = path.split('/')
list_path.remove('data')

# create_csv(os.path.realpath('p2-mammography/train_original'), os.path.join('/'.join(list_path), 'mammography_trainoriginal.csv'))

def read_csv(path_folder, file_name):
    path = os.path.realpath(path_folder)
    list_path = path.split('/')
    list_path.remove(list_path[-1])
    path_info_copy = os.path.join('/'.join(list_path), file_name)
    return pd.read_csv(path_info_copy, sep=';')

# print(read_csv('data', 'breast_before.csv'))


# print(os.path.realpath('p2-mammography'))

def create_dataset(df, output_dir="p2-mammography", train_size=0.8, random_state=42):

    train_dir = os.path.join(os.path.realpath(output_dir), "train")
    test_dir = os.path.join(os.path.realpath(output_dir), "test")

    for target_dir in df['dirname'].apply(lambda x: x.split('/')[-1]).unique():

        if target_dir != 'TIFF Images':

            class_df = df[df['dirname'].str.contains(target_dir, case=False)]
    
            train_files, test_files = train_test_split(
                class_df, train_size=train_size, random_state=random_state
            )

            for dir_name, split_df, dest_dir in [
                ("train", train_files, train_dir),
                    ("test", test_files, test_dir),
            ]:
    
                for index, rows in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {dir_name} {target_dir}"):
                    # pattern = r"(Density|density)(\d+)([A-Za-z]+)"
                    # replacement = r"\1\2_\3"
                    list_dirname = rows['dirname'].split('/')
                    # class_name = re.sub(pattern, replacement, list_dirname[-1].capitalize())
                    # class_name = re.sub(r'([Dd]ensity\d)([Bb]enign|[Mm]alignant)', r'\1_\2', list_dirname[-1])
                    class_name = re.sub(r'([Dd]ensity)(\d)(benign|malignant)', lambda m: f"Density{m.group(2)}_{m.group(3).capitalize()}", list_dirname[-1])
                    src = os.path.join(rows['dirname'], rows['filename'])
                    dst_class_dir = os.path.join(dest_dir, class_name)
                    dst = os.path.join(dst_class_dir, rows['filename'])
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    else:
                        print(f"File already exists: {dst}. Skipping.")


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


def data_augmentation(df, name_dataset, output_dir="p2-mammography"):

    augment_dir = os.path.join(os.path.realpath(output_dir), name_dataset + '_augmented')

    for target_dir in df['dirname'].apply(lambda x: x.split('/')[-1]).unique():

        class_df = df[df['dirname'].str.contains(target_dir, case=False)]

        dir_name, class_df, dest_dir = (name_dataset + "_augmented", class_df, augment_dir)

        for index, rows in tqdm(class_df.iterrows(), total=len(class_df), desc=f"Augmenting {dir_name} {target_dir}"):
            list_dirname = rows['dirname'].split('/')
            class_name = list_dirname[-1]
            input_image = os.path.join(rows['dirname'], rows['filename'])
            output_directory = os.path.join(dest_dir, class_name)
            if class_name == 'Density1_Benign':
                num_augmentations = 17
            elif class_name == 'Density2_Benign':
                num_augmentations = 4
            elif class_name == 'Density3_Benign':
                num_augmentations = 40
            elif class_name == 'Density4_Benign':
                num_augmentations = 35
            elif class_name == 'Density1_Malignant':
                num_augmentations = 14
            elif class_name == 'Density2_Malignant':
                num_augmentations = 6
            elif class_name == 'Density3_Malignant':
                num_augmentations = 60
            elif class_name == 'Density4_Malignant':
                num_augmentations = 122

            augment_image(input_image, output_directory, num_augmentations)
   

list_csv = ['mammography_test_original.csv']
output = 'data'

for csv in list_csv:

    df_train = read_csv( output,csv)
    name_dataset = csv.split('_')
    data_augmentation(df_train, name_dataset[1])

# Example Usage:

path = os.path.realpath('p2-mammography/train_original')
# print(path)

input_image = os.path.join(os.path.realpath('p2-mammography'), "train_original/Density1_Benign/20588680.png")
output_directory = os.path.join(os.path.realpath('p2-mammography'), "train_augmented/Density1_Benign")
# os.makedirs(output_directory, exist_ok=True)
# augment_image(input_image, output_directory, num_augmentations=36)