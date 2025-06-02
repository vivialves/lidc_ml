import pandas as pd
import os
import re
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_patient_id(filename):
    # Adjust this regex depending on how patient IDs appear
    match = re.search(r"lidc-idri-\d+", filename.lower())
    return match.group(0) if match else None

def create_dataframe(path_file):
    df = pd.DataFrame(columns=['filename', 'dirname', 'class', 'patient_id'])

    for dirname, _, filenames in os.walk(path_file):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                patient_id = extract_patient_id(filename)
                label = os.path.basename(dirname)
                df.loc[len(df)] = {
                    'filename': filename,
                    'dirname': dirname,
                    'class': label,
                    'patient_id': patient_id
                }
    return df

def create_dataset(path_file, output_dir="data/LIDC_split", train_frac=0.6, val_frac=0.2, test_frac=0.2, random_state=42):

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1."

    df = create_dataframe(path_file)
    df = df.dropna(subset=['patient_id'])

    # Get all unique patient IDs and their labels
    patient_class_map = df.groupby("patient_id")["class"].first()

    # Split patient IDs into train and temp (val+test)
    train_patients, temp_patients = train_test_split(
        patient_class_map.index.tolist(),
        train_size=train_frac,
        stratify=patient_class_map.values,
        random_state=random_state
    )

    # Now split temp into validation and test
    temp_class_map = patient_class_map.loc[temp_patients]
    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_frac / (val_frac + test_frac),
        stratify=temp_class_map.values,
        random_state=random_state
    )


    # Split df based on patient IDs
    train_df = df[df["patient_id"].isin(train_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]
    val_df = df[df["patient_id"].isin(val_patients)]

    for split_name, split_df in [("train", train_df), ("test", test_df), ("val", val_df)]:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name}"):
            class_name = row["class"]
            src = os.path.join(row["dirname"], row["filename"])
            dst_dir = os.path.join(output_dir, split_name, class_name)
            dst = os.path.join(dst_dir, row["filename"])

            os.makedirs(dst_dir, exist_ok=True)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            else:
                print(f"File already exists: {dst}")

create_dataset("data/LIDC_classes_slices_512")