import pandas as pd
import os
import re
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split


def create_dataframe(path_file):

    df = pd.DataFrame(columns=['filename', 'dirname'])

    for dirname, _, filenames in os.walk(path_file):
        for filename in filenames:
            df.loc[len(df)] = {'filename': filename, 'dirname': dirname}
    return df

# print(create_dataframe(os.path.realpath('data/LIDC_ready')))

def create_dataset(path_file, output_dir="data/LIDC_pre", train_size=0.8, random_state=42):

    train_dir = os.path.join(os.path.realpath(output_dir), "train")
    test_dir = os.path.join(os.path.realpath(output_dir), "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    df = create_dataframe(path_file)

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
                    list_dirname = rows['dirname'].split('/')
                    class_name = list_dirname[-1]
                    src = os.path.join(rows['dirname'], rows['filename'])
                    dst_class_dir = os.path.join(dest_dir, class_name)
                    dst = os.path.join(dst_class_dir, rows['filename'])

                    os.makedirs(dst_class_dir, exist_ok=True)
                    
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    else:
                        print(f"File already exists: {dst}. Skipping.")


create_dataset(os.path.realpath('data/LIDC_ready'))

