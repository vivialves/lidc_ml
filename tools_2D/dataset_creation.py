import os
import sys
from PIL import Image
import pandas as pd
import time

from tools import read_csv, create_dataset, data_augmentation



if __name__ == '__main__':

    print('Processing...')

    list_csv_preparation = ['breast_before.csv','mammography_csv.csv']
    list_csv_augment = ['mammography_trainoriginal.csv','mammography_testoriginal.csv']
    output = 'data'

    for csv in list_csv_augment:
        df = read_csv(output, csv)
        print('Processing...')
        start_time = time.time()
        # create_dataset(df)
        data_augmentation(df,num_augmentations=36)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time:.4f} seconds")

        print('Finished processing!')
