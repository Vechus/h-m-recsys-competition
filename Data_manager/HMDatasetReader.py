"""
Created on 19/03/2022
@author: Riccardo Pazzi
"""

import zipfile, shutil
import numpy as np
import pandas as pd
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.DatasetMapperManager import DatasetMapperManager
from DataProcessing.extract_URM import generate_URM_all
from DataProcessing.extract_ICMs import gen_ICM_product_type_name
from DataProcessing.split_train_validation_leave_timestamp_out import split_train_validation_leave_timestamp_out


DATASET_NAME = 'hm'

DATASET_SOURCE = "hm-temporal/"
DATASET_ZIP_NAME = "dataset_URM.zip"


class HMDatasetReader(DataReader):
    DATASET_SUBFOLDER = DATASET_SOURCE
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []
    DATASET_SPLIT_ROOT_FOLDER = "./processed/"

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original
        print("Processed dataset not found, loading from original csv...")
        # Code taken from process_data_script.py to gracefully handle errors
        try:
            transactions = pd.read_csv('../dataset/transactions_train.csv')
            articles = pd.read_csv('../dataset/articles.csv')
        except FileNotFoundError:
            print("ERROR: dataset csv files are missing in ./dataset/")
            return

        manager = DatasetMapperManager()

        # URM ALL
        generate_URM_all(manager, transactions)
        # ICM product type name
        gen_ICM_product_type_name(manager, articles)
        # URM split
        print("Splitting URM in train and test...")
        split_train_validation_leave_timestamp_out(manager, transactions,
                                                   (pd.Timestamp("2019-09-23"), pd.Timestamp("2019-09-30")),
                                                   (0, 0), False)

        # generate dataset with URM (Implicit=True)
        print("Creating dataset object...")
        dataset = manager.generate_Dataset(DATASET_NAME, True)
        dataset.save_data('../processed/{}/'.format(DATASET_NAME))
        dataset.print_statistics()
        dataset.print_statistics_global()
        return dataset

