import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset

from DataProcessing.extract_URM import generate_URM_all
from DataProcessing.extract_ICMs import gen_ICM_list, get_ICM_all, gen_ICM_mix
from DataProcessing.split_train_validation_leave_timestamp_out import *
import DataProcessing.split_train_validation_exponential_decay as exp_decay
from DataProcessing.extract_UCMs import gen_UCM_list

DATASET_NAME = 'hm'

if __name__ == '__main__':
    # load .env file
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')

    transactions = pd.read_parquet('{}/processed_transactions_train.parquet'.format(DATASET_PATH))
    articles = pd.read_parquet('{}/processed_articles.parquet'.format(DATASET_PATH))
    customers = pd.read_parquet('{}/processed_customers.parquet'.format(DATASET_PATH))

    print('Loaded all files')

    manager = DatasetMapperManager()

    # generate all ICMs
    gen_ICM_list(manager, articles)
    get_ICM_all(manager, articles)
    gen_ICM_mix(manager, articles, top_number=5)
    gen_ICM_mix(manager, articles, top_number=10)
    gen_ICM_mix(manager, articles, top_number=15)

    # URM split
    # timestamp_list_train = [("2019-06-22", "2019-09-23")]
    # timestamp_list_validation = [("2019-09-23", "2019-09-30")]

    timestamp_list_train = [("2019-06-22", "2019-09-23")]
    timestamp_list_validation = [("2019-09-23", "2019-09-30")]
    split_train_validation_multiple_intervals(manager, transactions, timestamp_list_train, timestamp_list_validation, URM_train='URM_train', URM_validation='URM_validation')    
    split_train_validation_multiple_intervals_Explicit_By_Repeat_Purchase(manager, transactions, timestamp_list_train, timestamp_list_validation, URM_train='URM_train_explicit', URM_validation='URM_validation_explicit')
    exp_decay.split_train_validation_multiple_intervals(manager, transactions, timestamp_list_train, timestamp_list_validation, exponential_decay=30, URM_train='URM_train_exp', URM_validation='URM_validation_exp')

    # URM_train for submission
    timestamp_list_submission = [("2020-06-22", "2020-09-23")]
    split_submission_train_intervals(manager, transactions, timestamp_list_submission)
    split_submission_train_intervals_explicit(manager, transactions, timestamp_list_submission)


    # generate UCMs
    gen_UCM_list(manager, customers)

    # generate dataset with URM (Implicit=True)
    dataset = manager.generate_Dataset(DATASET_NAME, is_implicit=False)

    # PROCESSED_PATH = os.getenv('PROCESSED_PATH')
    dataset.save_data('{}/processed_train_20190622_20190923_val_20190923_20190930_Explicit_and_exp/{}/'.format(DATASET_PATH, DATASET_NAME))

    dataset.print_statistics()
    dataset.print_statistics_global()
