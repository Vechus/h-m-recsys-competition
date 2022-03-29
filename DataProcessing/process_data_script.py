import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset

from DataProcessing.extract_URM import generate_URM_all
from DataProcessing.extract_ICMs import gen_ICM_list
from DataProcessing.split_train_validation_leave_timestamp_out import *
from DataProcessing.extract_UCMs import gen_UCM_list

DATASET_NAME = 'hm'

if __name__ == '__main__':
    # load .env file
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')

    transactions = pd.read_csv('{}/processed_transactions_train.csv'.format(DATASET_PATH))
    articles = pd.read_csv('{}/processed_articles.csv'.format(DATASET_PATH))
    customers = pd.read_csv('{}/processed_customers.csv'.format(DATASET_PATH))

    print('Loaded all files')

    manager = DatasetMapperManager()

    # URM ALL
    # generate_URM_all(manager, transactions)

    # generate all ICMs
    gen_ICM_list(manager, articles)
    # URM split
    # split_train_validation_leave_timestamp_out(manager, transactions, (pd.Timestamp("2019-09-23"), pd.Timestamp("2019-09-30")),
    #                                            (0, 0), False)
    timestamp_list_train = [("2019-06-22", "2019-09-23")]
    timestamp_list_validation = [("2019-09-23", "2019-09-30")]
    split_train_validation_multiple_intervals(manager, transactions, timestamp_list_train, timestamp_list_validation)

    # generate UCMs
    # gen_UCM_list(manager, customers)

    # generate dataset with URM (Implicit=True)
    dataset = manager.generate_Dataset(DATASET_NAME, True)

    PROCESSED_PATH = os.getenv('PROCESSED_PATH')
    dataset.save_data('{}/{}/'.format(PROCESSED_PATH, DATASET_NAME))

    dataset.print_statistics()
    dataset.print_statistics_global()
