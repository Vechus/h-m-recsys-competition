import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset

from DataProcessing.extract_URM import generate_URM_all
from DataProcessing.extract_ICMs import gen_ICM_list
from DataProcessing.split_train_validation_leave_timestamp_out import split_train_validation_leave_timestamp_out
from DataProcessing.extract_UCMs import gen_UCM_list

DATASET_NAME = 'hm'


if __name__ == '__main__':
    # load .env file
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')

    transactions = pd.read_csv('./dataset/transactions_train.csv')
    articles = pd.read_csv('./dataset/articles.csv')
    customers = pd.read_csv('./dataset/customers.csv')

    print('Loaded all files')

    manager = DatasetMapperManager()

    # URM ALL
    generate_URM_all(manager, transactions)
    # generate all ICMs
    gen_ICM_list(manager, articles)
    # URM split
    split_train_validation_leave_timestamp_out(manager, transactions, (pd.Timestamp("2019-09-23"), pd.Timestamp("2019-09-30")),
                                               (0, 0), False)
    # generate UCMs
    gen_UCM_list(manager, customers)

    # generate dataset with URM (Implicit=True)
    dataset = manager.generate_Dataset(DATASET_NAME, True)
    dataset.save_data('./processed/{}/'.format(DATASET_NAME))
    dataset.print_statistics()
    dataset.print_statistics_global()
