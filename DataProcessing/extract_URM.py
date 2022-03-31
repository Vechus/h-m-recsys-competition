import numpy as np
import pandas as pd

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset

DATASET_NAME = 'hm_URM_all'


# generates the dataset object with the URM_all in place
def generate_URM_all(manager, transactions):
    # urm_df = transactions.drop('t_dat', axis=1)
    # # urm_df.drop('price', inplace=True, axis=1)
    # # urm_df.drop('sales_channel_id', inplace=True, axis=1)
    # urm_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    # urm_df['ItemID'] = urm_df['ItemID'].astype(str)
    # urm_df['Data'] = 1.0
    # # transactions.convert_dtypes(convert_integer=False)
    # urm_df.drop_duplicates(subset=["UserID", "ItemID"], inplace=True)
    # print(urm_df)

    transactions.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    transactions['ItemID'] = transactions['ItemID'].astype(str)
    transactions['Data'] = 1.0
    transactions = transactions[['UserID', 'ItemID', 'Data']]
    transactions.drop_duplicates()

    # update dataset manager
    manager.add_URM(transactions, 'URM_all')

