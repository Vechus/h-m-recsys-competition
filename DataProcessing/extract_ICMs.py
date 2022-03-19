import numpy as np
import pandas as pd

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset


DATASET_NAME = 'hm_ICMs'

# TODO: for all functions, take input the object of the dataset manager to append the ICMs to

def gen_ICM_product_type_name(manager, articles):
    print('Creating ICM product type name')

    icm_df = articles[['article_id', 'product_type_name']]
    print(icm_df)
    icm_df.rename(columns={"product_type_name": "FeatureID", "article_id": "ItemID"}, inplace=True)
    icm_df['ItemID'] = icm_df['ItemID'].astype(str)
    icm_df['Data'] = 1.0
    print(icm_df)

    manager.add_ICM(icm_df, 'ICM_prod_type')

