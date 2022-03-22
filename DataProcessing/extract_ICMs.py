import numpy as np
import pandas as pd

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset


DATASET_NAME = 'hm_ICMs'

# TODO: for all functions, take input the object of the dataset manager to append the ICMs to

ICM_list = [
    'product_type_name',
    'product_type_no',
    'product_group_name',
    'graphical_appearance_name',
    'colour_group_name'
]

def gen_ICM_list(manager, articles):
    for column in ICM_list:
        # TODO fai la roba
        print('Creating ICM for column {}'.format(column))

        icm_df = articles[['article_id', column]]
        icm_df.rename(columns={column: "FeatureID", "article_id": "ItemID"}, inplace=True)
        icm_df['ItemID'] = icm_df['ItemID'].astype(str)
        icm_df['FeatureID'] = icm_df['FeatureID'].astype(str)
        icm_df['Data'] = 1.0
        manager.add_ICM(icm_df, 'ICM_{}'.format(column))
