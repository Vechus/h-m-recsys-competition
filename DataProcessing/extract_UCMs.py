import numpy as np
import pandas as pd

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset


UCM_list = [
    'fashion_news_frequency',
    'club_member_status'
]

def gen_UCM_list(manager, customers):
    for column in UCM_list:
        print('Creating UCM for column {}'.format(column))

        ucm_df = customers[['customer_id', column]]
        ucm_df.rename(columns={column: "FeatureID", "customer_id": "UserID"}, inplace=True)
        ucm_df['UserID'] = ucm_df['UserID'].astype(str)
        ucm_df['FeatureID'] = ucm_df['FeatureID'].astype(str)
        ucm_df['Data'] = 1.0
        manager.add_UCM(ucm_df, 'UCM_{}'.format(column))