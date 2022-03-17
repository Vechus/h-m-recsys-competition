import numpy as np
import pandas as pd

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset


DATASET_NAME = 'hm'


# generates the dataset object with the URM_all in place
def generate_dataset():
    
    transactions = pd.read_csv('./dataset/transactions_train.csv')

    urm_df = transactions.drop('t_dat', inplace=True, axis=1)
    urm_df.drop('price', inplace=True, axis=1)
    urm_df.drop('sales_channel_id', inplace=True, axis=1)
    urm_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    urm_df['ItemID'] = transactions['ItemID'].astype(str)
    urm_df['Data'] = 1.0
    #transactions.convert_dtypes(convert_integer=False)
    urm_df.drop_duplicates(inplace=True)
    print(urm_df)

    # create dataset manager
    manager = DatasetMapperManager()
    manager.add_URM(urm_df, 'URM_all')

    # generate dataset with URM (Implicit=True)
    dataset = manager.generate_Dataset(DATASET_NAME, True)
    dataset.save_data('./processed/{}/'.format(DATASET_NAME))
    dataset.print_statistics()
    print(dataset.get_URM_all())

if __name__ == '__main__':
    generate_dataset()
