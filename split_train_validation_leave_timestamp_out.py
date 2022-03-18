"""
Created on 14/03/2022

@author: Riccardo Pazzi
"""

import numpy as np
import scipy.sparse as sps
import pandas as pd

from Data_manager.DatasetMapperManager import DatasetMapperManager

# Constants
timestamp_column = 't_dat'
userid_column = 'customer_id'
itemid_column = 'article_id'
DATASET_NAME = 'hm-temporal'


def retrieve_timeframe_interactions(timestamp_df, validation_ts_tuple, test_ts_tuple, use_validation_set):
    """
        Retrieve interactions in a certain time window from dataframe and return a split list of dataframes
        :param timestamp_df:
        :param validation_ts_tuple:
        :param test_ts_tuple:
        :param use_validation_set:
        :return:
    """
    # lists of tuples containing the timeframe interactions
    # Starting date included, ending date not included
    # Intervals must not be overlapping to avoid data spilling from validation to test
    if use_validation_set:
        if not (validation_ts_tuple[0] > test_ts_tuple[1] or validation_ts_tuple[1] < test_ts_tuple[0]):
            raise ValueError

    interactions = []

    t1 = timestamp_df[timestamp_column].searchsorted(test_ts_tuple[0])
    t2 = timestamp_df[timestamp_column].searchsorted(test_ts_tuple[1])
    # print(t1, t2)
    test_interactions = timestamp_df.iloc[t1:t2 - 1]
    # print(test_interactions.head)

    if use_validation_set:
        t1_val = timestamp_df[timestamp_column].searchsorted(validation_ts_tuple[0])
        t2_val = timestamp_df[timestamp_column].searchsorted(validation_ts_tuple[1])
        validation_interactions = timestamp_df.iloc[t1_val:t2_val - 1]
        # Create train set depending on split dates
        if validation_ts_tuple[0] < test_ts_tuple[0]:
            # Validation is before test set
            train_interactions = pd.concat([timestamp_df.iloc[0:t1_val - 1], timestamp_df.iloc[t2_val:t1 - 1],
                                            timestamp_df.iloc[t2:]])
        else:
            # Test is before validation
            train_interactions = pd.concat([timestamp_df.iloc[0:t1 - 1], timestamp_df.iloc[t2:t1_val - 1],
                                            timestamp_df.iloc[t2_val:]])
    else:
        # Just test set, create train on everything else
        train_interactions = pd.concat([timestamp_df.iloc[:t1 - 1], timestamp_df.iloc[t2:]])

    # Create interaction list with TRAIN, TEST, VALIDATION
    print(train_interactions.columns)
    print(train_interactions.head())
    interactions.append(train_interactions)
    interactions.append(test_interactions)
    if use_validation_set:
        interactions.append(validation_interactions)
    else:
        # Append empty dataframe if no validation set should be provided
        interactions.append(pd.DataFrame())
    return interactions


def split_train_validation_leave_timestamp_out(timestamp_df, test_ts_tuple, validation_ts_tuple=(0, 0),
                                               use_validation_set=True):
    """
        The function splits an URM in two matrices selecting on the base of timestamp
    :param URM:
    :param timestamp_df:
    :param user_id_mapping: dictionary used to map user - user_id association
    :param item_id_mapping:
    :param validation_ts_tuple: tuples with starting timestamp and ending timestamps included
    :param test_ts_tuple:
    :param use_validation_set:
    :return:
        """

    # create dataset manager
    manager = DatasetMapperManager()

    # Retrieve which users fall in the wanted timeframe
    timestamp_df[timestamp_column] = pd.to_datetime(timestamp_df[timestamp_column], format='%Y-%m-%d')
    timestamp_df.drop('price', inplace=True, axis=1)
    timestamp_df.drop('sales_channel_id', inplace=True, axis=1)
    timestamp_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    timestamp_df['ItemID'] = timestamp_df['ItemID'].astype(str)
    timestamp_df['Data'] = 1.0

    interactions = retrieve_timeframe_interactions(timestamp_df, validation_ts_tuple,
                                                   test_ts_tuple, use_validation_set)

    train_interactions = interactions[0]
    test_interactions = interactions[1]
    validation_interactions = interactions[2]

    print(train_interactions.columns)
    print(train_interactions.head())
    print(test_interactions.head())
    # Drop duplicates, this could be changed since number of bought items could be an important information
    train_interactions.drop(timestamp_column, inplace=True, axis=1)
    test_interactions.drop(timestamp_column, inplace=True, axis=1)

    print("Dropped matrices")
    print(train_interactions.head())
    print(test_interactions.head())

    train_interactions.drop_duplicates(inplace=True)
    test_interactions.drop_duplicates(inplace=True)

    manager.add_URM(train_interactions, 'URM_train')
    manager.add_URM(test_interactions, 'URM_test')
    if use_validation_set:
        validation_interactions.drop(timestamp_column, inplace=True, axis=1)
        validation_interactions.drop_duplicates(inplace=True)
        manager.add_URM(validation_interactions, 'URM_validation')

    # generate dataset with URM (Implicit=True)
    dataset = manager.generate_Dataset(DATASET_NAME, True)
    dataset.save_data('./processed/{}/'.format(DATASET_NAME))
    dataset.print_statistics_global()
    # print(dataset.get_URM_all())
    # train_URM = sps.coo_matrix((train_data, (train_users, train_items)))
    # train_URM = train_URM.tocsr()

    """users = []
    items = []
    data = []
    for index, row in test_interactions.iterrows():
        users.append(user_id_mapping[row[userid_column]])
        items.append(item_id_mapping[row[itemid_column]])
        data.append(1)

    URM_test_builder.add_data_lists(users, items, data)
    test_URM = URM_test_builder.get_SparseMatrix()
    # test_URM = sps.coo_matrix((data, (users, items)))
    # test_URM = test_URM.tocsr()

    if use_validation_set:
        users = []
        items = []
        data = []
        for index, row in validation_interactions.iterrows():
            users.append(user_id_mapping[row[userid_column]])
            items.append(item_id_mapping[row[itemid_column]])
            data.append(1)

        URM_validation_builder.add_data_lists(users, items, data)
        validation_URM = URM_validation_builder.get_SparseMatrix()
        # validation_URM = sps.coo_matrix((data, (users, items)))
        # validation_URM = validation_URM.tocsr()
        return train_URM, test_URM, validation_URM"""

    return 0


if __name__ == "__main__":
    transactions = pd.read_csv('./dataset/transactions_train.csv')
    print("Loaded data into memory...")
    split_train_validation_leave_timestamp_out(transactions, (pd.Timestamp("2019-09-23"), pd.Timestamp("2019-09-30")),
                                               (0, 0), False)
