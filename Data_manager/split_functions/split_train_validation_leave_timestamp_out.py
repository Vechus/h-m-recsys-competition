"""
Created on 14/03/2022

@author: Riccardo Pazzi
"""

import numpy as np
import scipy.sparse as sps
import pandas as pd
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

# Constants
timestamp_column = 't_dat'
userid_column = 'customer_id'
itemid_column = 'article_id'


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
    if not (validation_ts_tuple[0] > test_ts_tuple[1] or validation_ts_tuple[1] < test_ts_tuple[0]):
        raise ValueError

    interactions = []

    t1 = timestamp_df[timestamp_column].searchsorted(test_ts_tuple[0])
    t2 = timestamp_df[timestamp_column].searchsorted(test_ts_tuple[1])
    test_interactions = timestamp_df.loc[t1:t2 - 1]

    if use_validation_set:
        t1_val = timestamp_df[timestamp_column].searchsorted(validation_ts_tuple[0])
        t2_val = timestamp_df[timestamp_column].searchsorted(validation_ts_tuple[1])
        validation_interactions = timestamp_df.loc[t1_val:t2_val - 1]
        # Create train set depending on split dates
        if validation_ts_tuple[0] < test_ts_tuple[0]:
            # Validation is before test set
            train_interactions = pd.concat([timestamp_df.loc[0:t1_val - 1], timestamp_df.loc[t2_val:t1 - 1],
                                            timestamp_df.loc[t2:]])
        else:
            # Test is before validation
            train_interactions = pd.concat([timestamp_df.loc[0:t1 - 1], timestamp_df.loc[t2:t1_val - 1],
                                            timestamp_df.loc[t2_val:]])
    else:
        # Just test set, create train on everything else
        train_interactions = pd.concat([timestamp_df.loc[:t1 - 1], timestamp_df.loc[t2:]])

    # Create interaction list with TRAIN, TEST, VALIDATION
    interactions.append(train_interactions)
    interactions.append(test_interactions[userid_column, itemid_column])
    if use_validation_set:
        interactions.append(validation_interactions[userid_column, itemid_column])
    else:
        # Append empty dataframe if no validation set should be provided
        interactions.append(pd.DataFrame())
    return interactions


def split_train_validation_leave_timestamp_out(URM, timestamp_df, user_id_mapping, item_id_mapping,
                                               test_ts_tuple, validation_ts_tuple=(0, 0),
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
    :param leave_random_out:
    :return:
        """

    URM = sps.csr_matrix(URM)
    n_users, n_items = URM.shape

    URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                auto_create_col_mapper=False, n_cols=n_items)

    URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                               auto_create_col_mapper=False, n_cols=n_items)

    if use_validation_set:
        URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                         auto_create_col_mapper=False, n_cols=n_items)

    # Retrieve which users fall in the wanted timeframe

    interactions = retrieve_timeframe_interactions(timestamp_df, validation_ts_tuple,
                                                   test_ts_tuple, use_validation_set)

    train_interactions = interactions[0]
    test_interactions = interactions[1]
    validation_interactions = interactions[2]

    # Remove extra interactions to avoid errors during creation of COO matrix (duplicates would be summed)

    train_interactions.drop_duplicates()
    test_interactions.drop_duplicates()
    validation_interactions.drop_duplicates()

    # Create COO matrices and populate with data
    # Could use a function to do this

    train_users = []
    train_items = []
    train_data = []
    for index, row in train_interactions.iterrows():
        # Use dictionary to retrieve corresponding user id
        train_users.append(user_id_mapping[row[userid_column]])
        train_items.append(item_id_mapping[row[itemid_column]])
        train_data.append(1)
    train_URM = sps.coo_matrix((train_data, (train_users, train_items)))
    train_URM = train_URM.tocsr()

    users = []
    items = []
    data = []
    for index, row in test_interactions.iterrows():
        users.append(user_id_mapping[row[userid_column]])
        items.append(item_id_mapping[row[itemid_column]])
        data.append(1)
    test_URM = sps.coo_matrix((data, (users, items)))
    test_URM = test_URM.tocsr()

    if use_validation_set:
        users = []
        items = []
        data = []
        for index, row in validation_interactions.iterrows():
            users.append(user_id_mapping[row[userid_column]])
            items.append(item_id_mapping[row[itemid_column]])
            data.append(1)
        validation_URM = sps.coo_matrix((data, (users, items)))
        validation_URM = validation_URM.tocsr()
        return train_URM, test_URM, validation_URM

    return train_URM, test_URM
