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
DATASET_NAME = 'hm-Sept-2020'


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
    print(train_interactions.size)

    print(test_interactions.columns)
    print(test_interactions.head())
    print(test_interactions.size)

    interactions.append(train_interactions)
    interactions.append(test_interactions)
    if use_validation_set:
        interactions.append(validation_interactions)
    else:
        # Append empty dataframe if no validation set should be provided
        interactions.append(pd.DataFrame())
    return interactions


def split_train_validation_leave_timestamp_out(manager, timestamp_df, test_ts_tuple, validation_ts_tuple=(0, 0),
                                               use_validation_set=True):
    """
        The function splits an URM in two matrices selecting on the base of timestamp
    :param manager:
    :param timestamp_df:
    :param validation_ts_tuple: tuples with starting timestamp and ending timestamps included
    :param test_ts_tuple:
    :param use_validation_set:
    :return:
        """

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

    return 0


def merge_splits(timestamp_df: pd.DataFrame, timestamp_array, columns_name=None):
    """
    :param timestamp_df: The entire timestamp dataframe
    :param timestamp_array: The array of tuples of time intervals that will create the test set
    :param columns_name:
    """
    if columns_name is None:
        columns_name = ["UserID", "ItemID", "Data"]

    final_df = pd.DataFrame(columns=columns_name)
    i = 1
    for timeframe in timestamp_array:
        print("Processing interval {}...".format(i))
        i += 1
        t1 = timestamp_df[timestamp_column].searchsorted(timeframe[0])
        t2 = timestamp_df[timestamp_column].searchsorted(timeframe[1])
        cutout = timestamp_df.iloc[t1:t2 - 1]
        timestamp_df.drop(timestamp_df.index[[t1, t2 - 1]], inplace=True)
        final_df = pd.concat([final_df, cutout])

    return timestamp_df, final_df


def split_train_test_multiple_intervals(manager, timestamp_df, timestamp_array):
    # Retrieve which users fall in the wanted list of time frames
    print("Preprocessing dataframe...")
    timestamp_df[timestamp_column] = pd.to_datetime(timestamp_df[timestamp_column], format='%Y-%m-%d')
    timestamp_df.drop('price', inplace=True, axis=1)
    timestamp_df.drop('sales_channel_id', inplace=True, axis=1)
    timestamp_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    timestamp_df['ItemID'] = timestamp_df['ItemID'].astype(str)
    timestamp_df['Data'] = 1.0

    # Create test/train splits
    train_interactions, test_interactions = merge_splits(timestamp_df, timestamp_array)

    print(train_interactions.head())
    print(test_interactions.head())

    train_interactions.drop(timestamp_column, inplace=True, axis=1)
    test_interactions.drop(timestamp_column, inplace=True, axis=1)

    train_interactions.drop_duplicates(inplace=True)
    test_interactions.drop_duplicates(inplace=True)
    manager.add_URM(train_interactions, 'URM_train')
    manager.add_URM(test_interactions, 'URM_test')


def merge_splits_without_overwrite_origin_dataset(timestamp_df: pd.DataFrame, timestamp_array, columns_name=None):
    """
    :param timestamp_df: The entire timestamp dataframe
    :param timestamp_array: The array of tuples of time intervals that will create the test set
    :param columns_name:
    """
    if columns_name is None:
        columns_name = ["UserID", "ItemID", "Data"]

    final_df = pd.DataFrame(columns=columns_name)

    dropped_timestamp_df = timestamp_df.copy()
    i = 1
    for timeframe in timestamp_array:
        print("Processing interval {}...".format(i))
        i += 1
        t1 = timestamp_df[timestamp_column].searchsorted(timeframe[0])
        t2 = timestamp_df[timestamp_column].searchsorted(timeframe[1])
        cutout = timestamp_df.iloc[t1:t2 - 1]
        dropped_timestamp_df.drop(timestamp_df.index[[t1, t2 - 1]], inplace=True)
        final_df = pd.concat([final_df, cutout])

    return dropped_timestamp_df, final_df


def split_train_validation_multiple_intervals(manager, timestamp_df, timestamp_array_train, timestamp_array_validation,
                                              URM_train='URM_train', URM_validation='URM_validation'):
    # Retrieve which users fall in the wanted list of time frames
    timestamp_df = timestamp_df.copy()
    print("Preprocessing dataframe...")
    timestamp_df[timestamp_column] = pd.to_datetime(timestamp_df[timestamp_column], format='%Y-%m-%d')

    # timestamp_df.drop('price', inplace=True, axis=1)
    # timestamp_df.drop('sales_channel_id', inplace=True, axis=1)
    timestamp_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    timestamp_df['ItemID'] = timestamp_df['ItemID'].astype(str)
    timestamp_df['Data'] = 1.0

    timestamp_df = timestamp_df[[timestamp_column, 'UserID', 'ItemID', 'Data']]

    # Create test/train splits
    rest_interactions, train_interactions = merge_splits_without_overwrite_origin_dataset(timestamp_df,
                                                                                          timestamp_array_train)

    rest_interactions2, validation_interactions = merge_splits_without_overwrite_origin_dataset(timestamp_df,
                                                                                                timestamp_array_validation)

    print(train_interactions.head())
    print(train_interactions.tail())
    print(validation_interactions.head())
    print(validation_interactions.tail())

    train_interactions.drop(timestamp_column, inplace=True, axis=1)
    validation_interactions.drop(timestamp_column, inplace=True, axis=1)

    train_interactions.drop_duplicates(inplace=True)
    validation_interactions.drop_duplicates(inplace=True)

    manager.add_URM(train_interactions, URM_train)
    manager.add_URM(validation_interactions, URM_validation)


def split_train_validation_multiple_intervals_Explicit_By_Repeat_Purchase(manager, timestamp_df, timestamp_array_train,
                                                                          timestamp_array_validation,
                                                                          URM_train='URM_train',
                                                                          URM_validation='URM_validation'):
    # Retrieve which users fall in the wanted list of time frames
    timestamp_df = timestamp_df.copy()
    print("Preprocessing dataframe...")
    timestamp_df[timestamp_column] = pd.to_datetime(timestamp_df[timestamp_column], format='%Y-%m-%d')

    timestamp_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    timestamp_df['ItemID'] = timestamp_df['ItemID'].astype(str)

    timestamp_df.drop_duplicates()

    timestamp_df_with_timestamp = timestamp_df.copy()

    timestamp_df = timestamp_df[['UserID', 'ItemID']]

    timestamp_df = timestamp_df.groupby(['UserID', 'ItemID']).size().reset_index(name='Data')

    # Drop the abnormal data
    timestamp_df['Data'] = timestamp_df['Data'].apply(lambda x: 20 if x >= 20 else x)

    # Normalization
    max_value = timestamp_df['Data'].max()
    timestamp_df['Data'] = timestamp_df['Data'].apply(lambda x: x / max_value)

    timestamp_df = pd.merge(left=timestamp_df_with_timestamp, right=timestamp_df, how='left', on=['UserID', 'ItemID'])

    print(timestamp_df.head())
    # Create test/train splits
    rest_interactions, train_interactions = merge_splits_without_overwrite_origin_dataset(timestamp_df,
                                                                                          timestamp_array_train)

    rest_interactions2, validation_interactions = merge_splits_without_overwrite_origin_dataset(timestamp_df,
                                                                                                timestamp_array_validation)

    print(train_interactions.head())
    print(train_interactions.tail())
    print(validation_interactions.head())
    print(validation_interactions.tail())

    train_interactions.drop(timestamp_column, inplace=True, axis=1)
    validation_interactions.drop(timestamp_column, inplace=True, axis=1)

    train_interactions.drop_duplicates(inplace=True)
    validation_interactions.drop_duplicates(inplace=True)

    manager.add_URM(train_interactions, URM_train)
    manager.add_URM(validation_interactions, URM_validation)


def split_submission_train_intervals(manager, timestamp_df, timestamp_array_train):
    timestamp_df = timestamp_df.copy()
    # Retrieve which users fall in the wanted list of time frames
    print("Preprocessing URM_submission dataframe...")
    timestamp_df[timestamp_column] = pd.to_datetime(timestamp_df[timestamp_column], format='%Y-%m-%d')

    # timestamp_df.drop('price', inplace=True, axis=1)
    # timestamp_df.drop('sales_channel_id', inplace=True, axis=1)
    timestamp_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    timestamp_df['ItemID'] = timestamp_df['ItemID'].astype(str)
    timestamp_df['Data'] = 1.0

    timestamp_df = timestamp_df[[timestamp_column, 'UserID', 'ItemID', 'Data']]

    df_submission_train = timestamp_df.query(
        "'" + timestamp_array_train[0][0] + "'<=t_dat<'" + timestamp_array_train[0][1] + "'")

    df_submission_train.drop(timestamp_column, inplace=True, axis=1)

    df_submission_train.drop_duplicates(inplace=True)

    print(df_submission_train.head())

    manager.add_URM(df_submission_train, 'URM_submission_train')


def split_submission_train_intervals_explicit(manager, timestamp_df, timestamp_array_submission_explicit):
    # Retrieve which users fall in the wanted list of time frames
    timestamp_df = timestamp_df.copy()
    print("Preprocessing dataframe...")
    timestamp_df[timestamp_column] = pd.to_datetime(timestamp_df[timestamp_column], format='%Y-%m-%d')

    timestamp_df.rename(columns={"customer_id": "UserID", "article_id": "ItemID"}, inplace=True)
    timestamp_df['ItemID'] = timestamp_df['ItemID'].astype(str)

    timestamp_df.drop_duplicates()

    timestamp_df_with_timestamp = timestamp_df.copy()

    timestamp_df = timestamp_df[['UserID', 'ItemID']]

    timestamp_df = timestamp_df.groupby(['UserID', 'ItemID']).size().reset_index(name='Data')

    # Drop the abnormal data
    timestamp_df['Data'] = timestamp_df['Data'].apply(lambda x: 20 if x >= 20 else x)

    # Normalization
    max_value = timestamp_df['Data'].max()
    timestamp_df['Data'] = timestamp_df['Data'].apply(lambda x: x / max_value)

    timestamp_df = pd.merge(left=timestamp_df_with_timestamp, right=timestamp_df, how='left', on=['UserID', 'ItemID'])

    print(timestamp_df.head())
    # Create test/train splits
    rest_interactions, submission_explicit_interactions = merge_splits_without_overwrite_origin_dataset(timestamp_df,
                                                                                                        timestamp_array_submission_explicit)

    print(submission_explicit_interactions.head())
    print(submission_explicit_interactions.tail())

    submission_explicit_interactions.drop(timestamp_column, inplace=True, axis=1)
    submission_explicit_interactions.drop(timestamp_column, inplace=True, axis=1)

    manager.add_URM(submission_explicit_interactions, 'URM_submission_explicit_train')


if __name__ == "__main__":
    timestamp_list = [("2019-09-01", "2019-09-30")]
    transactions = pd.read_csv('../dataset/transactions_train.csv')
    print("Loaded transaction csv...")

    manager = DatasetMapperManager()
    split_train_validation_multiple_intervals(manager, transactions, timestamp_list, [("2020-08-01", "2020-08-02")])

    # generate dataset with URM (Implicit=True)
    dataset = manager.generate_Dataset(DATASET_NAME, True)
    print("Done! Saving dataset in processed/{}/".format(DATASET_NAME))
    dataset.save_data('./processed/{}/'.format(DATASET_NAME))
    print("Dataset stats:")
    dataset.print_statistics()
    dataset.print_statistics_global()
