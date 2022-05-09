import numpy as np
import pandas as pd
from DataProcessing.split_train_validation_exponential_decay import exponential_decayed_result
from Data_manager.DatasetMapperManager import DatasetMapperManager

DATASET_NAME = 'hm-exponential-age-clustered-2020Train'

if __name__ == "__main__":

    manager = DatasetMapperManager()

    timestamp_df = pd.read_csv('../dataset/transactions_train.csv', dtype={'t_dat': object, 'customer_id': str,
                                                                           'article_id': str,
                                                                           'price': np.float64,
                                                                           'sales_channel_id': np.int32})
    print(timestamp_df.dtypes)
    customers_df = pd.read_csv('../dataset/customers.csv')

    print("Loaded transactions and customers...")

    timestamp_list = [("2020-06-22", "2020-09-23")]
    validation_timestamp = [("2018-09-29", "2018-09-30")]

    # Cleanup customer dataframe
    customers_df.drop(["FN", "Active", "club_member_status", "fashion_news_frequency", "postal_code"], axis=1,
                      inplace=True)
    # customers_df.drop_duplcates()
    # Define customer clusters
    age_clusters = [0, 20, 30, 50, 99]  # Intervals are right inclusive e.g. (20, 30] or (50, 99]


    def age_clustering(age):
        index = 1
        maxLength = len(age_clusters)
        while (age > age_clusters[index] and index < maxLength):
            index += 1
        return index

    print("Generating clusters...")
    customers_df['age_cluster'] = customers_df['age'].apply(age_clustering)

    print("Applying exponential decay...")
    train_df, validation_df = exponential_decayed_result(timestamp_df, timestamp_list, validation_timestamp, 16)

    # I create a URM for each age_cluster, then use explicitTopPop on each URM/cluster and create a list for each
    train_df.set_index(["UserID", "ItemID"], inplace=True)
    customers_df.rename(columns={"customer_id": "UserID"}, inplace=True)
    customers_df.set_index("UserID", inplace=True)

    print("Saving customers mappings...")
    # Save age cluster mapping
    # customers_df.to_csv("customers_age_group.csv")
    # print("Saved!")

    join_result = customers_df.join(train_df, how="left")
    join_result.reset_index(inplace=True)
    print(join_result)

    for k in range(1, len(age_clusters)):
        print("Creating {} cluster...".format(k))
        current_cluster = join_result[join_result["age_cluster"] == k]
        # print(set(current_cluster.columns))
        manager.add_URM(current_cluster, "URM_train_cluster_{}".format(k))

    manager.add_URM(validation_df, "URM_validation")
    dataset = manager.generate_Dataset(DATASET_NAME, False)
    print("Done! Saving dataset in processed/{}/".format(DATASET_NAME))
    dataset.save_data('../processed/{}/'.format(DATASET_NAME))





