import pandas as pd
import os
from Data_manager.HMDatasetReader import HMDatasetReader
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.NonPersonalizedRecommender import TopPop, Random, ExplicitTopPopAgeClustered
from Recommenders.TopPop_weight_decayed import TopPop_weight_decayed

if __name__ == '__main__':
    dataset_name = "hm-exponential-age-clustered-2020Train"
    reader = HMDatasetReader(False)
    timestamp_column = 't_dat'
    # DATASET_PATH = os.getenv('DATASET_PATH')
    # PROCESSED_PATH = os.getenv('PROCESSED_PATH')

    dataset = reader.load_data('processed/{}/'.format(dataset_name))
    timestamp_df = pd.read_csv("dataset/transactions_train.csv")
    timestamp_df[timestamp_column] = pd.to_datetime(timestamp_df[timestamp_column], format='%Y-%m-%d')
    t1 = timestamp_df[timestamp_column].searchsorted("2020-06-22")
    t2 = timestamp_df[timestamp_column].searchsorted("2020-09-23")
    train_df = timestamp_df.iloc[t1:t2 - 1]

    item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    user_original_ID_to_index_mapper = dataset.get_user_original_ID_to_index_mapper()
    mapper_inv_items = {value: key for key, value in item_original_ID_to_index_mapper.items()}
    mapper_inv_users = {value: key for key, value in user_original_ID_to_index_mapper.items()}

    print(max(item_original_ID_to_index_mapper.values()))

    list_of_URMs = []
    print("Reading mapper for age_groups...")
    customer_mapper_df = pd.read_csv("customers_age_group.csv")
    customer_mapper_df.set_index("UserID", inplace=True)
    print(customer_mapper_df)

    for URM_name, URM in dataset.AVAILABLE_URM.items():
        if URM_name != "URM_validation":
            list_of_URMs.append(URM)

    recommender_object = ExplicitTopPopAgeClustered(list_of_URMs, dataset, customer_mapper_df)

    # recommender = TopPop(URM_submission_train)
    # recommender.fit()
    # recommender = TopPop(URM_submission_train)
    # recommender.fit()

    # recommender_pop_weight = TopPop_weight_decayed()

    # ######## ignore out of stock
    # df_articles = pd.read_parquet('{}/processed_articles.parquet'.format(DATASET_PATH))
    # df_article_out_of_stock = df_articles.query("out_of_stock==1")[
    #     ['article_id', 'out_of_stock']]  # ['article_id'].unique().tolist()
    #
    # item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    # df_article_out_of_stock['article_id_index'] = df_article_out_of_stock.apply(
    #     lambda x: item_original_ID_to_index_mapper[x.article_id], axis=1)
    #
    # print(df_article_out_of_stock)
    #
    # out_of_stock_list = df_article_out_of_stock['article_id_index'].unique().tolist()
    # recommender.set_items_to_ignore(out_of_stock_list)
    #
    # #####################
    recommender_object.fit()

    path = os.getenv('DATASET_PATH')
    df_sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    save_path = os.path.join(path, "{}-submission-final.csv".format("toppop_age_clustered"))

    f = open(save_path, "w")
    f.write("customer_id,prediction\n")
    # print(train_df["customer_id"].values)

    customer_list_train_submission = train_df.customer_id.unique().tolist()
    customer_list_sample_submission = df_sample_submission.customer_id.unique().tolist()
    remaining_customer_list = list(set(customer_list_sample_submission).difference(set(customer_list_train_submission)))
    print(f"{len(customer_list_train_submission)} customers to go through...")
    counter = 0
    for i in customer_list_train_submission:
        i_index = user_original_ID_to_index_mapper[i]
        recommended_items = recommender_object.recommend(i_index, cutoff=12, remove_seen_flag=True)
        well_formatted = " ".join(["0" + str(mapper_inv_items[x]) for x in recommended_items])
        f.write(f"{i},{well_formatted}\n")
        print("[%d] %s:%s" % (counter, i, well_formatted))
        counter += 1

    print("...Moving to cold customers...\n")

    for i in remaining_customer_list:
        recommended_items = recommender_object.recommend(i, cutoff=12, remove_seen_flag=False, useHMindex=True)
        well_formatted = " ".join(["0" + str(mapper_inv_items[x]) for x in recommended_items])
        f.write(f"{i},{well_formatted}\n")
        print("%s:%s" % (i, well_formatted))
        # remove_custom_items_flag=True)
    # f.write(recommender_object.recommend(df_sample_submission.customer_id.unique(), cutoff=12))
    f.close()
    print("save complete")
