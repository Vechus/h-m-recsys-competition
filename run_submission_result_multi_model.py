import pandas as pd
import os
from Data_manager.HMDatasetReader import HMDatasetReader
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.NonPersonalizedRecommender import TopPop

if __name__ == '__main__':
    dataset_name = "hm"
    reader = HMDatasetReader(False)
    DATASET_PATH = os.getenv('DATASET_PATH')
    PROCESSED_PATH = os.getenv('PROCESSED_PATH')

    dataset = reader.load_data('{}/processed/{}/'.format(DATASET_PATH, dataset_name))
    ICM_name = "ICM_idxgrp_idx_prdtyp"
    ICM_train = dataset.get_ICM_from_name(ICM_name)

    item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    user_original_ID_to_index_mapper = dataset.get_user_original_ID_to_index_mapper()
    mapper_inv = {value: key for key, value in item_original_ID_to_index_mapper.items()}

    print(max(item_original_ID_to_index_mapper.values()))

    transactions = pd.read_parquet('{}/processed_transactions_train.parquet'.format(DATASET_PATH))
    timestamp_submission_train = [("2020-06-22", "2020-09-23")]
    df_train = transactions.query(
        "'" + timestamp_submission_train[0][0] + "'<=t_dat<'" + timestamp_submission_train[0][1] + "'")

    URM_submission_train = dataset.get_URM_from_name('URM_submission_train')

    recommender = ItemKNN_CFCBF_Hybrid_Recommender(URM_submission_train, ICM_train)
    recommender.fit(topK=748, shrink=317, similarity='tversky', normalize=True, tversky_alpha=0.0,
                    tversky_beta=1.6153392035973693,
                    ICM_weight=0.011213820771787788)

    recommender_pop = TopPop(URM_submission_train)
    recommender_pop.fit()

    path = os.getenv('DATASET_PATH')
    df_sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    save_path = os.path.join(path, "{}-submission-final-{}.csv".format(recommender.RECOMMENDER_NAME, ICM_name))

    f = open(save_path, "w")
    f.write("customer_id,prediction\n")

    customer_list_train_submission = df_train.customer_id.unique().tolist()
    customer_list_sample_submission = df_sample_submission.customer_id.unique().tolist()
    remaining_customer_list = list(set(customer_list_sample_submission).difference(set(customer_list_train_submission)))

    ######## ignore out of stock
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

    #####################

    for i in df_train.customer_id.unique():
        i_index = user_original_ID_to_index_mapper[i]
        recommended_items = recommender.recommend(i_index, cutoff=12, remove_seen_flag=False,
                                                  remove_custom_items_flag=True)
        well_formatted = " ".join([str(mapper_inv[x]) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, well_formatted))
    for i in remaining_customer_list:
        recommended_items = recommender_pop.recommend(i, cutoff=12, remove_seen_flag=False)
        well_formatted = " ".join([str(mapper_inv[x]) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, well_formatted))
    f.close()
    print("save complete")
