import pandas as pd
import os
from Data_manager.HMDatasetReader import HMDatasetReader
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.NonPersonalizedRecommender import TopPop, Random
from Recommenders.TopPop_weight_decayed import TopPop_weight_decayed
from Recommenders.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender

if __name__ == '__main__':
    dataset_name = "hm"
    reader = HMDatasetReader(False)
    DATASET_PATH = os.getenv('DATASET_PATH')
    PROCESSED_PATH = os.getenv('PROCESSED_PATH')

    dataset = reader.load_data('{}/processed_train_20190622_20190923_val_20190923_20190930_Explicit_and_exp/{}/'.format(DATASET_PATH, dataset_name))

    item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    user_original_ID_to_index_mapper = dataset.get_user_original_ID_to_index_mapper()
    mapper_inv = {value: key for key, value in item_original_ID_to_index_mapper.items()}

    print(max(item_original_ID_to_index_mapper.values()))

    transactions = pd.read_parquet('{}/processed_transactions_train.parquet'.format(DATASET_PATH))
    timestamp_submission_train = [("2020-06-22", "2020-09-23")]
    df_train = transactions.query(
        "'" + timestamp_submission_train[0][0] + "'<=t_dat<'" + timestamp_submission_train[0][1] + "'")

    URM_submission_train = dataset.get_URM_from_name('URM_submission_explicit_train')

    p3alphaRecommender = P3alphaRecommender(URM_submission_train)
    p3alphaRecommender.fit(topK=615, alpha=0.4603011612937017, normalize_similarity=True)

    rp3betaRecommender = RP3betaRecommender(URM_submission_train)
    rp3betaRecommender.fit(topK=694, alpha=0.3458962138661726, beta=0.07256855505772421, normalize_similarity=True)

    itemKNN_CFCBF_Hybrid_Recommenders_Top10 = ItemKNN_CFCBF_Hybrid_Recommender(URM_submission_train,
                                                                               dataset.get_loaded_ICM_dict()[
                                                                                   "ICM_mix_top_10_accTo_CBF"])
    itemKNN_CFCBF_Hybrid_Recommenders_Top10.fit(topK=663, shrink=900, similarity='asymmetric', normalize=True,
                                                asymmetric_alpha=0.03882135719640912, feature_weighting='TF-IDF',
                                                ICM_weight=0.14382621361392856)

    recommender = GeneralizedMergedHybridRecommender(URM_submission_train,
                                                     recommenders=[p3alphaRecommender, rp3betaRecommender,
                                                                   itemKNN_CFCBF_Hybrid_Recommenders_Top10])
    recommender.fit(alphas=[
        0.9136483036344931,
        0.543545719488054,
        0.89899546618579
    ])

    recommender_random = Random(URM_submission_train)
    recommender_random.fit()

    recommender_pop_weight = TopPop_weight_decayed()
    recommender_pop_weight.fit()

    path = os.getenv('DATASET_PATH')
    df_sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    save_path = os.path.join(path, "{}-submission-final.csv".format(recommender.RECOMMENDER_NAME))

    f = open(save_path, "w")
    f.write("customer_id,prediction\n")

    customer_list_train_submission = df_train.customer_id.unique().tolist()
    customer_list_sample_submission = df_sample_submission.customer_id.unique().tolist()
    remaining_customer_list = list(set(customer_list_sample_submission).difference(set(customer_list_train_submission)))

    ######## ignore out of stock
    df_articles = pd.read_parquet('{}/processed_articles.parquet'.format(DATASET_PATH))
    df_article_out_of_stock = df_articles.query("out_of_stock==1")[
        ['article_id', 'out_of_stock']]  # ['article_id'].unique().tolist()

    item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    df_article_out_of_stock['article_id_index'] = df_article_out_of_stock.apply(
        lambda x: item_original_ID_to_index_mapper[x.article_id], axis=1)

    print(df_article_out_of_stock)

    out_of_stock_list = df_article_out_of_stock['article_id_index'].unique().tolist()
    recommender.set_items_to_ignore(out_of_stock_list)

    #####################

    for i in df_train.customer_id.unique():
        i_index = user_original_ID_to_index_mapper[i]
        recommended_items = recommender.recommend(i_index, cutoff=12, remove_seen_flag=False,remove_custom_items_flag=True)
        well_formatted = " ".join([str(mapper_inv[x]) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, well_formatted))





    recommender_pop_weight.recommend(remaining_customer_list, f)
    for i in remaining_customer_list:
        recommended_items = recommender_random.recommend(i, cutoff=12, remove_seen_flag=False)
        well_formatted = " ".join([str(mapper_inv[x]) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, recommended_items))
    f.close()
    print("save complete")
