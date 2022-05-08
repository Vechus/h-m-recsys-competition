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

    dataset = reader.load_data(
        '{}/processed_train_20190622_20190923_val_20190923_20190930_and_exp/{}/'.format(DATASET_PATH, dataset_name))
    # ICM_name = "ICM_idxgrp_idx_prdtyp"
    # ICM_train = dataset.get_ICM_from_name(ICM_name)

    item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    user_original_ID_to_index_mapper = dataset.get_user_original_ID_to_index_mapper()
    mapper_inv = {value: key for key, value in item_original_ID_to_index_mapper.items()}

    print(max(item_original_ID_to_index_mapper.values()))

    transactions = pd.read_parquet('{}/processed_transactions_train.parquet'.format(DATASET_PATH))
    timestamp_submission_train = [("2020-06-22", "2020-09-23")]
    df_train = transactions.query(
        "'" + timestamp_submission_train[0][0] + "'<=t_dat<'" + timestamp_submission_train[0][1] + "'")

    URM_submission_train = dataset.get_URM_from_name('URM_submission_train')

    # get URM_train, URM_test, URM_validation
    URM_train = dataset.get_URM_from_name('URM_train')
    # URM_test = dataset.get_URM_from_name('URM_test')
    # URM_validation = dataset.get_URM_from_name('URM_validation')

    URM_train_explicit = dataset.get_URM_from_name('URM_train_explicit')

    URM_train_exp = dataset.get_URM_from_name('URM_train_exp')
    # URM_validation_exp = dataset.get_URM_from_name('URM_validation_exp')

    toppop = TopPop(URM_submission_train)
    toppop.fit()

    toppop_exp = TopPop(URM_train_exp)
    toppop_exp.fit()

    p3alphaRecommender = P3alphaRecommender(URM_train)
    p3alphaRecommender.fit(topK=537, alpha=0.0, normalize_similarity=True)

    # rp3beta updated 27/04/22
    rp3betaRecommender = RP3betaRecommender(URM_train)
    rp3betaRecommender.fit(topK=626, alpha=0.21827333332714935, beta=0.0, normalize_similarity=True)

    #
    # #recommender = P3alphaRecommender(URM_submission_train)
    # #recommender.fit(topK=537, alpha=0.0, normalize_similarity=True)
    # rec1 = ItemKNNCBFRecommender(URM_submission_train, ICM_train=dataset.get_ICM_from_name('ICM_all'))
    # rec1.load_model(
    #         folder_path='result_experiments/ItemKNNCBF_CFCBF_URM_Train_2019-06-22_2019-09-23_Val_2019-09-23_2019-09-30/',
    #         file_name='ItemKNNCBFRecommender_ICM_all_cosine.zip')
    #
    # rec2 = P3alphaRecommender(URM_submission_train)
    # rec2.fit(topK=537, alpha=0.0, normalize_similarity=True)
    #
    # rec3 = RP3betaRecommender(URM_submission_train)
    # rec3.fit(topK=626, alpha=0.21827333332714935, beta=0.0, normalize_similarity=True)
    #
    # rec4 = ItemKNN_CFCBF_Hybrid_Recommender(URM_submission_train, ICM_train=dataset.get_ICM_from_name('ICM_mix_top_15_accTo_CBF'))
    # rec4.load_model(
    #         folder_path='result_experiments/ItemKNNCBF_CFCBF_URM_Train_2019-06-22_2019-09-23_Val_2019-09-23_2019-09-30/',
    #         file_name='ItemKNN_CFCBF_HybridRecommender_ICM_mix_top_15_asymmetric_best_model_last.zip')

    recommender = GeneralizedMergedHybridRecommender(URM_submission_train,
                                                     recommenders=[toppop_exp, p3alphaRecommender, rp3betaRecommender])
    recommender.fit(alphas=[
        8.443813078762297e-05,
        0.023386163942575533
    ])

    # recommender_random = Random(URM_submission_train)
    # recommender_random.fit()

    # recommender_pop_weight = TopPop_weight_decayed()
    # recommender_pop_weight.fit()

    path = os.getenv('DATASET_PATH')
    df_sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    save_path = os.path.join(path, "{}-submission-final-withTopWD_20220508_control_group.csv".format(
        recommender.RECOMMENDER_NAME))

    f = open(save_path, "w")
    f.write("customer_id,prediction\n")

    customer_list_train_submission = df_train.customer_id.unique().tolist()
    customer_list_sample_submission = df_sample_submission.customer_id.unique().tolist()
    remaining_customer_list = list(set(customer_list_sample_submission).difference(set(customer_list_train_submission)))

    ####### ignore out of stock
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
        recommended_items = recommender.recommend(i_index, cutoff=12, remove_seen_flag=False,
                                                  remove_custom_items_flag=True)
        well_formatted = " ".join([str(mapper_inv[x]) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, well_formatted))

    for i in remaining_customer_list:
        recommended_items = toppop.recommend(i, cutoff=12, remove_seen_flag=False)
        well_formatted = " ".join([str(mapper_inv[x]) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, recommended_items))
    f.close()
    print("save complete")
