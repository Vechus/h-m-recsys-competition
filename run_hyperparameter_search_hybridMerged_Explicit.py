import os
import threading
import traceback
from datetime import datetime

from bayes_opt import BayesianOptimization
from dotenv import load_dotenv

from Data_manager.HMDatasetReader import HMDatasetReader
from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP
from Recommenders.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
from Recommenders.Recommender_import_list import *
from Utils.Logger import Logger


def read_data_split_and_search_hybrid():
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')

    # dataReader = HMDatasetReader()
    # dataset = dataReader.load_data(save_folder_path=DATASET_PATH)

    dataset_name = "hm"
    reader = HMDatasetReader(False)

    dataset = reader.load_data(
        '{}/processed_train_20190622_20190923_val_20190923_20190930_Explicit_and_exp/hm/'.format(DATASET_PATH))
    print("Loaded dataset into memory...")

    # get URM_train, URM_test, URM_validation
    URM_train = dataset.get_URM_from_name('URM_train')
    # URM_test = dataset.get_URM_from_name('URM_test')
    URM_validation = dataset.get_URM_from_name('URM_validation')

    URM_train_explicit = dataset.get_URM_from_name('URM_train_explicit')
    URM_validation_explicit = dataset.get_URM_from_name('URM_validation_explicit')

    URM_train_exp = dataset.get_URM_from_name('URM_train_exp')
    URM_validation_exp = dataset.get_URM_from_name('URM_validation_exp')

    cutoff_list = [12]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 12

    n_cases = 50
    n_random_starts = int(n_cases / 3)

    toppop_exp = TopPop(URM_train_exp)
    toppop_exp.fit()

    toppop_explicit = TopPop(URM_train_explicit)
    toppop_explicit.fit()

    toppop_normal = TopPop(URM_train)
    toppop_normal.fit()

    p3alphaRecommender = P3alphaRecommender(URM_train_explicit)
    p3alphaRecommender.fit(topK=615, alpha=0.4603011612937017, normalize_similarity=True)

    rp3betaRecommender = RP3betaRecommender(URM_train_explicit)
    rp3betaRecommender.fit(topK=694, alpha=0.3458962138661726, beta=0.07256855505772421, normalize_similarity=True)

    itemKNN_CFCBF_Hybrid_Recommenders = ItemKNN_CFCBF_Hybrid_Recommender(URM_train_explicit,
                                                                         dataset.get_loaded_ICM_dict()[
                                                                             "ICM_mix_top_10_accTo_CBF"])
    itemKNN_CFCBF_Hybrid_Recommenders.fit(topK=663, shrink=900, similarity='asymmetric', normalize=True,
                                          asymmetric_alpha=0.03882135719640912, feature_weighting='TF-IDF',
                                          ICM_weight=0.14382621361392856)

    Hybrid_Recommenders_List = [
        # ItemKNNCBFRecommenders + ItemKNN_CFCBF_Hybrid_Recommenders,
        # ItemKNNCBFRecommenders + [p3alphaRecommender, rp3betaRecommender, ItemKNN_CFCBF_Hybrid_Recommenders[2]],
        # ItemKNNCBFRecommenders + [p3alphaRecommender, rp3betaRecommender],
        # [p3alphaRecommender, rp3betaRecommender],
        # ItemKNNCBFRecommenders + [rp3betaRecommender],
        # [toppop_normal, toppop_exp, ItemKNN_CFCBF_Hybrid_Recommenders[2]],
        # [toppop_exp, ItemKNN_CFCBF_Hybrid_Recommenders[2], ItemKNNCBFRecommenders[0]],
        # [toppop_exp, p3alphaRecommender, rp3betaRecommender],
        # [toppop_normal, toppop_explicit, toppop_exp],
        # [toppop_explicit, p3alphaRecommender, rp3betaRecommender],
        # [toppop_explicit, ItemKNN_CFCBF_Hybrid_Recommenders[2]],
        # [toppop_explicit, ItemKNN_CFCBF_Hybrid_Recommenders[2], ItemKNNCBFRecommenders[0]]
        [toppop_explicit, p3alphaRecommender, rp3betaRecommender, itemKNN_CFCBF_Hybrid_Recommenders],

    ]

    print('There are {} recommenders to hybridize'.format(len(Hybrid_Recommenders_List)))

    hybrid_recommenders = [
        [GeneralizedMergedHybridRecommender(URM_train_explicit.copy(), recommenders=recommenders.copy())] for
        recommenders in Hybrid_Recommenders_List]

    def hybrid_parameter_search(hybrid_recommender: list):  # list of GeneralizedMergedHybridRecommender
        evaluator_validation = K_Fold_Evaluator_MAP([URM_validation_explicit.copy()], cutoff_list=cutoff_list.copy(),
                                                    verbose=False)
        results = []

        tuning_params = {}
        for i in range(len(hybrid_recommender[0].recommenders)):
            tuning_params['hybrid{}'.format(i)] = (1e-2, 1)

        if len(hybrid_recommender[0].recommenders) == 2:

            def BO_func(
                    hybrid0,
                    hybrid1
            ):
                hybrid_recommender[0].fit(alphas=[
                    hybrid0,
                    hybrid1
                ])
                result = evaluator_validation.evaluateRecommender(hybrid_recommender)
                results.append(result)
                # print(result)
                return sum(result) / len(result)

        elif len(hybrid_recommender[0].recommenders) == 3:

            def BO_func(
                    hybrid0,
                    hybrid1,
                    hybrid2
            ):
                hybrid_recommender[0].fit(alphas=[
                    hybrid0,
                    hybrid1,
                    hybrid2
                ])
                result = evaluator_validation.evaluateRecommender(hybrid_recommender)
                results.append(result)
                # print(result)
                return sum(result) / len(result)

        elif len(hybrid_recommender[0].recommenders) == 4:

            def BO_func(
                    hybrid0,
                    hybrid1,
                    hybrid2,
                    hybrid3
            ):
                hybrid_recommender[0].fit(alphas=[
                    hybrid0,
                    hybrid1,
                    hybrid2,
                    hybrid3
                ])
                result = evaluator_validation.evaluateRecommender(hybrid_recommender)
                results.append(result)
                # print(result)
                return sum(result) / len(result)

        elif len(hybrid_recommender[0].recommenders) == 5:

            def BO_func(
                    hybrid0,
                    hybrid1,
                    hybrid2,
                    hybrid3,
                    hybrid4
            ):
                hybrid_recommender[0].fit(alphas=[
                    hybrid0,
                    hybrid1,
                    hybrid2,
                    hybrid3,
                    hybrid4
                ])
                result = evaluator_validation.evaluateRecommender(hybrid_recommender)
                results.append(result)
                # print(result)
                return sum(result) / len(result)

        optimizer = BayesianOptimization(
            f=BO_func,
            pbounds=tuning_params,
            verbose=5,
            random_state=5,
        )

        optimizer.maximize(
            init_points=n_random_starts,
            n_iter=n_cases,
        )

        import json

        with open("result_experiments/hybrid_Explicit/" + hybrid_recommender[0].RECOMMENDER_NAME + "_logs.json",
                  'w') as json_file:
            json.dump(optimizer.max, json_file)

        with open("result_experiments/hybrid_Explicit/" + hybrid_recommender[0].RECOMMENDER_NAME + "_all_logs.json",
                  'w') as json_file:
            json.dump(results, json_file)

    threads = []
    for recommender in hybrid_recommenders:
        threads.append(threading.Thread(target=hybrid_parameter_search, args=(recommender,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':

    # current date and time
    start = datetime.now()

    log_for_telegram_group = True
    logger = Logger('Hybrid - Start time:' + str(start))
    if log_for_telegram_group:
        logger.log('Started Hyper-parameter tuning. Hybrid recsys')
    print('Started Hyper-parameter tuning')
    try:
        read_data_split_and_search_hybrid()
    except Exception as e:
        if log_for_telegram_group:
            logger.log('We got an exception! Check log and turn off the machine.')
            logger.log('Exception: \n{}'.format(str(e)))
        print('We got an exception! Check log and turn off the machine.')
        print('Exception: \n{}'.format(str(e)))
        print(traceback.format_exc())
    if log_for_telegram_group:
        end = datetime.now()
        logger.log('Hyper parameter search finished! Check results and turn off the machine. '
                   'End time:' + str(end) + '  Program duration:' + str(end - start))
    print('Hyper parameter search finished! Check results and turn off the machine.')
