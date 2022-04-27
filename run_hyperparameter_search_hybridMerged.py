from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Recommender_import_list import *

import traceback

import os, multiprocessing
from dotenv import load_dotenv
from Data_manager.HMDatasetReader import HMDatasetReader
from functools import partial

from Utils.Logger import Logger
from datetime import datetime

from Data_manager.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, \
    runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from bayes_opt import BayesianOptimization
from Recommenders.Hybrid.GeneralizedMergedHybridRecommender import GeneralizedMergedHybridRecommender
from Evaluation.K_Fold_Evaluator import K_Fold_Evaluator_MAP

def read_data_split_and_search_hybrid():
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')

    # dataReader = HMDatasetReader()
    # dataset = dataReader.load_data(save_folder_path=DATASET_PATH)

    dataset_name = "hm"
    reader = HMDatasetReader(False)

    PROCESSED_PATH = os.getenv('PROCESSED_PATH')
    dataset = reader.load_data('{}/processed_train-20190622_20190923-val-20190923_20190930/hm/'.format(DATASET_PATH))
    print("Loaded dataset into memory...")

    # get URM_train, URM_test, URM_validation
    URM_train = dataset.get_URM_from_name('URM_train')
    # URM_test = dataset.get_URM_from_name('URM_test')
    URM_validation = dataset.get_URM_from_name('URM_validation')

    # URM_train, URM_test = split_train_in_two_percentage_global_sample(dataset.get_URM_all(), train_percentage = 0.80)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

    output_folder_path = "result_experiments/collaborative_algorithm_URM_2019-06-22_2019-09-23/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    cutoff_list = [12]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 12

    n_cases = 50
    n_random_starts = int(n_cases / 3)

    evaluator_validation = K_Fold_Evaluator_MAP([URM_validation], cutoff_list=cutoff_list, verbose=False)

    # |  74       |  0.003248 |  0.0      |  1.006e-0 |  5.551e-1 |  1.0      |
    # p3alpha updated 27/04/22
    p3alphaRecommender = P3alphaRecommender(URM_train)
    p3alphaRecommender.fit(topK=537, alpha=0.0, normalize_similarity=True)

    # rp3beta updated 27/04/22
    rp3betaRecommender = RP3betaRecommender(URM_train)
    rp3betaRecommender.fit(topK=626, alpha=0.21827333332714935, beta=0.0, normalize_similarity=True)

    ItemKNNCBFRecommenders_ICMs = [
        'ICM_all'
    ]
    ItemKNNCBFRecommenders_Filenames = [
        'ItemKNNCBFRecommender_ICM_all_cosine.zip'
    ]
    ItemKNNCBFRecommenders = [
        ItemKNNCBFRecommender(URM_train, ICM_train=dataset.get_ICM_from_name(icm)) for icm in
        ItemKNNCBFRecommenders_ICMs
    ]
    for i in range(len(ItemKNNCBFRecommenders_ICMs)):
        ItemKNNCBFRecommenders[i].load_model(
            folder_path='result_experiments/ItemKNNCBF_CFCBF_URM_Train_2019-06-22_2019-09-23_Val_2019-09-23_2019-09-30/',
            file_name=ItemKNNCBFRecommenders_Filenames[i])

    ItemKNN_CFCBF_Hybrid_Recommenders_ICMs = [
        'ICM_mix_top_10_accTo_CBF',
        'ICM_mix_top_5_accTo_CBF',
        'ICM_mix_top_15_accTo_CBF'
    ]
    ItemKNN_CFCBF_Hybrid_Recommenders_Filenames = [
        'ItemKNN_CFCBF_HybridRecommender_ICM_mix_top_10_asymmetric_best_model_last.zip',
        'ItemKNN_CFCBF_HybridRecommender_ICM_mix_top_5_tversky_best_model_last.zip',
        'ItemKNN_CFCBF_HybridRecommender_ICM_mix_top_15_asymmetric_best_model_last.zip'
    ]
    ItemKNN_CFCBF_Hybrid_Recommenders = [
        ItemKNN_CFCBF_Hybrid_Recommender(URM_train, ICM_train=dataset.get_ICM_from_name(icm)) for icm in
        ItemKNN_CFCBF_Hybrid_Recommenders_ICMs
    ]

    for i in range(len(ItemKNN_CFCBF_Hybrid_Recommenders_ICMs)):
        ItemKNN_CFCBF_Hybrid_Recommenders[i].load_model(
            folder_path='result_experiments/ItemKNNCBF_CFCBF_URM_Train_2019-06-22_2019-09-23_Val_2019-09-23_2019-09-30/',
            file_name=ItemKNN_CFCBF_Hybrid_Recommenders_Filenames[i])

    #best_recommenders = ItemKNNCBFRecommenders + ItemKNN_CFCBF_Hybrid_Recommenders
    #best_recommenders = ItemKNNCBFRecommenders + [p3alphaRecommender, rp3betaRecommender, ItemKNN_CFCBF_Hybrid_Recommenders[2]]
    best_recommenders = ItemKNNCBFRecommenders + [p3alphaRecommender, rp3betaRecommender]

    tuning_params = {}
    for i in range(len(best_recommenders)):
        tuning_params['hybrid{}'.format(i)] = (0, 1)

    print('There are {} recommenders to hybridize!'.format(len(best_recommenders)))
    print(str(tuning_params))

    results = []
    hybrid_recommender = [GeneralizedMergedHybridRecommender(URM_train, recommenders=best_recommenders)]

    def BO_func(
            hybrid0,
            hybrid1,
            hybrid2,
            #hybrid3
    ):
        hybrid_recommender[0].fit(alphas=[
            hybrid0,
            hybrid1,
            hybrid2,
            #hybrid3
        ])
        result = evaluator_validation.evaluateRecommender(hybrid_recommender)
        results.append(result)
        print(result)
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

    with open("result_experiments/hybrid/" + hybrid_recommender[0].RECOMMENDER_NAME + "_logs.json", 'w') as json_file:
        json.dump(optimizer.max, json_file)

    with open("result_experiments/hybrid/" + hybrid_recommender.RECOMMENDER_NAME + "_all_logs.json", 'w') as json_file:
        json.dump(results, json_file)


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
