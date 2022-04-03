#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_import_list import *

import traceback

import os, multiprocessing
from dotenv import load_dotenv
from Data_manager.HMDatasetReader import HMDatasetReader
from functools import partial

from Utils.Logger import Logger

from Data_manager.DataReader import DataReader
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, \
    runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')

    # dataReader = HMDatasetReader()
    # dataset = dataReader.load_data(save_folder_path=DATASET_PATH)

    dataset_name = "hm"
    reader = HMDatasetReader(False)

    PROCESSED_PATH = os.getenv('PROCESSED_PATH')
    dataset = reader.load_data('{}/processed/{}/'.format(DATASET_PATH, dataset_name))
    print("Loaded dataset into memory...")

    # get URM_train, URM_test, URM_validation
    URM_train = dataset.get_URM_from_name('URM_train')
    # URM_test = dataset.get_URM_from_name('URM_test')
    URM_validation = dataset.get_URM_from_name('URM_validation')

    # URM_train, URM_test = split_train_in_two_percentage_global_sample(dataset.get_URM_all(), train_percentage = 0.80)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.80)

    output_folder_path = "result_experiments/TopPop/URM_ALL/"

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    collaborative_algorithm_list = [
        # Random,
        TopPop,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
    ]

    from Evaluation.Evaluator import EvaluatorHoldout

    cutoff_list = [6, 12, 24]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 12

    n_cases = 50
    n_random_starts = int(n_cases / 3)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_test = None  # EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)

    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       metric_to_optimize=metric_to_optimize,
                                                       cutoff_to_optimize=cutoff_to_optimize,
                                                       n_cases=n_cases,
                                                       n_random_starts=n_random_starts,
                                                       evaluator_validation_earlystopping=evaluator_validation,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluate_on_test='no',
                                                       evaluator_test=None,
                                                       output_folder_path=output_folder_path,
                                                       resume_from_saved=True,
                                                       similarity_type_list=None,  # all
                                                       parallelizeKNN=False)

    #
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    #
    #
    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #

    # dataset.get_loaded_ICM_dict()

    ################################################################################################
    ###### Content Baselines
    # n = 0
    # for ICM_name, ICM_object in dataset.get_loaded_ICM_dict().items():
    #     n = n + 1
    #     if n not in [5, 10]:
    #         continue
    #     if ICM_name not in ['ICM_all']:
    #         continue
    #     try:
    #         runHyperparameterSearch_Content(ItemKNNCBFRecommender,
    #                                     URM_train = URM_train,
    #                                     URM_train_last_test = URM_train + URM_validation,
    #                                     metric_to_optimize = metric_to_optimize,
    #                                     cutoff_to_optimize = cutoff_to_optimize,
    #                                     evaluator_validation = evaluator_validation,
    #                                     evaluate_on_test='no',
    #                                     evaluator_test=None,
    #                                     output_folder_path = output_folder_path,
    #                                     parallelizeKNN = True,
    #                                     allow_weighting = True,
    #                                     resume_from_saved = True,
    #                                     similarity_type_list = None, # all
    #                                     ICM_name = ICM_name,
    #                                     ICM_object = ICM_object.copy(),
    #                                     n_cases = n_cases,
    #                                     n_random_starts = n_random_starts)
    #
    #     except Exception as e:
    #
    #         print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
    #         traceback.print_exc()
    #
    #     try:
    #
    #         runHyperparameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
    #                                        URM_train=URM_train,
    #                                        URM_train_last_test=URM_train + URM_validation,
    #                                        metric_to_optimize=metric_to_optimize,
    #                                        cutoff_to_optimize=cutoff_to_optimize,
    #                                        evaluator_validation=evaluator_validation,
    #                                        evaluate_on_test='no',
    #                                        evaluator_test=None,
    #                                        output_folder_path=output_folder_path,
    #                                        parallelizeKNN=True,
    #                                        allow_weighting=True,
    #                                        resume_from_saved=True,
    #                                        similarity_type_list=None,  # all
    #                                        ICM_name=ICM_name,
    #                                        ICM_object=ICM_object.copy(),
    #                                        n_cases=n_cases,
    #                                        n_random_starts=n_random_starts)
    #
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
    #         traceback.print_exc()


if __name__ == '__main__':

    log_for_telegram_group = True
    logger = Logger('HPS-test')
    if log_for_telegram_group:
        logger.log('Started Hyper-parameter tuning')
    print('Started Hyper-parameter tuning')
    try:
        read_data_split_and_search()
    except Exception as e:
        if log_for_telegram_group:
            logger.log('We got an exception! Check log and turn off the machine.')
            logger.log('Exception: \n{}'.format(str(e)))
        print('We got an exception! Check log and turn off the machine.')
        print('Exception: \n{}'.format(str(e)))
    if log_for_telegram_group:
        logger.log('Hyper parameter search finished! Check results and turn off the machine.')
    print('Hyper parameter search finished! Check results and turn off the machine.')
