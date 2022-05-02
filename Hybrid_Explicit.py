import os
import traceback
from datetime import datetime
from Data_manager.HMDatasetReader import HMDatasetReader
from dotenv import load_dotenv

import numpy as np
from skopt.space import Real, Categorical, Integer

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Utils.Logger import Logger
from W_sparse_Hybrid_Recommender import W_sparse_Hybrid_Recommender
from itemscore_Hybrid_Recommender import itemscore_Hybrid_Recommender


def hyperparameter_tuning(recommender_class, recommender1, recommender2):
    if recommender_class is itemscore_Hybrid_Recommender:
        hyperparameters_range_dictionary = {"w1": Real(low=0, high=4.0, prior='uniform'),
                                            "w2": Real(low=0, high=4.0, prior='uniform'),
                                            "norm": Categorical([1, 2, np.inf])}

    if recommender_class is W_sparse_Hybrid_Recommender:
        hyperparameters_range_dictionary = {"alpha": Real(low=0, high=1.0, prior='uniform'),
                                            "norm": Categorical(['max']),
                                            # Categorical(['l1', 'l2', 'max']),
                                            "selectTopK": Categorical([True]),
                                            "topK": Integer(5, 2000)}

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_explicit, recommender1, recommender2],
        # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_explicit, recommender1, recommender2],
        # For a CBF model simply put [URM_train_validation, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    n_cases = 50
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 12

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                save_model="no",
                                output_folder_path=output_folder_path,  # Where to save the results
                                output_file_name_root=recommender_class.RECOMMENDER_NAME + "_" + recommender1.RECOMMENDER_NAME + "+" + recommender2.RECOMMENDER_NAME,
                                # recommender_class.RECOMMENDER_NAME + "_" + recommender1_object.RECOMMENDER_NAME + "+" + recommender2_object.RECOMMENDER_NAME,
                                # How to call the files
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                )


def search_hybrid():
    itemKNN_CFCBF_Hybrid_Recommenders = ItemKNN_CFCBF_Hybrid_Recommender(URM_train_explicit,
                                                                         dataset.get_loaded_ICM_dict()[
                                                                             "ICM_mix_top_10_accTo_CBF"])
    itemKNN_CFCBF_Hybrid_Recommenders.fit(topK=663, shrink=900, similarity='asymmetric', normalize=True,
                                          asymmetric_alpha=0.03882135719640912, feature_weighting='TF-IDF',
                                          ICM_weight=0.14382621361392856)

    p3alphaRecommender = P3alphaRecommender(URM_train_explicit)
    p3alphaRecommender.fit(topK=615, alpha=0.4603011612937017, normalize_similarity=True)

    rp3betaRecommender = RP3betaRecommender(URM_train_explicit)
    rp3betaRecommender.fit(topK=694, alpha=0.3458962138661726, beta=0.07256855505772421, normalize_similarity=True)

    # Hybrid_CFCBF_P3alpha = W_sparse_Hybrid_Recommender(URM_train_explicit, itemKNN_CFCBF_Hybrid_Recommenders, p3alphaRecommender)
    # Hybrid_CFCBF_P3alpha.fit(alpha=0.45553132119339973, norm='max', selectTopK=True, topK=1913)

    recommender_class = itemscore_Hybrid_Recommender
    # itemscore_Hybrid_Recommender
    # ItemKNN_CFCBF_Variant_Hybrid_Recommender
    # LinearCombination_normalized_Hybrid_Recommender
    # W_sparse_Hybrid_Recommender

    hyperparameter_tuning(recommender_class, itemKNN_CFCBF_Hybrid_Recommenders, p3alphaRecommender)


if __name__ == '__main__':

    start = datetime.now()

    log_for_telegram_group = True
    logger = Logger('Hybrid - Start time:' + str(start))
    if log_for_telegram_group:
        logger.log('Started Hyper-parameter tuning. Hybrid recsys')
    print('Started Hyper-parameter tuning')
    try:
        load_dotenv()
        DATASET_PATH = os.getenv('DATASET_PATH')

        reader = HMDatasetReader(False)

        dataset = reader.load_data(
            '{}/processed_train_20190622_20190923_val_20190923_20190930_Explict_By_Repeat_Purchase/hm/'.format(
                DATASET_PATH))
        print("Loaded dataset into memory...")

        URM_train_explicit = dataset.get_URM_from_name('URM_train')
        URM_validation_explicit = dataset.get_URM_from_name('URM_validation')

        evaluator_validation = EvaluatorHoldout(URM_validation_explicit, cutoff_list=[12])

        output_folder_path = "result_experiments/Hybrid_Explicit/"
        search_hybrid()
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
