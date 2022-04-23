from Recommenders.Recommender_import_list import *

from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.split_functions.split_train_validation_multiple_splits import split_multiple_times
from Data_manager.HMDatasetReader import HMDatasetReader
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from Evaluation.Evaluator import EvaluatorHoldout, EvaluatorMultipleURMs
import traceback, os

dataset_name = "processed_URM_20190622_20190923/hm"
"""
Name of the folder inside processed where the dataset was saved with Dataset.save_data()
"""


def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):
    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


if __name__ == '__main__':

    reader = HMDatasetReader(False)

    # PROCESSED_PATH = os.getenv('PROCESSED_PATH')

    dataset_object = reader.load_data('{}/{}/'.format("dataset", dataset_name))
    print("Loaded dataset into memory...")
    # print(dataset_object.AVAILABLE_URM)
    # Here all URMs and ICMs must be loaded, if no URM_all is present an error will occur in Dataset library
    URM_train = dataset_object.get_URM_from_name('URM_train')
    URM_test = dataset_object.get_URM_from_name('URM_validation')
    for ICM_name, ICM_object in dataset_object.get_loaded_ICM_dict().items():
        print(ICM_name)
    UCM_all = []
    print(URM_train.shape)
    print(URM_test.shape)
    dataset_object.print_statistics_global()

    recommender_class_list = [
        ItemKNN_CFCBF_Hybrid_Recommender
    ]

    evaluator = EvaluatorHoldout(URM_test, [5, 12], exclude_seen=True)

    # Take random splits of 5% of total validation
    test_5_95_list = split_multiple_times(URM_test, 5, 0.95, keep_only_test=True)

    # Evaluate on all splits
    evaluator9_95 = EvaluatorMultipleURMs(test_5_95_list, [5, 12])

    # from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": EvaluatorHoldout(URM_test, [20], exclude_seen=True),
                              "lower_validations_allowed": 5,
                              "validation_metric": "MAP",
                              }

    output_root_path = "./result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    for recommender_class in recommender_class_list:

        try:

            print("Algorithm: {}".format(recommender_class))

            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)

            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15, **earlystopping_keywargs}
            else:
                fit_params = {}

            recommender_object.fit(**fit_params)

            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)
            results_5_95 = evaluator9_95.evaluate_with_statistics(recommender_object)


            # recommender_object.save_model(output_root_path, file_name="temp_model.zip")
            #
            # recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)
            # recommender_object.load_model(output_root_path, file_name="temp_model.zip")
            #
            # os.remove(output_root_path + "temp_model.zip")
            #
            # results_run_2, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)
            #
            # if recommender_class not in [Random]:
            #     assert results_run_1.equals(results_run_2)

            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            print("Algorithm: {}, list of MAP on 5 splits: {}".format(recommender_class, results_5_95))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender_class, results_run_string_1))
            logFile.write("Algorithm: {}, list of MAP on 5 splits: {}".format(recommender_class, results_5_95))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
