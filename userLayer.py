import numpy as np
import scipy.sparse as sps
import pandas as pd
import os

from Data_manager.HMDatasetReader import HMDatasetReader
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout

dataset_name = "hm"
reader = HMDatasetReader(False)
DATASET_PATH = os.getenv('DATASET_PATH')
PROCESSED_PATH = os.getenv('PROCESSED_PATH')

dataset = reader.load_data('{}/processed_URM_20190622_20190923/{}/'.format(DATASET_PATH, dataset_name))

# URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.85)
ICM_name = "ICM_idxgrp_idx_prdtyp"
ICM_train = dataset.get_ICM_from_name(ICM_name)

UCM_name = "UCM_age"
UCM_train = dataset.get_loaded_UCM_dict()[UCM_name]

URM_train = dataset.get_URM_from_name('URM_train')
URM_validation = dataset.get_URM_from_name('URM_validation')
URM_test = URM_validation

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[12])
evaluator_test = EvaluatorHoldout(URM_validation, cutoff_list=[12])

profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
profile_length = [x for x in profile_length if x > 0]

block_size = int(len(profile_length) * 0.2)
sorted_users = np.argsort(profile_length)

print(block_size, sorted_users,max(profile_length))

for group_id in range(0, 5):
    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = np.array(profile_length)[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
        group_id,
        users_in_group.shape[0],
        users_in_group_p_len.mean(),
        np.median(users_in_group_p_len),
        users_in_group_p_len.min(),
        users_in_group_p_len.max()))

from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, \
    MultiThreadSLIM_SLIMElasticNetRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender

from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender

MAP_recommender_per_group = {}

collaborative_recommender_class = {
    "SLIMEN": MultiThreadSLIM_SLIMElasticNetRecommender,
    "TopPop": TopPop,
    "UserKNNCF": UserKNNCFRecommender,
    "ItemKNNCF": ItemKNNCFRecommender,
    "P3alpha": P3alphaRecommender,
    "RP3beta": RP3betaRecommender,
    "PureSVD": PureSVDRecommender,
}

hybird_recommender_class = { "ItemKNNCFCBF": ItemKNN_CFCBF_Hybrid_Recommender,
                            "UserKNNCFCBF": UserKNN_CFCBF_Hybrid_Recommender
                            }

recommender_object_dict = {}

for label, recommender_class in collaborative_recommender_class.items():
    recommender_object = recommender_class(URM_train)
    recommender_object.fit()
    recommender_object_dict[label] = recommender_object

for label, recommender_class in hybird_recommender_class.items():
    if label=="ItemKNNCFCBF":
        recommender_object = recommender_class(URM_train, ICM_train)
        recommender_object.fit()
        recommender_object_dict[label] = recommender_object
    else:
        recommender_object = recommender_class(URM_train, UCM_train)
        recommender_object.fit()
        recommender_object_dict[label] = recommender_object

cutoff = 12

for group_id in range(0, 5):

    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = np.array(profile_length)[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
        group_id,
        users_in_group.shape[0],
        users_in_group_p_len.mean(),
        np.median(users_in_group_p_len),
        users_in_group_p_len.min(),
        users_in_group_p_len.max()))

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

    for label, recommender in recommender_object_dict.items():
        result_df, _ = evaluator_test.evaluateRecommender(recommender)
        if label in MAP_recommender_per_group:
            MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
        else:
            MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]

import matplotlib.pyplot as plt

_ = plt.figure(figsize=(16, 9))
for label, recommender in recommender_object_dict.items():
    results = MAP_recommender_per_group[label]
    plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
plt.ylabel('MAP')
plt.xlabel('User Group')
plt.legend()
plt.show()
plt.savefig(os.path.join(DATASET_PATH, 'userwise.png'))


