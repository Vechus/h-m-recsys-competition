#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Massimo Quadrana
@additions: explicit_TopPop, ExplicitTopPopAgeClustered Riccardo Pazzi
"""

import numpy as np
import pandas as pd

from Data_manager.HMDatasetReader import HMDatasetReader
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.DataIO import DataIO


class TopPop(BaseRecommender):
    """Top Popular recommender"""

    RECOMMENDER_NAME = "TopPopRecommender"

    def __init__(self, URM_train):
        super(TopPop, self).__init__(URM_train)

    def fit(self):

        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis=0)

        return item_scores

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


class explicit_TopPop(BaseRecommender):
    """Top Popular recommender which considers explicit values as quantities"""

    RECOMMENDER_NAME = "ExplicitTopPopRecommender"

    def __init__(self, URM_train):
        super(explicit_TopPop, self).__init__(URM_train)

    def fit(self):

        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        self.item_pop = self.URM_train.tocsc().sum(axis=0)
        self.n_items = self.URM_train.shape[1]

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis=0)

        return item_scores

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


class ExplicitTopPopAgeClustered(BaseRecommender):
    """
    Top
    Popular
    recommender
    which
    considers
    explicit
    values as quantities
    """

    RECOMMENDER_NAME = "ExplicitClusteredTopPopRecommender"

    def __init__(self, URM_list, dataset_object, age_cluster_mapper):
        super(ExplicitTopPopAgeClustered, self).__init__(URM_list[0])
        self.user_original_ID_to_index_mapper = dataset_object.get_user_original_ID_to_index_mapper()
        self.URM_list = URM_list
        self.item_pop = []
        self.n_items = []
        self.age_cluster_mapper = age_cluster_mapper  # HM User ID -> age_group

    def fit(self):
        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        for URM_train in self.URM_list:
            self.item_pop.append(URM_train.tocsc().sum(axis=0))
            self.n_items.append(URM_train.shape[1])

    def _compute_item_score(self, user_id_array, items_to_compute=None, useHMIndex=False):
        initialized = False
        if useHMIndex:
            for user_id in user_id_array:
                HM_user_id = user_id
                age_cluster_number = int(self.age_cluster_mapper.loc[HM_user_id]["age_cluster"])
                age_cluster_index = age_cluster_number - 1
                # The age cluster number starts from 1

                if items_to_compute is not None:
                    item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
                    item_pop_to_copy[items_to_compute] = self.item_pop[age_cluster_index][items_to_compute].copy()
                else:
                    item_pop_to_copy = self.item_pop[age_cluster_index].copy()

                item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
                if not initialized:
                    total_scores = item_scores
                    initialized = True
                else:
                    total_scores = np.vstack((total_scores, item_scores))
            # item_scores = np.repeat(item_scores, len(user_id_array), axis=0)
        else:
            mapper_inv = {value: key for key, value in self.user_original_ID_to_index_mapper.items()}
            # First I check mappings and see which group user_id belongs to
            for user_id in user_id_array:
                HM_user_id = mapper_inv[user_id]
                age_cluster_number = int(self.age_cluster_mapper.loc[HM_user_id]["age_cluster"])
                age_cluster_index = age_cluster_number - 1
                # The age cluster number starts from 1

                if items_to_compute is not None:
                    item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
                    item_pop_to_copy[items_to_compute] = self.item_pop[age_cluster_index][items_to_compute].copy()
                else:
                    item_pop_to_copy = self.item_pop[age_cluster_index].copy()

                item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
                if not initialized:
                    total_scores = item_scores
                    initialized = True
                else:
                    total_scores = np.vstack((total_scores, item_scores))
        # item_scores = np.repeat(item_scores, len(user_id_array), axis=0)

        return total_scores

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False,
                  useHMindex=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        cutoff = min(cutoff, self.URM_train.shape[1] - 1)

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        if useHMindex:
            scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute,
                                                    useHMIndex=True)
        else:
            scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


class GlobalEffects(BaseRecommender):
    """docstring for GlobalEffects"""

    RECOMMENDER_NAME = "GlobalEffectsRecommender"

    def __init__(self, URM_train):
        super(GlobalEffects, self).__init__(URM_train)

    def fit(self, lambda_user=10, lambda_item=25):

        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.n_items = self.URM_train.shape[1]

        # convert to csc matrix for faster column-wise sum
        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        # 1) global average
        self.mu = self.URM_train.data.sum(dtype=np.float32) / self.URM_train.data.shape[0]

        # 2) item average bias
        # compute the number of non-zero elements for each column
        col_nnz = np.diff(self.URM_train.indptr)

        # it is equivalent to:
        # col_nnz = X.indptr[1:] - X.indptr[:-1]
        # and it is **much faster** than
        # col_nnz = (X != 0).sum(axis=0)

        URM_train_unbiased = self.URM_train.copy()
        URM_train_unbiased.data -= self.mu
        self.item_bias = URM_train_unbiased.sum(axis=0) / (col_nnz + self.lambda_item)
        self.item_bias = np.asarray(self.item_bias).ravel()  # converts 2-d matrix to 1-d array without anycopy

        # 3) user average bias
        # NOTE: the user bias is *useless* for the sake of ranking items. We just show it here for educational purposes.

        # first subtract the item biases from each column
        # then repeat each element of the item bias vector a number of times equal to col_nnz
        # and subtract it from the data vector
        URM_train_unbiased.data -= np.repeat(self.item_bias, col_nnz)

        # now convert the csc matrix to csr for efficient row-wise computation
        URM_train_unbiased_csr = URM_train_unbiased.tocsr()
        row_nnz = np.diff(URM_train_unbiased_csr.indptr)
        # finally, let's compute the bias
        self.user_bias = URM_train_unbiased_csr.sum(axis=1).ravel() / (row_nnz + self.lambda_user)

        # 4) precompute the item ranking by using the item bias only
        # the global average and user bias won't change the ranking, so there is no need to use them
        # self.item_ranking = np.argsort(self.bi)[::-1]

        self.URM_train = check_matrix(self.URM_train, 'csr', dtype=np.float32)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_bias_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
            item_bias_to_copy[items_to_compute] = self.item_bias[items_to_compute].copy()
        else:
            item_bias_to_copy = self.item_bias.copy()

        item_scores = np.array(item_bias_to_copy, dtype=np.float).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis=0)

        return item_scores

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_bias": self.item_bias}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")


class Random(BaseRecommender):
    """Random recommender"""

    RECOMMENDER_NAME = "RandomRecommender"

    def __init__(self, URM_train):
        super(Random, self).__init__(URM_train)

    def fit(self, random_seed=42):
        np.random.seed(random_seed)
        self.n_items = self.URM_train.shape[1]

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a random block (len(user_id_array), n_items) array with the item score

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.random.rand(len(user_id_array), len(items_to_compute))

        else:
            item_scores = np.random.rand(len(user_id_array), self.n_items)

        return item_scores

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")
