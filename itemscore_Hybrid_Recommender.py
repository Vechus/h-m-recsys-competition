from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps

from Recommenders.DataIO import DataIO
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from numpy import linalg as LA


class itemscore_Hybrid_Recommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender1, recommender2):
        super(itemscore_Hybrid_Recommender, self).__init__(URM_train)

        self.norm = 1
        self.URM_train = URM_train
        self.recommender_1 = recommender1
        self.recommender_2 = recommender2

    # def fit(self, norm ,alpha=0.5):
    #     self.alpha = alpha
    #     self.norm=norm

    def fit(self, norm, w1=1, w2=1):
        self.w1 = w1
        self.w2 = w2
        self.norm = norm

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array, items_to_compute=None)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array, items_to_compute=None)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)

        if norm_item_weights_1 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

        if norm_item_weights_2 == 0:
            raise ValueError(
                "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

        # item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (
        #             1 - self.alpha)
        item_weights = item_weights_1 / norm_item_weights_1 * self.w1 + item_weights_2 / norm_item_weights_2 * self.w2

        return item_weights

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # data_dict_to_save = {"alpha": self.alpha,
        #                      "norm":self.norm}
        data_dict_to_save = {"w1": self.w1,
                             "w2": self.w2,
                             "norm": self.norm}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")
