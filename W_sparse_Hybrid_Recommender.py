from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
from sklearn.preprocessing import normalize

from Recommenders.DataIO import DataIO
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from numpy import linalg as LA
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from numpy import linalg as LA

class W_sparse_Hybrid_Recommender(ItemKNNCustomSimilarityRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "W_sparse_HybridRecommender"

    def __init__(self, URM_train,recommender1,recommender2):
        super(W_sparse_Hybrid_Recommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender1
        self.recommender_2 = recommender2

    def fit(self, alpha=0.5, norm='l1', **fit_args):
        self.alpha = alpha
        self.norm = norm

        W_sparse1=normalize(self.recommender_1.W_sparse, norm)
        W_sparse2=normalize(self.recommender_2.W_sparse, norm)

        self.W_sparse=(1 - alpha) * W_sparse1 + alpha * W_sparse2
        super(W_sparse_Hybrid_Recommender, self).fit(self.W_sparse,**fit_args)




    # def save_model(self, folder_path, file_name = None):
    #
    #     if file_name is None:
    #         file_name = self.RECOMMENDER_NAME
    #
    #     self._print("Saving model in file '{}'".format(folder_path + file_name))
    #
    #     data_dict_to_save = {"alpha": self.alpha, "selectTopK":self.selectTopK, "topK":self.topK}
    #
    #     dataIO = DataIO(folder_path=folder_path)
    #     dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)
    #
    #     self._print("Saving complete")
