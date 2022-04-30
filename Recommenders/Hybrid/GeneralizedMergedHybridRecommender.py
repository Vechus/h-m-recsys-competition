from Recommenders.BaseRecommender import BaseRecommender
import numpy as np


class GeneralizedMergedHybridRecommender(BaseRecommender):
    """
    This recommender merges two recommendes by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedMergedHybridRecommender"

    def __init__(
            self,
            URM_train,
            recommenders: list,
            verbose=True
    ):
        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        super(GeneralizedMergedHybridRecommender, self).__init__(
            URM_train,
            verbose=verbose
        )

        self.recommenders = recommenders

    def fit(self, alphas=None):
        self.alphas = alphas

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_score = self.recommenders[0]._compute_item_score(user_id_array,items_to_compute)
        print('item_score[0]: \tmin {}\tmax {}\tmean {}\tstd {}'.format(item_score.min(), item_score.max(), item_score.mean(), item_score.std()))
        max_item_score = np.max(item_score)
        item_score = item_score / max_item_score
        result = self.alphas[0]*item_score
        for index in range(1,len(self.alphas)):
            item_score = self.recommenders[index]._compute_item_score(user_id_array,items_to_compute)
            print('item_score[{}]: \tmin {}\tmax {}\tmean {}\tstd {}'.format(index, item_score.min(), item_score.max(), item_score.mean(), item_score.std()))
            max_item_score = np.max(item_score)
            item_score = item_score / max_item_score
            result = result + self.alphas[index]*item_score
        return result