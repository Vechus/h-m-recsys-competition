import numpy as np
from Recommenders.Recommender_utils import check_matrix

def linear_scaling_confidence(URM_train, alpha):
    C = check_matrix(URM_train, format="csr", dtype=np.float32)
    C.data = 1.0 + alpha * C.data

    return C