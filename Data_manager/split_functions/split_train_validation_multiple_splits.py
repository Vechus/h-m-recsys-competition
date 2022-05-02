""""
Created 09/04/22
@author: Riccardo Pazzi
"""

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_user_wise, \
    split_train_in_two_percentage_global_sample


def split_multiple_times(URM_all, number_of_splits=1, train_percentage=0.1, is_global=True, keep_only_test=False):
    """
    The function splits an URM in multiple matrices selecting the number of interactions globally
    :param URM_all:
    :param number_of_splits: Number of random (train, test) pairs of matrices
    :param train_percentage:
    :param is_global: If True divides URM on the whole dataset, if False divides user-wise
    :param keep_only_test: If True discards train URMs in the returned list
    :return: list of TRAIN/TEST URMs or TEST URMs with selected train_percentage
    """

    test_list = []
    train_list = []

    if not is_global:
        for k in range(number_of_splits):

            train_URM, test_URM = split_train_in_two_percentage_user_wise(URM_all, train_percentage=train_percentage)
            if keep_only_test:
                test_list.append(test_URM)
            else:
                test_list.append(test_URM)
                train_list.append(train_URM)

    else:
        for k in range(number_of_splits):
            train_URM, test_URM = split_train_in_two_percentage_global_sample(URM_all,
                                                                              train_percentage=train_percentage)

            if keep_only_test:
                test_list.append(test_URM)
            else:
                test_list.append(test_URM)
                train_list.append(train_URM)

    return test_list, train_list


