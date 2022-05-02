from bayes_opt import BayesianOptimization
import os
from dotenv import load_dotenv

from bayes_opt import BayesianOptimization

from Recommenders.NonPersonalizedRecommender import explicit_TopPop
from Evaluation.Evaluator import EvaluatorHoldout

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset

from DataProcessing.split_train_validation_exponential_decay import *



# load .env file
load_dotenv()
DATASET_PATH = os.getenv('DATASET_PATH')

timestamp_list_train = [("2019-06-22", "2019-09-23")]
timestamp_list_validation = [("2019-09-23", "2019-09-30")]

transactions = pd.read_parquet('{}/processed_transactions_train.parquet'.format(DATASET_PATH))
articles = pd.read_parquet('{}/processed_articles.parquet'.format(DATASET_PATH))
customers = pd.read_parquet('{}/processed_customers.parquet'.format(DATASET_PATH))

print('Loaded all files')

hypertuning_params = {
    'decay': (5, 60)
}

n_cases = 10
n_random_starts = int(n_cases / 3)
cutoff_list = [12]
metric_to_optimize = "MAP"
cutoff_to_optimize = 12

def BO_func(decay):
    manager = DatasetMapperManager()
    split_train_validation_multiple_intervals(manager, transactions, timestamp_list_train, timestamp_list_validation, URM_train='URM_train', URM_validation='URM_validation', exponential_decay=decay)
    dataset = manager.generate_Dataset('hm', is_implicit=False)
    URM_train = dataset.get_URM_from_name('URM_train')
    URM_validation = dataset.get_URM_from_name('URM_validation')

    recommender = explicit_TopPop(URM_train)
    recommender.fit()

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list, verbose=False)
    result_map, _ = evaluator.evaluateRecommender(recommender)
    result_map = result_map.to_dict()["MAP"][12]
    return result_map

optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=hypertuning_params,
    verbose=5,
    random_state=5
)
optimizer.maximize(
    init_points=n_random_starts,
    n_iter=n_cases
)

import json

with open('result_experiments/exponential_decay_BO.json', 'w') as fp:
    json.dump(optimizer.max, fp)
