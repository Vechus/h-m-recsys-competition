import pandas as pd
import os

from DataProcessing.split_train_validation_leave_timestamp_out import split_train_validation_multiple_intervals
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.HMDatasetReader import HMDatasetReader
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender

if __name__ == '__main__':
    dataset_name = "hm"
    reader = HMDatasetReader(False)
    DATASET_PATH = os.getenv('DATASET_PATH')
    PROCESSED_PATH = os.getenv('PROCESSED_PATH')
    dataset = reader.load_data('{}/processed/{}/'.format(DATASET_PATH, dataset_name))
    ICM_train = dataset.get_ICM_from_name("ICM_colour_group_code")

    manager = DatasetMapperManager()
    transactions = pd.read_csv('{}/processed_transactions_train.csv'.format(DATASET_PATH))
    timestamp_list_train = [("2020-06-22", "2020-09-23")]
    timestamp_list_validation = [("2019-09-23", "2019-09-30")]
    split_train_validation_multiple_intervals(manager, transactions, timestamp_list_train, timestamp_list_validation)
    dataset = manager.generate_Dataset('hm', True)

    URM_train = dataset.get_URM_from_name('URM_train')

    recommender = ItemKNN_CFCBF_Hybrid_Recommender(URM_train, ICM_train)
    recommender.fit(topK=677, shrink=36, similarity='asymmetric', normalize=True,
                    asymmetric_alpha=0.01578422391120027,
                    feature_weighting='none', ICM_weight=1.1442500121376953)

    path = os.getenv('DATASET_PATH')
    df_sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    save_path = os.path.join(path, "submission-final.csv")

    f = open(save_path, "w")
    f.write("customer_id,prediction\n")
    for t in df_sample_submission.customer_id:
        recommended_items = recommender.recommend(t, cutoff=12, remove_seen_flag=False)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{t}, {well_formatted}\n")
        print("%s:%s" % (t, well_formatted))
    f.close()
    print("save complete")
