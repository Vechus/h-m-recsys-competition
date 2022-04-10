import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

import os
import joblib
import re
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(path, file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_parquet(os.path.join(path, file))
    df = reduce_mem_usage(df)
    return df


def train(df_data):
    cols = [col for col in df_data.columns if label != col]

    folds = StratifiedKFold(n_splits=n_fold, random_state=seed, shuffle=True)
    es = early_stopping(1000)
    le = log_evaluation(period=100)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(df_data, df_data[label])):
        print(f"=====fold {fold}=======")

        df_train = df_data.loc[train_idx].reset_index(drop=True)
        df_val = df_data.loc[val_idx].reset_index(drop=True)

        print("train shape", df_train.shape, "test shape", df_val.shape)

        model = LGBMClassifier(random_state=seed, **lgbm)

        model.fit(df_train[cols], df_train[label],
                  eval_set=(df_val[cols], df_val[label]),
                  callbacks=[es, le],
                  eval_metric="auc"
                  )

        # validation
        val_pred = model.predict(df_val[cols])
        val_score = roc_auc_score(df_val[label], val_pred)
        scores.append(val_score)

        # save_model
        joblib.dump(model, f"lgbm_fold_{fold}.joblib")

    return scores


def get_feat_imp(df_data):
    imps_list = []
    cols = [col for col in df_data.columns if label != col]
    for _fold in range(n_fold):
        with open(f"lgbm_fold_{_fold}.joblib", "rb") as f:
            model = joblib.load(f)
        imps = model.feature_importances_
        imps_list.append(imps)

    imps = np.mean(imps_list, axis=0)
    df_imps = pd.DataFrame({"columns": df_data[cols].columns.tolist(), "feat_imp": imps})
    df_imps = df_imps.sort_values("feat_imp", ascending=False).reset_index(drop=True)

    return df_imps


def add_features(df_truth, df_false, is_article_added=True, is_customer_added=True):
    global df_truth_merge
    df_article = import_data(path, article_path)
    df_customer = import_data(path, customer_path)

    if is_article_added is True and is_customer_added is False:
        df_truth_merge = pd.merge(df_truth, df_article, how='left', on='article_id')
        del df_truth
        df_false_merge = pd.merge(df_false, df_article, how='left', on='article_id')
        del df_false

    if is_customer_added is True and is_article_added is False:
        df_truth_merge = pd.merge(df_truth, df_customer, how='left', on='customer_id')
        del df_truth
        df_false_merge = pd.merge(df_false, df_customer, how='left', on='customer_id')
        del df_false

    if is_customer_added is True and is_article_added is True:
        df_truth_merge = pd.merge(df_truth, df_article, how='left', on='article_id')
        del df_truth
        df_false_merge = pd.merge(df_false, df_article, how='left', on='article_id')
        del df_false

        df_truth_merge = pd.merge(df_truth_merge, df_customer, how='left', on='customer_id')
        df_false_merge = pd.merge(df_false_merge, df_customer, how='left', on='customer_id')

    return df_truth_merge, df_false_merge


if __name__ == "__main__":
    load_dotenv()
    path = os.getenv('DATASET_PATH')
    transaction_path = "transactions_train.csv"
    customer_path = "processed_customers.parquet"
    article_path = "processed_articles.parquet"
    # image_feat_path = "../input/h-and-m-swint-image-embedding/swin_tiny_patch4_window7_224_emb.csv.gz"

    cwd = os.getcwd()
    output_dir = path
    # start_date = '2020-08-01'
    start_date = '2020-01-01'

    image_feat_dim = 768
    text_feat_dim = 384

    # n_fold = 2
    n_fold = 5
    seed = 2022
    lgbm = {"n_estimators": 50}

    label = "label"

    df_trans = pd.read_csv(os.path.join(path, transaction_path), dtype={'article_id': str}, parse_dates=['t_dat'])
    df_trans = df_trans[df_trans["t_dat"] >= start_date].reset_index(drop=True)
    df_trans = df_trans[df_trans.columns.difference(['t_dat'])]
    df_trans = reduce_mem_usage(df_trans)

    df_truth = df_trans[["customer_id", "article_id"]]

    df_false = df_truth.copy()
    df_false.loc[:, "article_id"] = df_false["article_id"].sample(frac=1).tolist()

    df_truth.loc[:, label] = 1
    df_false.loc[:, label] = 0

    df_article = import_data(path, article_path)
    df_customer = import_data(path, customer_path)

    df_truth_merge, df_false_merge = add_features(df_truth, df_false)

    df_total = pd.concat([df_truth_merge, df_false_merge])
    df_total = df_total.drop(["customer_id", "article_id"], axis=1)

    print("#Ture: ", df_total[df_total["label"] == 1].shape, "#False ", df_total[df_total["label"] == 0].shape)

    print(df_total.columns)

    df_total.to_pickle("{}/feature{}.pkl".format(output_dir, str(start_date)))

    scores = train(df_total)

    print(scores)
    print(np.mean(scores))

    df_fea_imp = get_feat_imp(df_total)
    df_fea_imp.head(20)
