import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from lightgbm import LGBMRanker

import os
import joblib
import re
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn import preprocessing


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
    es = early_stopping(20)
    le = log_evaluation(period=100)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(df_data, df_data[label])):
        print(f"=====fold {fold}=======")
        print(train_idx, val_idx)


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


def get_table_feat(df):
    article_id_cols = ['product_code',
                       'product_type_no',
                       'graphical_appearance_no', 'colour_group_code',
                       'perceived_colour_value_id', 'perceived_colour_master_id',
                       'department_no', 'index_group_no', 'section_no',
                       'garment_group_no',
                       # 'cleaned_detail_desc',
                       'on_discount', 'sale_periods_months',
                       'autumn_sales_indicator',
        'out_of_stock',
        'is_for_male_or_female', 'is_for_mama',
                        'rebuy_ratio_article']

    article_dummy_cols = [#'cleaned_prod_name',
        'idxgrp_idx_prdtyp',
        'product_seasonal_type',
        # 'cleaned_department_name',
                          'cleaned_product_type_name',
                          'cleaned_product_group_name', 'cleaned_graphical_appearance_name',
                          'cleaned_colour_group_name', 'cleaned_perceived_colour_value_name',
                          'cleaned_perceived_colour_master_name', #'cleaned_department_name',
                          'cleaned_index_name', 'cleaned_index_group_name',
        'transaction_peak_year_month',
                          'cleaned_section_name', 'cleaned_garment_group_name']

    article_drop_cols = ["index_code",
                         "cleaned_prod_name",
                         "cleaned_detail_desc",
                         "cleaned_department_name",
                         ]

    df = df.drop(article_drop_cols, axis=1)
    df = pd.get_dummies(df, columns=article_dummy_cols)
    return df


def create_article_feat(df_article,
                        # df_image
                        ):
    # rename image
    # rename_dic = {f"{i}": f"image_col_{i}" for i in range(Config.image_feat_dim)}
    # df_image = df_image.rename(columns=rename_dic)

    df_article_feat = get_table_feat(df_article)
    # df_article_feat = df_article_feat.merge(df_image, on="article_id", how="left")

    return df_article_feat


def create_customer_feat(df):
    customer_drop_cols = [
        #"postal_code"
        ]
    customer_dummy_cols = ["club_member_status", "fashion_news_frequency"]

    df = df.drop(customer_drop_cols, axis=1)
    #     df.loc[:, "FN"] = df["FN"].fillna(0)
    #     df.loc[:, "Active"] = df["Active"].fillna(0)
    #     df.loc[:, "club_member_status"] = df["club_member_status"].fillna("NONE")
    #     df.loc[:, "fashion_news_frequency"] = df["fashion_news_frequency"].fillna("NONE")
    #     df.loc[:, "age"] = df["age"].fillna(0)
    df.loc[:, "age"] = np.log1p(df["age"])

    df = pd.get_dummies(df, columns=customer_dummy_cols)

    return df


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


def split_transaction(df_trans, start_date=None, end_date=None):
    if start_date is not None and end_date is None:
        df_trans = df_trans[df_trans["t_dat"] >= start_date].reset_index(drop=True)
    elif start_date is None and end_date is not None:
        df_trans = df_trans[df_trans["t_dat"] < end_date].reset_index(drop=True)
    else:
        df_trans = df_trans[(df_trans["t_dat"] >= start_date) & (df_trans["t_dat"] < end_date)].reset_index(drop=True)

    print("min date of transaction: ", min(df_trans["t_dat"]), "max date of transaction: ", max(df_trans["t_dat"]))
    return df_trans



def create_dataset_faster(df_truth, df_article_feat, df_customer_feat):
    df_data = pd.concat([
        df_truth.reset_index(drop=True),
        df_article_feat.reindex(df_truth['article_id'].values).reset_index(drop=True)
    ], axis=1)
    df_data = pd.concat([
        df_data.reset_index(drop=True),
        df_customer_feat.reindex(df_data['customer_id'].values).reset_index(drop=True)
    ], axis=1)

    df_data = df_data.drop(["customer_id", "article_id"], axis=1)

    return df_data


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
    start_date = '2020-06-15'
    end_date = '2020-09-23'

    # image_feat_dim = 768
    # text_feat_dim = 384

    # n_fold = 2
    n_fold = 15
    seed = 2022
    lgbm = {"n_estimators": 100}

    label = "label"

    df_trans = pd.read_csv(os.path.join(path, transaction_path), dtype={'article_id': str}, parse_dates=['t_dat'])
    df_trans = split_transaction(df_trans, start_date, end_date)
    df_trans = df_trans[df_trans.columns.difference(['t_dat'])]
    df_trans = reduce_mem_usage(df_trans)

    df_truth = df_trans[["customer_id", "article_id"]]

    df_false = df_truth.copy()
    df_false.loc[:, "article_id"] = df_false["article_id"].sample(frac=1).tolist()
    # df_false_copy = df_truth.copy()
    # df_false_copy = df_false_copy.sample(int(df_false.shape[0]))
    # df_false_copy.loc[:, "article_id"] = df_false_copy["article_id"].sample(frac=1).tolist()
    # df_false = pd.concat([df_false, df_false_copy], axis=0).drop_duplicates()

    df_truth.loc[:, label] = 1
    df_false.loc[:, label] = 0

    df_truth = pd.concat([df_truth, df_false])

    df_article = import_data(path, article_path)
    df_customer = import_data(path, customer_path)

    df_article_feat = create_article_feat(df_article)
    df_customer_feat = create_customer_feat(df_customer)

    del df_article
    del df_customer

    df_article_feat = df_article_feat.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    df_customer_feat = df_customer_feat.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    df_article_feat = df_article_feat.set_index("article_id")
    df_customer_feat = df_customer_feat.set_index("customer_id")

    df_data = create_dataset_faster(df_truth, df_article_feat, df_customer_feat)

    # print(df_data.head())

    print("#Ture: ", df_data[df_data["label"] == 1].shape,
          round(df_data[df_data["label"] == 1].shape[0] / df_data.shape[0],2), "#False: ",
          df_data[df_data["label"] == 0].shape,
          round(df_data[df_data["label"] == 0].shape[0] / df_data.shape[0],2))

    print(df_data.columns)
    #
    # df_total.to_pickle("{}/feature{}.pkl".format(output_dir, str(start_date)))
    #
    scores = train(df_data)

    print(scores)
    print(np.mean(scores))

    df_fea_imp = get_feat_imp(df_data)
    print(df_fea_imp.head(30))
    #
    # df_fea_imp.to_csv("{}/feature_importance_result{}.csv".format(output_dir, str(start_date)))