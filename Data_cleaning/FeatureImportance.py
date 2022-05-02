import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from lightgbm import LGBMRanker

import os
import joblib
import re
from collections import Counter

from dotenv import load_dotenv
from sklearn import preprocessing
import tqdm


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


def train(df_data, categorical_features):
    cols = [col for col in df_data.columns if label != col]

    folds = StratifiedKFold(n_splits=n_fold, random_state=seed, shuffle=True)
    es = early_stopping(20)
    le = log_evaluation(period=100)
    f1_scores = []
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(df_data, df_data[label])):
        print(f"=====fold {fold}=======")
        print(train_idx, val_idx)

        df_train = df_data.loc[train_idx].reset_index(drop=True)
        df_val = df_data.loc[val_idx].reset_index(drop=True)

        print("train shape", df_train.shape, "test shape", df_val.shape)

        model = LGBMClassifier(random_state=seed, **lgbm)

        # model.set_params(**{"objective": f1_loss})

        model.fit(df_train[cols], df_train[label],
                  eval_set=(df_val[cols], df_val[label]),
                  callbacks=[es, le],
                  eval_metric="auc",
                  categorical_feature=['age_class_5']
                  )

        # validation
        val_pred = model.predict(df_val[cols])
        val_pred = val_pred.astype(np.int64)
        val_auc_score = roc_auc_score(df_val[label], val_pred)
        auc_scores.append(val_auc_score)

        # save_model
        joblib.dump(model, f"lgbm_fold_{fold}_0501_1.joblib")

    return auc_scores


def get_table_feat(df, df_transactions, end_date):
    df = df.drop(["rebuy_ratio_article", "mean_price", "max_price", "min_price", "median_price",
                  "buy_with_discount_ratio_article"], axis=1)
    df_transactions = df_transactions.query("t_dat<'" + end_date + "'")
    df_article_group = df_transactions.groupby(['article_id'])
    list2 = []
    for each in df_article_group:
        df_each = each[1]
        list1 = df_each['customer_id'].values.tolist()
        dict1 = Counter(list1)
        repetition = len([i for i in dict1.values() if i > 1]) / len(dict1)
        list2.append(repetition)
    dff_article = pd.DataFrame()
    dff_article['article_id'] = list(df_article_group.groups.keys())
    dff_article['rebuy_ratio_article'] = list2

    df = pd.merge(df, dff_article, how='left', on='article_id')
    print("rebuy_ratio_article added.")
    del dff_article

    df_article_price = df_transactions.groupby('article_id')['price'].agg(
        ['mean', 'max', 'min', 'median']).reset_index().rename(
        columns={'mean': 'mean_price', 'max': 'max_price', 'min': 'min_price', 'median': 'median_price'})
    df = pd.merge(df, df_article_price, how='left', on='article_id')
    print("prices added.")

    del df_article_price
    df_article_buy_discount = df_transactions.groupby('article_id')['buy_with_no_discount'].agg(['count', 'sum'])
    df_article_buy_discount['buy_with_discount_ratio_article'] = 1 - df_article_buy_discount['sum'] / \
                                                                 df_article_buy_discount['count']
    df_article_buy_discount = df_article_buy_discount.drop(['count', 'sum'], axis=1).reset_index()
    df = pd.merge(df, df_article_buy_discount, how='left', on='article_id')
    print("buy_with_discount_ratio_article added.")
    del df_article_buy_discount

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
                       'rebuy_ratio_article', 'mean_price', 'max_price', 'min_price', 'median_price',
                       'buy_with_discount_ratio_article']

    article_dummy_cols = [  # 'cleaned_prod_name',
        'idxgrp_idx_prdtyp',
        'product_seasonal_type',
        # 'cleaned_department_name',
        'cleaned_product_type_name',
        'cleaned_product_group_name', 'cleaned_graphical_appearance_name',
        'cleaned_colour_group_name', 'cleaned_perceived_colour_value_name',
        'cleaned_perceived_colour_master_name',  # 'cleaned_department_name',
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


def create_article_feat(df_article, df_transactions, end_date):
    df_article_feat = get_table_feat(df_article, df_transactions, end_date)
    df_article_feat.to_pickle("C:/Users/hezhitao/Desktop/article_features.pkl")

    return df_article_feat


def create_customer_feat(df, df_transactions, end_date):
    df = df.drop(["rebuy_ratio_customer", "buy_with_discount_ratio_customer"], axis=1)
    df_transactions = df_transactions.query("t_dat<'" + end_date + "'")
    print("max date: ", max(df_transactions['t_dat']))
    df_customer_group = df_transactions.groupby(['customer_id'])
    list2 = []
    for each in df_customer_group:
        df_each = each[1]
        list1 = df_each['article_id'].values.tolist()
        dict1 = Counter(list1)
        repetition = len([i for i in dict1.values() if i > 1]) / len(dict1)
        list2.append(repetition)
    dff_customer = pd.DataFrame()
    dff_customer['customer_id'] = list(df_customer_group.groups.keys())
    dff_customer['rebuy_ratio_customer'] = list2
    df = pd.merge(df, dff_customer, how='left', on='customer_id')
    print('rebuy_ratio_customer added.')

    del dff_customer

    df_customer_buy_discount = df_transactions.groupby('customer_id')['buy_with_no_discount'].agg(['count', 'sum'])
    df_customer_buy_discount['buy_with_discount_ratio_customer'] = 1 - df_customer_buy_discount['sum'] / \
                                                                   df_customer_buy_discount['count']
    df_customer_buy_discount = df_customer_buy_discount.drop(['count', 'sum'], axis=1).reset_index()
    df = pd.merge(df, df_customer_buy_discount, how='left', on='customer_id')
    print('buy_with_discount_ratio_customer added.')

    del df_customer_buy_discount

    customer_drop_cols = [
        # "postal_code"
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
    df = df.drop(['age', 'age_class_10'], axis=1)

    #df.to_parquet("C:/Users/hezhitao/Desktop/customer_features.parquet")
    df.to_pickle("C:/Users/hezhitao/Desktop/customer_features.pkl")

    return df


def get_feat_imp(df_data):
    imps_list = []
    cols = [col for col in df_data.columns if label != col]
    for _fold in range(n_fold):
        with open(f"lgbm_fold_{_fold}_0501_1.joblib", "rb") as f:
            model = joblib.load(f)
        imps = model.feature_importances_
        imps_list.append(imps)

    imps = np.mean(imps_list, axis=0)
    df_imps = pd.DataFrame({"columns": df_data[cols].columns.tolist(), "feat_imp": imps})
    df_imps = df_imps.sort_values("feat_imp", ascending=False).reset_index(drop=True)

    return df_imps


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

    lbl = preprocessing.LabelEncoder()
    df_data['article_id'] = lbl.fit_transform(df_data['article_id'].astype(str))
    df_data['customer_id'] = lbl.fit_transform(df_data['customer_id'].astype(str))

    df_data = df_data.drop(["customer_id", "article_id"], axis=1)

    return df_data



def inference(df_submission, df_article, df_article_feat, df_customer_feat, models, cols):
    article_candidates = []

    for customer in tqdm.tqdm(df_submission["customer_id"]):
        _df = df_article.copy()
        _df.loc[:, "customer_id"] = customer
        _df = create_dataset_faster(_df, df_article_feat, df_customer_feat)
        _df = _df[cols]

        preds = []
        for _fold in range(n_fold):
            pred = models[_fold].predict_proba(_df, num_iteration=models[_fold]._best_iteration)[:, 1]
            preds.append(pred)

        pred = np.mean(preds, axis=0)
        df_pred = pd.DataFrame({"article_id": df_article["article_id"].tolist(), "score": pred})

        df_pred = df_pred.sort_values("score", ascending=False).reset_index(drop=True)
        df_pred = df_pred.head(12)
        pred_str = [str(pred) for pred in df_pred["article_id"].tolist()]
        article_candidates.append(" ".join(pred_str))

    df_submission.loc[:, "prediction"] = article_candidates

    return df_submission


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
    start_date_train = '2020-09-15'
    end_date_train = '2020-09-23'

    n_fold = 15
    seed = 2022
    lgbm = {"n_estimators": 100}

    label = "label"

    df_trans = pd.read_csv(os.path.join(path, transaction_path), dtype={'article_id': str}, parse_dates=['t_dat'])
    df_trans_all = pd.read_parquet(os.path.join(path, "processed_transactions_train.parquet"))
    df_trans = split_transaction(df_trans, start_date_train, end_date_train)
    df_trans = df_trans[df_trans.columns.difference(['t_dat'])]
    df_trans = reduce_mem_usage(df_trans)

    df_truth = df_trans[["customer_id", "article_id"]]

    df_false = df_truth.copy()
    df_false = df_false.sample(int(df_false.shape[0]))
    df_false.loc[:, "article_id"] = df_false["article_id"].sample(frac=1).tolist()

    df_truth.loc[:, label] = 1
    df_false.loc[:, label] = 0

    df_truth = pd.concat([df_truth, df_false])

    df_article = import_data(path, article_path)
    df_customer = import_data(path, customer_path)

    df_article_feat_train = create_article_feat(df_article, df_trans_all, end_date_train)
    df_customer_feat_train = create_customer_feat(df_customer, df_trans_all, end_date_train)

    del df_article
    del df_customer
    del df_trans_all

    df_article_feat_train = df_article_feat_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    df_customer_feat_train = df_customer_feat_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    df_article_feat_train = df_article_feat_train.set_index("article_id")
    df_customer_feat_train = df_customer_feat_train.set_index("customer_id")

    df_data = create_dataset_faster(df_truth, df_article_feat_train, df_customer_feat_train)

    # print(df_data.head())

    print("#Ture: ", df_data[df_data["label"] == 1].shape,
          round(df_data[df_data["label"] == 1].shape[0] / df_data.shape[0], 2), "#False: ",
          df_data[df_data["label"] == 0].shape,
          round(df_data[df_data["label"] == 0].shape[0] / df_data.shape[0], 2))

    print(df_data.columns)
    #
    # df_total.to_pickle("{}/feature{}.pkl".format(output_dir, str(start_date)))
    #

    categorical_features_article = ["article_id",
                                    # 'product_code',
                                    # 'product_type_no',
                                    # 'graphical_appearance_no', 'colour_group_code',
                                    # 'perceived_colour_value_id', 'perceived_colour_master_id',
                                    # 'department_no', 'index_group_no', 'section_no',
                                    # 'garment_group_no',
                                    # # 'cleaned_detail_desc',
                                    # 'on_discount', 'sale_periods_months',
                                    # 'out_of_stock',
                                    # 'is_for_male_or_female', 'is_for_mama',
                                    # 'idxgrp_idx_prdtyp',
                                    # 'product_seasonal_type',
                                    # # 'cleaned_department_name',
                                    # 'cleaned_product_type_name',
                                    # 'cleaned_product_group_name', 'cleaned_graphical_appearance_name',
                                    # 'cleaned_colour_group_name', 'cleaned_perceived_colour_value_name',
                                    # 'cleaned_perceived_colour_master_name',  # 'cleaned_department_name',
                                    # 'cleaned_index_name', 'cleaned_index_group_name',
                                    # 'transaction_peak_year_month',
                                    # 'cleaned_section_name', 'cleaned_garment_group_name'
                                    ]
    categorical_features_customer = [
        "customer_id",
        # "club_member_status", "fashion_news_frequency",
        # "age", "postal_code", "FN", "Active"
    ]

    categorical_features = categorical_features_customer + categorical_features_article

    scores = train(df_data, categorical_features)

    print("auc_scores: ", scores)
    print("auc_mean_score: ", np.mean(scores))

    df_fea_imp = get_feat_imp(df_data)
    print(df_fea_imp.head(30))

    sample_submission_path = "sample_submission.csv"
    df_submission = pd.read_csv(os.path.join(path, sample_submission_path))
    print(df_submission.head())

    df_article = import_data(article_path)
    df_article = df_article[["article_id"]]

    models = []
    for _fold in range(n_fold):
        with open(f"lgbm_fold_{_fold}_0501_1.joblib", "rb") as f:
            model = joblib.load(f)
            models.append(model)

    cols = [col for col in df_data.columns if label != col]
    df_sub = inference(df_submission.head(10), df_article, df_article_feat_train, df_customer_feat_train, models, cols)

    df_sub.to_csv(os.path.join(path,"submit.csv"), index=None)
