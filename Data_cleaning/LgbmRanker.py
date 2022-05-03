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

from lightgbm.sklearn import LGBMRanker
from datetime import timedelta

from tqdm import tqdm


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


def create_article_feat(df_article, df_transactions, end_date, path):
    df_article_feat = get_table_feat(df_article, df_transactions, end_date)
    df_article_feat.to_pickle(os.path.join(path, "article_features.pkl"))

    return df_article_feat


def create_customer_feat(df, df_transactions, end_date, path):
    df = df.drop(["rebuy_ratio_customer", "buy_with_discount_ratio_customer"], axis=1)
    df_transactions = df_transactions.query("t_dat<'" + end_date + "'")
    # print("max date: ", max(df_transactions['t_dat']))
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
    df.loc[:, "age"] = np.log1p(df["age"])
    df = pd.get_dummies(df, columns=customer_dummy_cols)
    # df = df.drop(['age', 'age_class_10'], axis=1)

    df.to_pickle(os.path.join(path, "customer_features.pkl"))

    return df


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


def prepare_candidates(customers_id, n_candidates=12):
    """
  df - basically, dataframe with customers(customers should be unique)
  """
    prediction_dict = {}
    dummy_list = list((df_2w['article_id'].value_counts()).index)[:n_candidates]

    for i, cust_id in tqdm(enumerate(customers_id)):
        # comment this for validation
        if cust_id in purchase_dict_1w:
            #         print(purchase_dict_1w[cust_id])
            l = sorted((purchase_dict_1w[cust_id]).items(), key=lambda x: x[1], reverse=True)
            #         print(l)
            l = [y[0] for y in l]
            if len(l) > n_candidates:
                s = l[:n_candidates]
            else:
                dummy_list_1w_new = list(set(dummy_list_1w) - set(l))
                s = l + dummy_list_1w_new[:(n_candidates - len(l))]
        elif cust_id in purchase_dict_2w:
            l = sorted((purchase_dict_2w[cust_id]).items(), key=lambda x: x[1], reverse=True)
            l = [y[0] for y in l]
            if len(l) > n_candidates:
                s = l[:n_candidates]
            else:
                dummy_list_2w_new = list(set(dummy_list_2w) - set(l))
                s = l + dummy_list_2w_new[:(n_candidates - len(l))]
        elif cust_id in purchase_dict_3w:
            l = sorted((purchase_dict_3w[cust_id]).items(), key=lambda x: x[1], reverse=True)
            l = [y[0] for y in l]
            if len(l) > n_candidates:
                s = l[:n_candidates]
            else:
                dummy_list_3w_new = list(set(dummy_list_3w) - set(l))
                s = l + dummy_list_3w_new[:(n_candidates - len(l))]
        elif cust_id in purchase_dict_4w:
            l = sorted((purchase_dict_4w[cust_id]).items(), key=lambda x: x[1], reverse=True)
            l = [y[0] for y in l]
            if len(l) > n_candidates:
                s = l[:n_candidates]
            else:
                dummy_list_4w_new = list(set(dummy_list_4w) - set(l))
                s = l + dummy_list_4w_new[:(n_candidates - len(l))]
        else:
            s = dummy_list
        prediction_dict[cust_id] = s

    k = list(map(lambda x: x[0], prediction_dict.items()))
    v = list(map(lambda x: x[1], prediction_dict.items()))
    negatives_df = pd.DataFrame({'customer_id': k, 'negatives': v})
    negatives_df = (
        negatives_df
            .explode('negatives')
            .rename(columns={'negatives': 'article_id'})
    )
    return negatives_df


if __name__ == "__main__":
    load_dotenv()
    path = os.getenv('DATASET_PATH')
    transaction_path = "transactions_train.csv"
    customer_path = "processed_customers.parquet"
    article_path = "processed_articles.parquet"

    cwd = os.getcwd()
    output_dir = path

    start_date_train = '2020-08-15'
    end_date_train = '2020-09-23'
    end_date_validation = '2020-09-23'

    label = "label"

    df_trans = pd.read_csv(os.path.join(path, transaction_path), dtype={'article_id': str}, parse_dates=['t_dat'])
    df_trans_all = pd.read_parquet(os.path.join(path, "processed_transactions_train.parquet"))
    # df_trans = split_transaction(df_trans, start_date_train, end_date_train)
    # df_trans = df_trans[df_trans.columns.difference(['t_dat'])]
    # df_trans = reduce_mem_usage(df_trans)

    df_article = import_data(path, article_path)
    df_customer = import_data(path, customer_path)

    df_article_feat_train = create_article_feat(df_article, df_trans_all, end_date_train, path)
    df_customer_feat_train = create_customer_feat(df_customer, df_trans_all, end_date_train, path)

    del df_article
    del df_customer
    del df_trans_all

    user_features = df_customer_feat_train
    print(user_features)
    item_features = df_article_feat_train
    transactions_df = df_trans

    df_4w = transactions_df[transactions_df['t_dat'] >= pd.to_datetime('2020-08-24')].copy()
    df_3w = transactions_df[transactions_df['t_dat'] >= pd.to_datetime('2020-08-31')].copy()
    df_2w = transactions_df[transactions_df['t_dat'] >= pd.to_datetime('2020-09-07')].copy()
    df_1w = transactions_df[transactions_df['t_dat'] >= pd.to_datetime('2020-09-15')].copy()

    train = transactions_df.loc[(transactions_df.t_dat >= pd.to_datetime(start_date_train)) &
                                (transactions_df.t_dat < pd.to_datetime(end_date_train))]
    valid = transactions_df.loc[(transactions_df.t_dat >= pd.to_datetime(end_date_train)) &
                                (transactions_df.t_dat < pd.to_datetime(end_date_validation))]

    train = (train
             .merge(user_features, on=('customer_id'))
             .merge(item_features, on=('article_id'))
             )
    train.sort_values(['t_dat', 'customer_id'], inplace=True)

    valid = (valid
             .merge(user_features, on=('customer_id'))
             .merge(item_features, on=('article_id'))
             )
    valid.sort_values(['t_dat', 'customer_id'], inplace=True)

    del transactions_df

    print("train shape: ", train.shape, "validation shape: ", valid.shape)

    purchase_dict_4w = {}

    for i, x in enumerate(zip(df_4w['customer_id'], df_4w['article_id'])):
        cust_id, art_id = x
        if cust_id not in purchase_dict_4w:
            purchase_dict_4w[cust_id] = {}

        if art_id not in purchase_dict_4w[cust_id]:
            purchase_dict_4w[cust_id][art_id] = 0

        purchase_dict_4w[cust_id][art_id] += 1

    dummy_list_4w = list((df_4w['article_id'].value_counts()).index)[:12]

    purchase_dict_3w = {}

    for i, x in enumerate(zip(df_3w['customer_id'], df_3w['article_id'])):
        cust_id, art_id = x
        if cust_id not in purchase_dict_3w:
            purchase_dict_3w[cust_id] = {}

        if art_id not in purchase_dict_3w[cust_id]:
            purchase_dict_3w[cust_id][art_id] = 0

        purchase_dict_3w[cust_id][art_id] += 1

    dummy_list_3w = list((df_3w['article_id'].value_counts()).index)[:12]

    purchase_dict_2w = {}

    for i, x in enumerate(zip(df_2w['customer_id'], df_2w['article_id'])):
        cust_id, art_id = x
        if cust_id not in purchase_dict_2w:
            purchase_dict_2w[cust_id] = {}

        if art_id not in purchase_dict_2w[cust_id]:
            purchase_dict_2w[cust_id][art_id] = 0

        purchase_dict_2w[cust_id][art_id] += 1

    dummy_list_2w = list((df_2w['article_id'].value_counts()).index)[:12]

    purchase_dict_1w = {}

    for i, x in enumerate(zip(df_1w['customer_id'], df_1w['article_id'])):
        cust_id, art_id = x
        if cust_id not in purchase_dict_1w:
            purchase_dict_1w[cust_id] = {}

        if art_id not in purchase_dict_1w[cust_id]:
            purchase_dict_1w[cust_id][art_id] = 0

        purchase_dict_1w[cust_id][art_id] += 1

    dummy_list_1w = list((df_1w['article_id'].value_counts()).index)[:12]

    # take only last 15 transactions
    train['rank'] = range(len(train))
    train = (
        train
            .assign(
            rn=train.groupby(['customer_id'])['rank']
                .rank(method='first', ascending=False))
            .query("rn <= 15")
            .drop(columns=['price', 'sales_channel_id'])
            .sort_values(['t_dat', 'customer_id'])
    )
    train['label'] = 1

    del train['rank']
    del train['rn']

    valid.sort_values(['t_dat', 'customer_id'], inplace=True)

    last_dates = (
        train
            .groupby('customer_id')['t_dat']
            .max()
            .to_dict()
    )

    negatives = prepare_candidates(train['customer_id'].unique(), 15)
    negatives['t_dat'] = negatives['customer_id'].map(last_dates)
    trues = train[['customer_id', 'article_id']]
    df_common = pd.merge(trues, negatives, on=['customer_id', 'article_id'], how='inner')
    negatives_new = negatives.append(df_common).drop_duplicates(keep=False)

    negatives = (
        negatives_new
            .merge(user_features, on=('customer_id'))
            .merge(item_features, on=('article_id'))
    )
    negatives['label'] = 0

    train = pd.concat([train, negatives])
    train.sort_values(['customer_id', 't_dat'], inplace=True)

    # train = train.drop_duplicates()

    valid_baskets = valid.groupby(['customer_id'])['article_id'].count().values
    train_baskets = train.groupby(['customer_id'])['article_id'].count().values

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="dart",
        max_depth=7,
        n_estimators=300,
        importance_type='gain',
        verbose=10
    )

    igonored_cols = ['t_dat', 'customer_id', 'article_id', 'label']
    ranker = ranker.fit(
        train.drop(columns=igonored_cols),
        train.pop('label'),
        group=train_baskets,
        #     eval_set=[valid.drop(columns = ['t_dat', 'customer_id', 'article_id', 'label']),valid['label']],
        #     eval_group= valid_baskets
    )

    cols = [col for col in train.columns if col not in igonored_cols]

    imps = ranker.feature_importances_
    df_imps = pd.DataFrame({"columns": train[cols].columns.tolist(), "feat_imp": imps})
    df_imps = df_imps.sort_values("feat_imp", ascending=False).reset_index(drop=True)
    print(df_imps.head(30))
    print(df_imps.to_csv(os.path.join(path,"feature_importance.csv")))

    sample_sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

    df = pd.read_csv(os.path.join(path, "submission_toppop_weight_decay.csv"))
    print("start")
    df['prediction'] = df.apply(lambda x: x.prediction.split(" "), axis=1)
    df = (
        df.explode('prediction')
            .rename(columns={'prediction': 'article_id'})
    )
    print("end")

    candidates = prepare_candidates(sample_sub.customer_id.unique(), 12)
    print(candidates.shape)
    print(df.shape)

    candidates = (
        candidates
            .merge(user_features, on=('customer_id'))
            .merge(item_features, on=('article_id'))
    )

    df = (
        df
            .merge(user_features, on=('customer_id'))
            .merge(item_features, on=('article_id'))
    )

    candidates = pd.concat([candidates, df], axis=0)
    candidates = candidates.drop_duplicates()

    preds = []
    batch_size = 1000000
    for bucket in tqdm(range(0, len(candidates), batch_size)):
        outputs = ranker.predict(
            candidates.iloc[bucket: bucket + batch_size]
                .drop(columns=['customer_id', 'article_id'])
        )
        preds.append(outputs)

    preds = np.concatenate(preds)
    candidates['preds'] = preds
    preds = candidates[['customer_id', 'article_id', 'preds']]
    preds.sort_values(['customer_id', 'preds'], ascending=False, inplace=True)
    preds = (
        preds
            .groupby('customer_id')[['article_id']]
            .aggregate(lambda x: x.tolist())
    )
    preds['article_id'] = preds['article_id'].apply(lambda x: ' '.join([str(v) for k, v in enumerate(x) if k < 12]))

    preds = sample_sub[['customer_id']].merge(
        preds
            .reset_index()
            .rename(columns={'article_id': 'prediction'}), how='left')
    preds['prediction'].fillna(' '.join([str(art) for art in dummy_list_2w]), inplace=True)

    preds.to_csv(os.path.join(path, 'submission_ranking.csv'), index=False)
