import os
import pandas as pd


def transactions_cleaning(df_transactions_raw):
    # df_transactions_raw = pd.read_csv(root_dir + '/transactions_train.csv')
    df_transactions = df_transactions_raw.copy()
    # transform the type of t_date from obj to datetime
    df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    df_transactions['month'] = df_transactions['t_dat'].dt.strftime('%m')
    df_transactions['year'] = df_transactions['t_dat'].dt.strftime('%Y')
    df_transactions = df_transactions.groupby(df_transactions.columns.tolist()).size().reset_index().rename(
        columns={0: 'article_purchase_count'})
    print("transactions cleaning done!")
    return df_transactions


