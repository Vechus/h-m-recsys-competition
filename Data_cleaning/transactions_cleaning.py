import os
import pandas as pd

def transactions_cleaning(df_transactions_raw):
    #df_transactions_raw = pd.read_csv(root_dir + '/transactions_train.csv')
    df_transactions = df_transactions_raw.copy()
    df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    df_transactions= df_transactions.groupby(df_transactions.columns.tolist()).size().reset_index().rename(columns={0:'article_purchase_count'})
    return df_transactions