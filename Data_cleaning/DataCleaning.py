import os
import pandas as pd

import Data_cleaning
from Data_cleaning.customers_cleaning import *
from Data_cleaning.articles_cleaning import *
from Data_cleaning.transactions_cleaning import *

if __name__ == '__main__':
    dataset_dict = {"articles": "articles.csv", "customers": "customers.csv", "transactions": "transactions_train.csv"}

    path = '../dataset'

    df_articles_raw = pd.read_csv(os.path.join(path, dataset_dict["articles"]))
    df_customers_raw = pd.read_csv(os.path.join(path, dataset_dict["customers"]))
    df_transactions_raw = pd.read_csv(os.path.join(path, dataset_dict["transactions"]))

    df_customers = initial_all_missing_values(df_customers_raw)
    df_articles = articles_func(df_articles_raw)
    df_transactions = transactions_cleaning(df_transactions_raw)
    #print(df_articles.head())

    df_articles.to_csv(os.path.join(path, "cleaned_"+dataset_dict["articles"]))
    df_customers.to_csv(os.path.join(path, "cleaned_" + dataset_dict["customers"]))
    df_transactions.to_csv(os.path.join(path, "cleaned_" + dataset_dict["transactions"]))
