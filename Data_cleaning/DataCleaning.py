import os
import pandas as pd


dataset_dict = {"articles": "articles.csv", "customers": "customers.csv", "transactions": "transactions_train.csv"}

path = '../dataset'

df_articles = pd.read_csv(os.path.join(path, dataset_dict["articles"]))
df_customers = pd.read_csv(os.path.join(path, dataset_dict["customers"]))
df_transactions = pd.read_csv(os.path.join(path, dataset_dict["transactions"]))