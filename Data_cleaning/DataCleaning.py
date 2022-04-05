import pyarrow

from Data_cleaning.customers_cleaning import *
from Data_cleaning.articles_cleaning import *
from Data_cleaning.transactions_cleaning import *
from Data_cleaning.FeatureEngineering import *

import os
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    dataset_dict = {"articles": "articles.csv", "customers": "customers.csv", "transactions": "transactions_train.csv",
                    "sample_submission": "sample_submission.csv"}

    path = os.getenv('DATASET_PATH')

    df_articles = pd.read_csv(os.path.join(path, dataset_dict["articles"]), dtype={'article_id': str})
    df_customers = pd.read_csv(os.path.join(path, dataset_dict["customers"]))
    df_transactions = pd.read_csv(os.path.join(path, dataset_dict["transactions"]), dtype={'article_id': str},
                                  parse_dates=['t_dat'])

    # Clean the dataset by assigning different parameters
    df_customers = initial_all_missing_values(df_customers)
    df_articles = articles_cleaning(df_articles)
    df_transactions = transactions_cleaning(df_transactions)
    # df_transactions.to_csv(os.path.join(path, "processed_" + dataset_dict["transactions"]))
    df_transactions.to_parquet(
        os.path.join(path, "processed_" + dataset_dict["transactions"].replace('csv', 'parquet')))

    # Add new features into df_customers
    df_customers_final = customers_feature_engineering(df_customers, df_transactions)
    df_customers_final['club_member_status'] = df_customers_final['club_member_status'].astype(str)
    df_customers_final['fashion_news_frequency'] = df_customers_final['fashion_news_frequency'].astype(str)
    print(df_customers_final)
    df_customers_final.to_parquet(
        os.path.join(path, "processed_" + dataset_dict["customers"].replace('csv', 'parquet')))

    # Add new features into df_articles
    df_articles_final = articles_feature_engineering(df_articles, df_transactions)
    df_articles_final['transaction_peak_year_month'] = df_articles_final['transaction_peak_year_month'].astype(str)
    print(df_articles_final)
    # df_articles_final.to_csv(os.path.join(path, "processed_" + dataset_dict["articles"]))
    df_articles_final.to_parquet(os.path.join(path, "processed_" + dataset_dict["articles"].replace('csv', 'parquet')))
