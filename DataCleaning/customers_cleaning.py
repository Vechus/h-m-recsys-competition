import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from tqdm import tqdm

dataset_dict = {"articles": "articles.csv", "customers": "customers.csv", "transactions": "transactions_train.csv"}

path = '../SourcesData'

df_articles = pd.read_csv(os.path.join(path, dataset_dict["articles"]))
df_customers = pd.read_csv(os.path.join(path, dataset_dict["customers"]))
df_transactions = pd.read_csv(os.path.join(path, dataset_dict["transactions"]))


def clean_FN(df_customers_to_clean, strategy='Delete_Feature'):

    if strategy == 'Delete_Feature':
        # Deleted the FN Active Columns
        df_customers_to_clean.drop(columns=['FN'], inplace=True)
    elif strategy == 'Impute_Zero':
        df_customers_to_clean['FN'] = df_customers_to_clean['FN'].fillna(0)

    return df_customers_to_clean


def clean_Active(df_customers_to_clean, strategy='Delete_Feature'):

    if strategy == 'Delete_Feature':
        # Deleted the FN Active Columns
        df_customers_to_clean.drop(columns=['Active'],  inplace=True)
    elif strategy == 'Impute_Zero':
        df_customers_to_clean['Active'] = df_customers_to_clean['Active'].fillna(0)
    return df_customers_to_clean


def clean_club_member_status(df_customers_to_clean, strategy='Impute_Mode_Active'):
    # Impute the club_member_status with Mode - Active
    if strategy == 'Impute_Mode_Active':
        # Impute club_member_status by majority vote value which is "Active".
        df_customers_to_clean['club_member_status'].fillna('ACTIVE', inplace=True)

    return df_customers_to_clean


def clean_fashion_news_frequency(df_customers_to_clean, strategy='Impute_Mode_NONE'):
    mask = df_customers_to_clean[df_customers_to_clean['fashion_news_frequency'] == 'None'].index.values
    df_customers_to_clean.loc[mask, 'fashion_news_frequency'] = 'NONE'

    # Impute the fashion_news_frequency with Mode - None
    if strategy == 'Impute_Mode_NONE':
        # Impute fashion_news_frequency by majority vote value which is "NONE".
        df_customers_to_clean['fashion_news_frequency'].fillna('NONE', inplace=True)

    return df_customers_to_clean


def clean_Age(df_customers_to_clean, strategy='Mean_Age_of_club_member_status'):
    if strategy == 'Mean_Age_of_club_member_status':
        # Impute the Age with the mean age of club_member_status
        df_customers_to_clean_ = df_customers_to_clean.copy()
        map_means = df_customers_to_clean_.groupby('club_member_status')['age'].mean().to_dict()

        mask = df_customers_to_clean['age'].isnull()
        age_values = df_customers_to_clean.loc[mask, 'club_member_status'].map(map_means).values
        is_nan_age = df_customers_to_clean.loc[mask, 'age'].index.values

        for i in tqdm(range(len(is_nan_age))):
            df_customers_to_clean.loc[is_nan_age[i], 'age'] = age_values[i]

    elif strategy == 'Mean_Age_of_fashion_news_frequency':

        # Impute the Age with the mean age of fashion_news_frequency
        map_means = df_customers_to_clean.groupby('fashion_news_frequency')['age'].mean().to_dict()

        mask = df_customers_to_clean['age'].isnull()
        age_values = df_customers_to_clean.loc[mask, 'fashion_news_frequency'].map(map_means).values
        is_nan_age = df_customers_to_clean.loc[mask, 'age'].index.values

        for i in tqdm(range(len(is_nan_age))):
            df_customers_to_clean.loc[is_nan_age[i], 'age'] = age_values[i]

    return df_customers_to_clean
