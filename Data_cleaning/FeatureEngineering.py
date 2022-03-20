import pandas as pd
import numpy as np


# the function of calculating then number of month which don't have any transactions
def cal_inactive_months(x):
    if x['09'] > 0:
        return 0
    else:
        if x['08'] > 0:
            return 1
        else:
            if x['07'] > 0:
                return 2
            else:
                if x['06'] > 0:
                    return 3
                else:
                    return 4


def customers_feature_engineering(df_customers, df_transactions):
    # generate some new features from the transaction behaviours of 2020
    year = '2020'
    df = df_transactions[df_transactions['year'] == year]

    # calculate the number of transactions per month per customer
    df_month_avg_item_per_u = df.groupby(['customer_id', 'month'])['price'].count().unstack().reset_index()
    df_month_avg_item_per_u = pd.merge(df_month_avg_item_per_u, df_customers[['customer_id']], on='customer_id',
                                       how='outer')

    # new feature 1: number of months which don't have any transactions for each customer
    df_month_avg_item_per_u['num_missing_months' + '_' + year] = df_month_avg_item_per_u.isnull().sum(axis=1)
    df_month_avg_item_per_u = df_month_avg_item_per_u.fillna(0)
    # new feature 2: number of the latest continuous months don't have any transactions for each customer
    df_month_avg_item_per_u['lastest_continuouse_inactive_months' + '_' + year] = df_month_avg_item_per_u[
        df_month_avg_item_per_u.columns.difference(['customer_id', 'num_missing_months'])].apply(
        lambda x: cal_inactive_months(x), axis=1)

    # new feature 3: number of transactions in 2020 year
    df_avg_item_per_u = df.groupby(['customer_id'])['price'].count().reset_index()
    df_avg_item_per_u.columns = ['customer_id', 'num_transactions' + '_' + year]
    df_result = pd.merge(df_month_avg_item_per_u, df_avg_item_per_u, on='customer_id', how='outer')
    df_result = df_result.fillna(0)

    # new feature 4: average number of transactions each active month
    total_months = len(
        df_result.columns.difference(
            ['customer_id', 'num_missing_months' + '_' + year, 'lastest_continuouse_inactive_months' + '_' + year,
             'num_transactions' + '_' + year]))
    df_result['avg_transactions_in_active_month' + '_' + year] = df_result.apply(
        lambda x: x['num_transactions' + '_' + year] / (total_months - x['num_missing_months' + '_' + year]) if x[
                                                                                                                    'num_transactions' + '_' + year] > 0 else 0,
        axis=1)

    df_result = pd.merge(df_customers, df_result, on='customer_id', how='outer')

    print('New features already added into df_customers')

    return df_result


def articles_feature_engineering(df_articles, df_transactions):
    # generate some new features from the transaction behaviours of 2020
    year = '2020'
    df = df_transactions[df_transactions['year'] == year]
    df_month_price = df[['article_id', 'price', 'month', 'year']].drop_duplicates(
        ['article_id', 'price', 'month', 'year']).copy()

    # calculate mean price of each article monthly
    df_month_avg_price = df_month_price.groupby(['article_id', 'month'])['price'].mean().unstack().reset_index()
    print(1)
    # if the avg price is null of month 7,8,9, then it is marked as 'out of stock'
    df_out_of_stock = df_month_avg_price[df_month_avg_price['07'].isna() &
                                         df_month_avg_price['08'].isna() & df_month_avg_price['09'].isna()]
    df_out_of_stock = pd.DataFrame({'article_id': df_out_of_stock.article_id.values})
    df_out_of_stock['out_of_stock'] = 1

    # calculate mean price of each article yearly
    df_year_avg_price = df_month_price.groupby(['article_id'])['price'].mean().reset_index()
    df_on_sale = pd.merge(df_month_avg_price, df_year_avg_price, on='article_id')
    print(2)
    # if the current price(September 2020) is 10% lower than the yearly mean price, then it is marked as 'on sale'
    df_on_sale['on_sale'] = df_on_sale.apply(
        lambda x: 1 if x['09'] != -1 and abs(x['price'] - x['09']) / x['price'] > 0.1 else 0, axis=1)
    df_on_sale = df_on_sale[['article_id', 'on_sale']].copy()

    # Add the new features into the cleaned articles dataframe
    # new feature 1: out of stock
    df_result = pd.merge(df_articles[['article_id']], df_out_of_stock, on='article_id', how='outer')
    # new feature 2: on sale in September 2020
    df_result = pd.merge(df_result, df_on_sale, on='article_id', how='outer')
    df_result = df_result.fillna(0)
    df_result = pd.merge(df_articles, df_result, on='article_id', how='outer')

    print('New features already added into df_articles')

    return df_result
