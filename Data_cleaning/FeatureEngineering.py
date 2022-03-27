import pandas as pd
import numpy as np
import math


# the function of calculating then number of month which don't have any transactions
def cal_inactive_months(x):
    prefix = 'num_transactions_2020_'
    if x[prefix + '09'] > 0:
        return 0
    else:
        if x[prefix + '08'] > 0:
            return 1
        else:
            if x[prefix + '07'] > 0:
                return 2
            else:
                if x[prefix + '06'] > 0:
                    return 3
                else:
                    return 4


def customers_feature_engineering(df_customers, df_transactions):
    # generate some new features from the transaction behaviours for each year
    global df_result_accumulated
    years = df_transactions.year.unique()

    total_months_in_training_data = 0
    total_active_months_in_training_data = 0

    for index, year in enumerate(years):
        df = df_transactions[df_transactions['year'] == year]

        # calculate the number of transactions per month per customer
        df_month_avg_item_per_u = df.groupby(['customer_id', 'month'])['price'].count().unstack().add_prefix(
            'num_transactions_' + year + "_").reset_index()
        df_month_avg_item_per_u = pd.merge(df_month_avg_item_per_u, df_customers[['customer_id']], on='customer_id',
                                           how='outer')

        # new feature 1: number of months which don't have any transactions for each customer in the certain year
        df_month_avg_item_per_u['num_missing_months' + '_' + year] = df_month_avg_item_per_u.isnull().sum(axis=1)
        df_month_avg_item_per_u = df_month_avg_item_per_u.fillna(0)

        # get the total month of the certain year
        total_months = df_month_avg_item_per_u.count(axis=1) - 2
        total_months_in_training_data = total_months_in_training_data + total_months
        # new feature 2: the percent of missing months in the certain year
        df_month_avg_item_per_u['num_missing_months_perc' + '_' + year] = df_month_avg_item_per_u[
                                                                              'num_missing_months' + '_' + year] / total_months
        # new feature 3: number of the latest continuous months don't have any transactions for each customer in 2020
        if year == '2020':
            df_month_avg_item_per_u['latest_continuous_inactive_months' + '_' + year] = df_month_avg_item_per_u[
                df_month_avg_item_per_u.columns.difference(['customer_id', 'num_missing_months'])].apply(
                lambda x: cal_inactive_months(x), axis=1)

        # new feature 4: number of transactions of the certain year
        df_avg_item_per_u = df.groupby(['customer_id'])['price'].count().reset_index()
        df_avg_item_per_u.columns = ['customer_id', 'num_transactions' + '_' + year]
        df_result = pd.merge(df_month_avg_item_per_u, df_avg_item_per_u, on='customer_id', how='outer')
        df_result = df_result.fillna(0)

        # total_active_months = len(
        #     df_result.columns.difference(
        #         ['customer_id', 'num_missing_months' + '_' + year,
        #          'num_missing_months_perc' + '_' + year,
        #          'latest_continuous_inactive_months' + '_' + year,
        #          'num_transactions' + '_' + year]))
        # total_active_months_in_training_data = total_active_months_in_training_data + total_months

        # new feature 5: average number of transactions each active month
        df_result['avg_transactions_in_active_month' + '_' + year] = df_result.apply(
            lambda x: x['num_transactions' + '_' + year] / (
                    total_months - x['num_missing_months' + '_' + year]) if x[
                                                                                'num_transactions' + '_' + year] > 0 else 0,
            axis=1)

        final_columns = ['customer_id', 'num_missing_months_' + year, 'num_missing_months_perc_' + year,
                         'num_transactions_' + year, 'avg_transactions_in_active_month_' + year]
        if index == 0:
            df_result_accumulated = df_result[final_columns]
        else:
            if year == '2020':
                final_columns.append('latest_continuous_inactive_months' + '_' + year)
            df_result = df_result[final_columns]
            df_result_accumulated = pd.merge(df_result_accumulated, df_result, on='customer_id', how='outer')
        print('The new features of ' + year + ' added into df_customers')

    df_result_accumulated['num_missing_months_total'] = df_result_accumulated[
        ['num_missing_months_' + i for i in years]].sum(axis=1)
    df_result_accumulated['num_missing_months_prec_total'] = df_result_accumulated[
                                                                 'num_missing_months_total'] / total_months_in_training_data

    df_result_accumulated['num_transactions_total'] = df_result_accumulated[
        list(df_result_accumulated.filter(regex='num_transactions'))].sum(axis=1)
    df_result_accumulated['avg_transactions_in_active_month_total'] = df_result_accumulated.apply(
        lambda x: x['num_transactions_total'] / (
                total_months - x['num_missing_months_total']) if x['num_transactions_total'] > 0 else 0,
        axis=1)

    df_result = pd.merge(df_customers, df_result_accumulated, on='customer_id', how='outer')

    print('All new features already added into df_customers!')

    return df_result



def get_transaction_peak(df_transactions_peak, article_id):
    article_id = str(article_id)
    df1 = df_transactions_peak[df_transactions_peak['article_id'] == article_id]
    if (df1.empty != True):
        df2 = df_transactions_peak[df_transactions_peak['article_id'] == article_id][
            df_transactions_peak.columns.difference(['article_id', 'year'])]
        df2_stacked = df2.stack()
        peak_year = df1[df1.index == df2_stacked.sort_values(ascending=False).reset_index().rename(
            columns={'level_0': 'year_index'}).iloc[0].year_index].year.values[0]
        peak_month = \
            df2_stacked.sort_values(ascending=False).reset_index().rename(columns={'level_0': 'year_index'}).iloc[
                0].month
        peak = peak_year + '-' + peak_month
    else:
        peak = None
    return peak


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
    df_on_discounting = pd.merge(df_month_avg_price, df_year_avg_price, on='article_id')
    print(2)
    # if the current price(September 2020) is 10% lower than the yearly mean price, then it is marked as 'on sale'
    df_on_discounting['on_discounting'] = df_on_discounting.apply(
        lambda x: 1 if x['09'] != -1 and abs(x['price'] - x['09']) / x['price'] > 0.1 else 0, axis=1)
    df_on_discounting = df_on_discounting[['article_id', 'on_discounting']].copy()

    # Add the new features into the cleaned articles dataframe
    # new feature 1: out of stock
    df_result = pd.merge(df_articles[['article_id']], df_out_of_stock, on='article_id', how='outer')
    # new feature 2: on sale in September 2020
    df_result = pd.merge(df_result, df_on_discounting, on='article_id', how='outer')
    df_result = pd.merge(df_articles, df_result, on='article_id', how='outer')

    # new feature 3:sales months of each article
    df_articles_period_months = (df_transactions.groupby(['article_id'])['t_dat'].max() -
                                 df_transactions.groupby(['article_id'])['t_dat'].min()).map(
        lambda x: math.ceil(x.days / 30)).reset_index().rename(columns={'t_dat': 'sales_period_month'})

    df_result = pd.merge(df_result, df_articles_period_months, on='article_id', how='outer')

    print(3)

    # new feature 4:the transaction peak month of each article
    df_transactions_peak = df_transactions.groupby(['article_id', 'year', 'month'])[
        'article_purchase_count'].sum().unstack().reset_index()

    df_result['transaction_peak_year_month'] = df_result.apply(
        lambda x: get_transaction_peak(df_transactions_peak, x.article_id), axis=1)

    print(4)

    df_result = df_result.fillna(0)

    print('New features already added into df_articles!')

    return df_result
