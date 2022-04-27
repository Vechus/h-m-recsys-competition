import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter


# the function of calculating then number of month which don't have any transactions
def cal_inactive_months(x):
    prefix = 'num_transactions_2020_'
    if x[prefix + '9'] > 0:
        return 0
    else:
        if x[prefix + '8'] > 0:
            return 1
        else:
            if x[prefix + '7'] > 0:
                return 2
            else:
                if x[prefix + '6'] > 0:
                    return 3
                else:
                    return 4


def age_bin(age, age_bin):
    if age == -1:
        age_interval = "unknown"
    elif age < 20:
        age_interval = "<20"
    else:
        age_interval = "[{},{})".format(math.floor(age / age_bin) * age_bin, (math.floor(age / age_bin) + 1) * age_bin)
    return age_interval


# classify the gender for articles
def set_gender_flg(x):
    x['is_for_male_or_female'] = 0  # 0:unknown,1:female,2:male
    x['is_for_mama'] = 0
    if x['cleaned_index_group_name'] in ['ladieswear', 'divid']:
        x['is_for_male_or_female'] = 1
    if x['cleaned_index_group_name'] == 'menswear':
        x['is_for_male_or_female'] = 2
    if x['cleaned_index_group_name'] in ['babychildren', 'sport']:
        if 'boy' in x['cleaned_department_name'].lower() or 'men' in x['cleaned_department_name'].lower():
            x['is_for_male_or_female'] = 2
        if 'girl' in x['cleaned_department_name'].lower() or 'ladi' in x['cleaned_department_name'].lower():
            x['is_for_male_or_female'] = 1
    if x['cleaned_section_name'] == 'mama':
        x['is_for_mama'] = 1
    return x


def customers_feature_engineering(df_customers, df_transactions):
    print("New features generation of customers start.")
    # map original postal code into index
    postcode_original_ID_to_index_dict = {}
    for pc_id in df_customers["postal_code"].unique():
        postcode_original_ID_to_index_dict[pc_id] = len(postcode_original_ID_to_index_dict)
    df_customers["postcode"] = [postcode_original_ID_to_index_dict[pc_original] for pc_original in
                                df_customers["postal_code"].values]
    df_customers.drop(columns='postal_code', axis=1, inplace=True)
    df_customers = df_customers.rename(columns={'postcode': "postal_code"})

    df_customer_group = df_transactions.groupby(['customer_id'])
    list2 = []
    for each in df_customer_group:
        df = each[1]
        list1 = df['article_id'].values.tolist()
        dict1 = Counter(list1)
        repetition = len([i for i in dict1.values() if i > 1]) / len(dict1)
        list2.append(repetition)
    dff_customer = pd.DataFrame()
    dff_customer['customer_id'] = list(df_customer_group.groups.keys())
    dff_customer['rebuy_ratio_customer'] = list2
    df_customers = pd.merge(df_customers, dff_customer, how='left', on='customer_id')

    del dff_customer

    # generate some new features from the transaction behaviours for each year
    years = df_transactions.year.unique()

    total_months_in_training_data = 0

    for index, year in enumerate(years):
        year_str = str(year)
        df = df_transactions[df_transactions['year'] == year]

        # calculate the number of transactions per month per customer
        df_month_avg_item_per_u = df.groupby(['customer_id', 'month'])['price'].count().unstack().add_prefix(
            'num_transactions_' + year_str + "_").reset_index()
        df_month_avg_item_per_u = pd.merge(df_month_avg_item_per_u, df_customers[['customer_id']], on='customer_id',
                                           how='outer')

        # new feature 1: number of months which don't have any transactions for each customer in the certain year
        df_month_avg_item_per_u['num_missing_months' + '_' + year_str] = df_month_avg_item_per_u.isnull().sum(
            axis=1)

        df_month_avg_item_per_u['num_sale_months' + '_' + year_str] = df_month_avg_item_per_u.notnull().sum(
            axis=1) - 2
        df_month_avg_item_per_u = df_month_avg_item_per_u.fillna(0)

        # get the total month of the certain year
        total_months = df_month_avg_item_per_u.count(axis=1) - 3
        total_months_in_training_data = total_months_in_training_data + total_months
        # new feature 2: the percent of missing months in the certain year
        df_month_avg_item_per_u['num_missing_months_perc' + '_' + year_str] = df_month_avg_item_per_u[
                                                                                  'num_missing_months' + '_' + year_str] / total_months
        # new feature 3: number of the latest continuous months don't have any transactions for each customer in 2020
        if year == 2020:
            df_month_avg_item_per_u['latest_continuous_inactive_months' + '_' + year_str] = df_month_avg_item_per_u[
                df_month_avg_item_per_u.columns.difference(['customer_id', 'num_missing_months'])].apply(
                lambda x: cal_inactive_months(x), axis=1)

        # new feature 4: number of transactions of the certain year
        df_avg_item_per_u = df.groupby(['customer_id'])['price'].count().reset_index()
        df_avg_item_per_u.columns = ['customer_id', 'num_transactions' + '_' + year_str]
        df_month_avg_item_per_u = pd.merge(df_month_avg_item_per_u, df_avg_item_per_u, on='customer_id',
                                           how='outer')

        # new feature 5: average number of transactions each active month
        df_month_avg_item_per_u[
            'avg_transactions_in_active_month' + '_' + year_str] = df_month_avg_item_per_u.apply(
            lambda x: x['num_transactions' + '_' + year_str] / x['num_sale_months' + '_' + year_str] if x[
                                                                                                            'num_transactions' + '_' + year_str] > 0 else None,
            axis=1)

        df_result = df_month_avg_item_per_u.fillna(0)

        final_columns = ['customer_id', 'num_missing_months_' + year_str, 'num_missing_months_perc_' + year_str,
                         'num_sale_months_' + year_str, 'num_transactions_' + year_str,
                         'avg_transactions_in_active_month_' + year_str]
        if index == 0:
            df_result_accumulated = df_result[final_columns]
        else:
            if year == 2020:
                final_columns.append('latest_continuous_inactive_months' + '_' + year_str)
            df_result = df_result[final_columns]
            df_result_accumulated = pd.merge(df_result_accumulated, df_result, on='customer_id', how='outer')

        print("New feathers of " + year_str + " generation finished.")

    df_result_accumulated['num_missing_months_total'] = df_result_accumulated[
        ['num_missing_months_' + str(i) for i in years]].sum(axis=1)
    print("New feather 2:num_missing_months_total generation finished.")

    df_result_accumulated['num_sale_months_total'] = df_result_accumulated[
        ['num_sale_months_' + str(i) for i in years]].sum(axis=1)

    df_result_accumulated['num_missing_months_perc_total'] = df_result_accumulated[
                                                                 'num_missing_months_total'] / total_months_in_training_data
    print("New feather 3:num_missing_months_percent_total generation finished.")

    df_result_accumulated['num_transactions_total'] = df_result_accumulated[
        list(df_result_accumulated.filter(regex='num_transactions'))].sum(axis=1)

    print("New feather 4:num_transactions_total generation finished.")
    df_result_accumulated['avg_transactions_in_active_month_total'] = df_result_accumulated.apply(
        lambda x: x['num_transactions_total'] / x['num_sale_months_total'] if x['num_transactions_total'] > 0 else 0,
        axis=1)

    print("New feather 5:avg_transactions_in_active_month_total generation finished.")

    df_result = pd.merge(df_customers, df_result_accumulated, on='customer_id', how='outer')

    df_result["age_class_5"] = df_result.apply(lambda x: age_bin(x.age, 5), axis=1)
    df_result["age_class_10"] = df_result.apply(lambda x: age_bin(x.age, 10), axis=1)

    print('All new features added into df_customers!')

    return df_result
    # df_result


def articles_feature_engineering(df_articles, df_transactions):
    print("New features generation of articles start.")
    df_articles_copy = df_articles.copy()

    df_article_group = df_transactions.groupby(['article_id'])
    list2 = []
    for each in df_article_group:
        df = each[1]
        list1 = df['customer_id'].values.tolist()
        dict1 = Counter(list1)
        repetition = len([i for i in dict1.values() if i > 1]) / len(dict1)
        list2.append(repetition)
    dff_article = pd.DataFrame()
    dff_article['article_id'] = list(df_article_group.groups.keys())
    dff_article['rebuy_ratio_article'] = list2

    df_articles = pd.merge(df_articles, dff_article, how='left', on='article_id')
    del dff_article

    df_articles['idxgrp_idx_prdtyp'] = df_articles['cleaned_index_group_name'] + '_' + df_articles[
        'cleaned_index_name'] + '_' + df_articles['cleaned_product_type_name']
    # df_trans = df_trans.sample(frac=0.1) # downsampling due to memory limits
    df_transactions_new = pd.merge(df_transactions, df_articles[['article_id', 'idxgrp_idx_prdtyp']], on='article_id',
                                   how='left')

    # generate some new features from the transaction behaviours of 2020
    year = 2020
    df = df_transactions[df_transactions['year'] == year][df_transactions.columns.difference(['idxgrp_idx_prdtyp'])]
    df_month_price = df[['article_id', 'price', 'month', 'year']].drop_duplicates(
        ['article_id', 'price', 'month', 'year']).copy()

    # calculate mean price of each article monthly
    df_month_avg_price = df_month_price.groupby(['article_id', 'month'])['price'].mean().unstack().reset_index()
    # if the avg price is null of month 7,8,9, then it is marked as 'out of stock'
    df_out_of_stock = df_month_avg_price[df_month_avg_price[7].isna() &
                                         df_month_avg_price[8].isna() & df_month_avg_price[9].isna()]
    df_out_of_stock = pd.DataFrame({'article_id': df_out_of_stock.article_id.values})
    df_out_of_stock['out_of_stock'] = 1

    # calculate mean price of each article yearly
    df_year_avg_price = df_month_price.groupby(['article_id'])['price'].mean().reset_index()
    df_on_discount = pd.merge(df_month_avg_price, df_year_avg_price, on='article_id')
    # if the current price(September 2020) is 10% lower than the yearly average price,
    # then it is marked as 'on discounting'
    df_on_discount['on_discount'] = df_on_discount.apply(
        lambda x: 1 if x[9] != -1 and abs(x['price'] - x[9]) / x['price'] > 0.1 else 0, axis=1)
    df_on_discount = df_on_discount[['article_id', 'on_discount']].copy()

    # Add the new features into the cleaned articles dataframe
    # new feature 1: out of stock
    df_result = pd.merge(df_articles[['article_id']], df_out_of_stock, on='article_id', how='outer')
    print("Feature 1:out of stock generation finished.")
    # new feature 2: on discounting(September 2020)
    df_result = pd.merge(df_result, df_on_discount, on='article_id', how='outer')
    print("Feature 2:on discount generation finished.")

    # new feature 3:sales months of each article
    df_articles_period_months = (df_transactions.groupby(['article_id'])['t_dat'].max() -
                                 df_transactions.groupby(['article_id'])['t_dat'].min()).map(
        lambda x: math.ceil(x.days / 30)).reset_index().rename(columns={'t_dat': 'sale_periods_months'})

    df_result = pd.merge(df_result, df_articles_period_months, on='article_id', how='outer')
    df_result = df_result.fillna(0)
    df_result['sale_periods_months'] = df_result['sale_periods_months'].astype(int)
    print("Feature 3:sale periods(months) generation finished.")

    # new feature 4:the transaction peak month of each article
    YM = [201809, 201810]
    while YM[0] < 202010:
        start, end = "-".join(map(str, [YM[0] // 100, YM[0] % 100, 1])), "-".join(
            map(str, [YM[1] // 100, YM[1] % 100, 1]))
        monthly_sales = df_transactions.query(f"'{start}' <= t_dat < '{end}'").groupby('article_id')[
            'article_purchase_count'].sum().to_dict()

        df_articles_copy[YM[0]] = 0
        for i in df_articles_copy.index:
            key = df_articles_copy.iloc[i, :]['article_id']
            if key in monthly_sales.keys():
                df_articles_copy.at[i, YM[0]] = monthly_sales[df_articles_copy.at[i, "article_id"]]
            else:
                df_articles_copy.at[i, YM[0]] = 0

        print("\r Done :", YM[0], end="")
        YM[0] = YM[1]
        YM[1] = (YM[1] + 100 - 11) if YM[1] % 100 == 12 else (YM[1] + 1)

    df_articles_part = df_articles_copy.iloc[:, 25:]
    df_articles_part['transaction_peak_year_month'] = df_articles_part.apply(lambda x:
                                                                             str(df_articles_part.columns[:][
                                                                                     np.argmax(x[:])]) if np.argmax(
                                                                                 x[:]) > 0 else None, axis=1)
    df_result = pd.merge(df_result, df_articles_part[['transaction_peak_year_month']], left_index=True, right_index=True
                         , how='outer')

    print("\nFeature 4:transaction peak(year-month) generation finished.")

    df_result = df_result.fillna(0)
    df_result = pd.merge(df_articles, df_result, on='article_id', how='outer')

    del df_articles

    # new feature 5: seasonality for each article

    df_transactions_new['order_price'] = df_transactions_new['price'] * df_transactions_new['article_purchase_count']

    dfgrp1 = df_transactions_new.groupby(['idxgrp_idx_prdtyp'])[['order_price']].sum().reset_index()
    dfgrp2 = df_transactions_new.groupby(['idxgrp_idx_prdtyp', 'year', 'month'])[['order_price']].sum().reset_index()
    dfgrp2 = pd.merge(dfgrp2, dfgrp1, on='idxgrp_idx_prdtyp', how='left')

    dfgrp2['monthsales/ttl-sales'] = dfgrp2['order_price_x'] / dfgrp2['order_price_y'] * 100
    dfgrp2['ym_date'] = dfgrp2['year'].astype(str) + '-' + dfgrp2['month'].astype(str) + '-1'
    dfgrp2['ym_date'] = pd.to_datetime(dfgrp2['ym_date'])
    dfgrp2 = pd.pivot_table(dfgrp2, index='ym_date', columns='idxgrp_idx_prdtyp',
                            values='monthsales/ttl-sales').reset_index().fillna(0)

    feat_cols = [col for col in dfgrp2.columns if col != 'ym_date']

    df_pca = StandardScaler().fit_transform(dfgrp2[feat_cols])
    model_pca = PCA(n_components=5)
    model_pca.fit(df_pca)
    feature = model_pca.transform(df_pca)

    df_eigen = model_pca.components_.T
    df_eigen = pd.DataFrame(df_eigen,
                            index=None,
                            columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
    df_eigen['idxgrp_idx_prdtyp'] = feat_cols
    df_eigen = pd.merge(
        df_eigen,
        dfgrp2.corr()[['ladieswear_ladieswear_jacket']].reset_index().rename(
            columns={'ladieswear_ladieswear_jacket': 'autumn_sales_indicator'}),
        on='idxgrp_idx_prdtyp',
        how='left'
    )

    # print(model_pca.explained_variance_ratio_)

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=4, covariance_type='full')
    gmm.fit(df_eigen[['PC1', 'PC2', 'PC3']])

    # '0' means the article is clustered into autumn, '3' means the article is clustered into summer
    df_eigen['product_seasonal_type'] = gmm.predict(df_eigen[['PC1', 'PC2', 'PC3']])
    df_eigen['prob_cluster1'] = gmm.predict_proba(df_eigen[['PC1', 'PC2', 'PC3']])[:, 0]
    df_eigen['prob_cluster2'] = gmm.predict_proba(df_eigen[['PC1', 'PC2', 'PC3']])[:, 1]
    df_eigen['prob_cluster3'] = gmm.predict_proba(df_eigen[['PC1', 'PC2', 'PC3']])[:, 2]
    df_eigen['prob_cluster4'] = gmm.predict_proba(df_eigen[['PC1', 'PC2', 'PC3']])[:, 3]

    df_result = pd.merge(
        df_result,
        df_eigen[['idxgrp_idx_prdtyp', 'autumn_sales_indicator', 'product_seasonal_type']],
        on='idxgrp_idx_prdtyp',
        how='left'
    )

    df_result = df_result.fillna(0)

    df_result['product_seasonal_type'] = df_result['product_seasonal_type'].astype(int)

    # df_result.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    print("Feature 5:product_seasonal_type generation finished.")

    df_result = df_result.apply(set_gender_flg, axis=1)

    print("Feature 6:is_for_male/female/mama generation finished.")

    print('All new features added into df_articles!')

    return df_result
