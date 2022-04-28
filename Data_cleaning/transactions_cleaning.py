import pandas as pd
def transactions_cleaning(df_transactions_raw):
    # df_transactions_raw = pd.read_csv(root_dir + '/transactions_train.csv')
    df_transactions_raw['month'] = df_transactions_raw['t_dat'].dt.month
    # print("month extracted from date successfully.")
    df_transactions_raw['year'] = df_transactions_raw['t_dat'].dt.year
    # print("year extracted from date successfully.")
    df_transactions_raw = df_transactions_raw.groupby(df_transactions_raw.columns.tolist()).size().reset_index().rename(
        columns={0: 'article_purchase_count'})
    df_article_price = df_transactions_raw.groupby('article_id')['price'].agg(
        ['mean', 'max', 'min', 'median']).reset_index().rename(
        columns={'mean': 'mean_price', 'max': 'max_price', 'min': 'min_price', 'median': 'median_price'})
    df_transactions_raw = pd.merge(df_transactions_raw, df_article_price, how='left', on='article_id')
    df_transactions_raw['buy_with_no_discount'] = df_transactions_raw.apply(lambda x: 1 if x['price'] == x['max_price'] else 0,
                                                                    axis=1)
    del df_article_price
    print("transactions cleaning done!")
    return df_transactions_raw


