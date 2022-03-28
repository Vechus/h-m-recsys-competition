def transactions_cleaning(df_transactions_raw):
    # df_transactions_raw = pd.read_csv(root_dir + '/transactions_train.csv')
    df_transactions_raw['month'] = df_transactions_raw['t_dat'].dt.month
    # print("month extracted from date successfully.")
    df_transactions_raw['year'] = df_transactions_raw['t_dat'].dt.year
    # print("year extracted from date successfully.")
    df_transactions_raw = df_transactions_raw.groupby(df_transactions_raw.columns.tolist()).size().reset_index().rename(
        columns={0: 'article_purchase_count'})
    print("transactions cleaning done!")
    return df_transactions_raw


