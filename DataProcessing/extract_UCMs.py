UCM_list = [
    'club_member_status',
    'fashion_news_frequency',
    'age',
    'postal_code',
    'num_missing_months_2018',
    'num_missing_months_perc_2018',
    'num_sale_months_2018',
    'num_transactions_2018',
    'avg_transactions_in_active_month_2018',
    'num_missing_months_2019',
    'num_missing_months_perc_2019',
    'num_sale_months_2019',
    'num_transactions_2019',
    'avg_transactions_in_active_month_2019',
    'num_missing_months_2020',
    'num_missing_months_perc_2020',
    'num_sale_months_2020',
    'num_transactions_2020',
    'avg_transactions_in_active_month_2020',
    'latest_continuous_inactive_months_2020',
    'num_missing_months_total',
    'num_sale_months_total',
    'num_missing_months_perc_total',
    'num_transactions_total',
    'avg_transactions_in_active_month_total',
    'age_class_5',
    'age_class_10'
]


def gen_UCM_list(manager, customers):
    for column in UCM_list:
        print('Creating UCM for column {}'.format(column))

        ucm_df = customers[['customer_id', column]]
        ucm_df.rename(columns={column: "FeatureID", "customer_id": "UserID"}, inplace=True)
        ucm_df['UserID'] = ucm_df['UserID'].astype(str)
        ucm_df['FeatureID'] = ucm_df['FeatureID'].astype(str)
        ucm_df['Data'] = 1.0
        manager.add_UCM(ucm_df, 'UCM_{}'.format(column))
