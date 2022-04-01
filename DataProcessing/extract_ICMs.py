import numpy as np
import pandas as pd

from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.Dataset import Dataset


# ICM_list = [
#     # 'article_id',
#     # 'product_code',
#     'product_type_no',
#     'graphical_appearance_no',
#     'colour_group_code',
#     'perceived_colour_value_id',
#     'perceived_colour_master_id',
#     'department_no',
#     'index_code',
#     'index_group_no',
#     'section_no',
#     'garment_group_no',
#     # 'cleaned_prod_name',
#     'cleaned_product_type_name',
#     'cleaned_product_group_name',
#     'cleaned_graphical_appearance_name',
#     'cleaned_colour_group_name',
#     'cleaned_perceived_colour_value_name',
#     'cleaned_perceived_colour_master_name',
#     'cleaned_department_name',
#     'cleaned_index_name',
#     'cleaned_index_group_name',
#     'cleaned_section_name',
#     'cleaned_garment_group_name',
#     # 'cleaned_detail_desc',
#     # 'out_of_stock',
#     # 'out_of_stock',
#     'sale_periods_months',
#     'transaction_peak_year_month'
#       'autumn_sales_indicator'
# ]


def gen_ICM_list(manager, articles):
    ICM_list = articles.columns.difference(
        ['Unnamed: 0', 'article_id', 'product_code', 'cleaned_prod_name', 'cleaned_detail_desc', 'out_of_stock',
         'out_of_stock'])

    for column in ICM_list:
        print('Creating ICM for column {}'.format(column))

        icm_df = articles[['article_id', column]]
        icm_df.rename(columns={column: "FeatureID", "article_id": "ItemID"}, inplace=True)
        icm_df['ItemID'] = icm_df['ItemID'].astype(str)
        icm_df['FeatureID'] = icm_df['FeatureID'].astype(str)
        icm_df['Data'] = 1.0
        manager.add_ICM(icm_df, 'ICM_{}'.format(column))


def get_ICM_all(manager, articles):
    ICM_list = articles.columns.difference(
        ['Unnamed: 0', 'article_id', 'product_code', 'cleaned_prod_name', 'cleaned_detail_desc', 'out_of_stock',
         'out_of_stock'])

    print('Creating ICM_all')
    df_total = pd.DataFrame()

    for column in ICM_list:
        print('Creating ICM for column {}'.format(column))

        icm_df = articles[['article_id', column]]
        icm_df[column] = icm_df.apply(lambda x: column+'_'+str(x[column]), axis=1)
        icm_df.rename(columns={column: "FeatureID", "article_id": "ItemID"}, inplace=True)
        icm_df['ItemID'] = icm_df['ItemID'].astype(str)
        icm_df['FeatureID'] = icm_df['FeatureID'].astype(str)
        icm_df['Data'] = 1.0
        df_total = pd.concat([df_total, icm_df], axis=0)

    manager.add_ICM(df_total, 'ICM_all')
