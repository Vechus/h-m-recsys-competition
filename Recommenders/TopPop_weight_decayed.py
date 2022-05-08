import numpy as np
import pandas as pd
import os
import glob
# import reco
from tqdm import tqdm
import datetime
import gc
import random
from collections import Counter


class TopPop_weight_decayed:
    # recommend last week most popular items as alternatives to current week

    def get_alternate_most_popular(self, df_data, factor, return_orig=False):
        path = 'dataset/'
        dataset_dict = {"articles": "articles.csv", "customers": "customers.csv",
                        "transactions": "transactions_train.csv",
                        "sample_submission": "sample_submission.csv"}

        articles_df = pd.read_csv(os.path.join(path, dataset_dict["articles"]))  # , dtype={'article_id': str})

        next_best_match = []

        df = df_data.copy()
        df['article_count'] = df.groupby('article_id')['customer_id'].transform('count')
        df['article_min_price'] = df.groupby('article_id')['price'].transform('min')
        count_df = df[['article_id', 'article_count', 'article_min_price']].drop_duplicates().reset_index(drop=True)

        del df

        for article in tqdm(count_df.article_id.tolist()):
            prodname = articles_df[articles_df.article_id == int(article)]['prod_name'].iloc[0]
            other_article_list = articles_df[articles_df.prod_name == prodname]['article_id'].tolist()
            other_article_list.remove(int(article))
            k = len(other_article_list)
            if k == 1:
                next_best_match.append(other_article_list[0])
            if k > 1:
                if len(count_df[np.in1d(count_df['article_id'], other_article_list)]) != 0:
                    next_best_match.append(
                        count_df[np.in1d(count_df['article_id'], other_article_list)].sort_values('article_count',
                                                                                                  ascending=False)[
                            'article_id'].iloc[0])
                else:
                    next_best_match.append(np.nan)
            if k == 0:
                next_best_match.append(np.nan)

        count_df['next_best_article'] = next_best_match
        count_df['next_best_article'] = count_df['next_best_article'].fillna(0).astype(int)
        count_df['next_best_article'] = np.where(count_df['next_best_article'] == 0, count_df['article_id'],
                                                 str(0) + count_df['next_best_article'].astype(str))

        right_df = count_df[['next_best_article']].copy().rename(columns={'next_best_article': 'article_id'})

        next_best_count = []
        next_best_price = []
        for article in tqdm(right_df['article_id']):
            if len(count_df[count_df.article_id == article]['article_count']) > 0:
                next_best_count.append(count_df[count_df.article_id == article]['article_count'].iloc[0])
                next_best_price.append(count_df[count_df.article_id == article]['article_min_price'].iloc[0])
            else:
                next_best_count.append(0)
                next_best_price.append(0)

        count_df['count_next_best'] = next_best_count
        count_df['next_best_min_price'] = next_best_price

        more_popular_alternatives = count_df[(count_df.article_min_price >= count_df.next_best_min_price) &
                                             (
                                                     count_df.count_next_best > factor * count_df.article_count)].copy().reset_index(
            drop=True)
        more_popular_alt_list = more_popular_alternatives.article_id.unique().tolist()

        if return_orig:
            return more_popular_alt_list, more_popular_alternatives, count_df
        else:
            return more_popular_alt_list, more_popular_alternatives

    def recommend(self, df_customers, file):
        path = 'dataset/'
        dataset_dict = {"articles": "articles.csv", "customers": "customers.csv",
                        "transactions": "transactions_train.csv",
                        "sample_submission": "sample_submission.csv"}
        data = pd.read_csv(os.path.join(path, dataset_dict["transactions"]), dtype={'article_id': str},
                           parse_dates=['t_dat'])

        print("All Transactions Date Range: {} to {}".format(data['t_dat'].min(), data['t_dat'].max()))

        data["t_dat"] = pd.to_datetime(data["t_dat"])
        train1 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 9, 8)) & (data['t_dat'] < datetime.datetime(2020, 9, 16))]
        train2 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 9, 1)) & (data['t_dat'] < datetime.datetime(2020, 9, 8))]
        train3 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 8, 23)) & (data['t_dat'] < datetime.datetime(2020, 9, 1))]
        train4 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 8, 15)) & (data['t_dat'] < datetime.datetime(2020, 8, 23))]
        train5 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 8, 7)) & (data['t_dat'] < datetime.datetime(2020, 8, 15))]

        val = data.loc[data["t_dat"] >= datetime.datetime(2020, 9, 16)]

        # articles_df = pd.read_csv(os.path.join(path, dataset_dict["articles"]))  # , dtype={'article_id': str})
        # customers_df = pd.read_csv(os.path.join(path, dataset_dict["customers"]))

        alt_list_1v, alt_df_1v = self.get_alternate_most_popular(train2, 2)
        alt_list_2v, alt_df_2v = self.get_alternate_most_popular(train3, 2)
        alt_list_3v, alt_df_3v = self.get_alternate_most_popular(train4, 2)
        alt_list_4v, alt_df_4v = self.get_alternate_most_popular(train5, 2)

        train1 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 9, 16)) & (data['t_dat'] < datetime.datetime(2020, 9, 23))]
        train2 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 9, 8)) & (data['t_dat'] < datetime.datetime(2020, 9, 16))]
        train3 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 8, 31)) & (data['t_dat'] < datetime.datetime(2020, 9, 8))]
        train4 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 8, 23)) & (data['t_dat'] < datetime.datetime(2020, 8, 31))]
        train5 = data.loc[
            (data["t_dat"] >= datetime.datetime(2020, 8, 15)) & (data['t_dat'] < datetime.datetime(2020, 8, 23))]

        alt_list_1, alt_df_1 = self.get_alternate_most_popular(train2, 2, return_orig=False)
        alt_list_2, alt_df_2 = alt_list_1v, alt_df_1v
        alt_list_3, alt_df_3 = alt_list_2v, alt_df_2v
        alt_list_4, alt_df_4 = alt_list_3v, alt_df_3v

        positive_items_per_user1 = train1.groupby(['customer_id'])['article_id'].apply(list)
        positive_items_per_user2 = train2.groupby(['customer_id'])['article_id'].apply(list)
        positive_items_per_user3 = train3.groupby(['customer_id'])['article_id'].apply(list)
        positive_items_per_user4 = train4.groupby(['customer_id'])['article_id'].apply(list)

        train = pd.concat([train1, train2], axis=0)
        train['pop_factor'] = train['t_dat'].apply(lambda x: 1 / (datetime.datetime(2020, 9, 23) - x).days)
        popular_items_group = train.groupby(['article_id'])['pop_factor'].sum()

        _, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

        user_group = pd.concat([train1, train2, train3, train4], axis=0).groupby(['customer_id'])['article_id'].apply(
            list)

        outputs = []
        cnt = 0

        recommended_items = []

        for user in tqdm(df_customers):
            if user in positive_items_per_user1.keys():
                most_common_items_of_user = {k: v for k, v in Counter(positive_items_per_user1[user]).most_common()}
                l = list(most_common_items_of_user.keys())
                al = []
                for j in range(0, len(l)):
                    if l[j] in alt_list_1:
                        al.append(alt_df_1[alt_df_1.article_id == l[j]]['next_best_article'].iloc[0])
                l = l + al
                recommended_items += l[:12]

            if user in positive_items_per_user2.keys():
                most_common_items_of_user = {k: v for k, v in Counter(positive_items_per_user2[user]).most_common()}
                l = list(most_common_items_of_user.keys())
                al = []
                for j in range(0, len(l)):
                    if l[j] in alt_list_2:
                        al.append(alt_df_2[alt_df_2.article_id == l[j]]['next_best_article'].iloc[0])
                l = l + al
                recommended_items += l[:12]

            if user in positive_items_per_user3.keys():
                most_common_items_of_user = {k: v for k, v in Counter(positive_items_per_user3[user]).most_common()}
                l = list(most_common_items_of_user.keys())
                al = []
                for j in range(0, len(l)):
                    if l[j] in alt_list_3:
                        al.append(alt_df_3[alt_df_3.article_id == l[j]]['next_best_article'].iloc[0])
                l = l + al
                recommended_items += l[:12]

            if user in positive_items_per_user4.keys():
                most_common_items_of_user = {k: v for k, v in Counter(positive_items_per_user4[user]).most_common()}
                l = list(most_common_items_of_user.keys())
                al = []
                for j in range(0, len(l)):
                    if l[j] in alt_list_4:
                        al.append(alt_df_4[alt_df_4.article_id == l[j]]['next_best_article'].iloc[0])
                l = l + al
                recommended_items += l[:12]

            recommended_items += list(popular_items[:12 - len(recommended_items)])
            recommended_items_str = " ".join([str(x) for x in recommended_items])

            file.write(f"{user}, {recommended_items_str}\n")
            # print("%s:%s" % (user, recommended_items))
