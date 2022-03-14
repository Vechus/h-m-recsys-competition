import numpy as np
import pandas as pd


def map_customers():
    customers = pd.read_csv('../dataset/customers.csv')
    print(customers)

    users_mapping = {'user_id': [], 'map_id': []}
    for index, row in customers.iterrows():
        users_mapping['user_id'].append(row['customer_id'])
        users_mapping['map_id'].append(index)

    # save csv
    pd.DataFrame.from_dict(users_mapping).to_csv('../matrices/users_mapping.csv', index=False)


def map_articles():
    articles = pd.read_csv('../dataset/articles.csv')
    print(articles)

    articles_mapping = {'article_id': [], 'map_id': []}
    for index, row in articles.iterrows():
        articles_mapping['article_id'].append(row['article_id'])
        articles_mapping['map_id'].append(index)

    pd.DataFrame.from_dict(articles_mapping).to_csv('../matrices/articles_mapping.csv', index=False)

def generate_URM():
    transactions = pd.read_csv('../dataset/transactions_train.csv', nrows=100)
    print(transactions)
    #df.loc[df['column_name'] == some_value]
    u_map = pd.read_csv('../matrices/users_mapping.csv')
    a_map = pd.read_csv('../matrices/articles_mapping.csv')
    print('[!] read csv...')

    # row=user, col=item, data=1.0
    urm_dict = {'row': [], 'col': [], 'data': []}

    for index, row in u_map.iterrows():
        # transactions for a user
        t_u = transactions.loc[transactions['customer_id'] == row['user_id']]
        # for each t_u, now find all article ids, and map to map_id of article
        articles = t_u['article_id']
        user_interactions = []
        for article in articles:
            # list of (mapped ids) articles interacted by that user
            user_interactions = user_interactions + a_map['map_id'][a_map['article_id'] == article].index.tolist()
        
        if user_interactions != []:
            print(row, user_interactions)
            for i in range(len(user_interactions)):
                urm_dict['row'].append(row['map_id'])
                urm_dict['col'].append(user_interactions[i])
                urm_dict['data'].append(1.0)
    
    pd.DataFrame.from_dict(urm_dict).to_csv('../matrices/URM_all.csv', index=False)


    #print(transactions.loc[transactions['customer_id'] == '05943a58bd172641b80919a9bdf14012df940800bc74d0707abdd38efc1dd4f1'])

if __name__ == '__main__':
    generate_URM()