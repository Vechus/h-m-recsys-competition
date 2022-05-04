import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = os.getenv('DATASET_PATH')

csv_a = pd.read_csv(DATASET_PATH + '/test.csv')

csv_b = pd.read_csv(DATASET_PATH + '/P3alphaRP3betaItemKNN_CFCBF_HybridHybridRecommender-submission-final.csv')

for index, row in csv_b.iterrows():
    prediction_a = csv_a[csv_a['customer_id'] == row['customer_id']]['prediction'].values[0]
    prediction_b = row['prediction']
    prediction_list = " ".join(prediction_a.split(' ')[:2]+prediction_b.split(' ')[1:11])
    print(prediction_list)

    csv_a.loc[csv_a['customer_id'] == row['customer_id'], 'prediction'] = prediction_list

csv_a.to_csv(DATASET_PATH + "/new.csv")
