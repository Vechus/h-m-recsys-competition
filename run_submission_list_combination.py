import multiprocessing
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from Utils.Logger import Logger
from multiprocessing import Manager


def func(csv_b):
    global csv_a
    for index, row in csv_b.iterrows():
        prediction_a = csv_a[csv_a['customer_id'] == row['customer_id']]['prediction'].values[0]
        prediction_b = row['prediction']
        prediction_list = " ".join(prediction_a.split(' ')[:2] + prediction_b.split(' ')[1:11])
        # print(prediction_list)

        csv_a.loc[csv_a['customer_id'] == row['customer_id'], 'prediction'] = prediction_list

        print(csv_a[csv_a['customer_id'] == row['customer_id']])


def parallelize_dataframe():
    csv_b = pd.read_csv(DATASET_PATH + '/P3alphaRP3betaItemKNN_CFCBF_HybridHybridRecommender-submission-final.csv')
    num_cores = multiprocessing.cpu_count() - 5
    num_partitions = num_cores
    df_split = np.array_split(csv_b[0:1000], num_partitions)
    pool = multiprocessing.Pool(num_cores)
    pool.map(func, df_split)

    pool.close()
    pool.join()


if __name__ == '__main__':

    # current date and time
    start = datetime.now()

    log_for_telegram_group = False
    logger = Logger('Hybrid - Start time:' + str(start))
    if log_for_telegram_group:
        logger.log('Started Hyper-parameter tuning. Hybrid recsys')
    print('Started Hyper-parameter tuning')
    try:
        load_dotenv()
        DATASET_PATH = os.getenv('DATASET_PATH')
        csv_a = pd.read_csv(DATASET_PATH + '/test.csv')
        parallelize_dataframe()
        csv_a.to_csv(DATASET_PATH + "/new.csv", index=False)
        print("Save file to " + DATASET_PATH + "/new.csv")
    except Exception as e:
        if log_for_telegram_group:
            logger.log('We got an exception! Check log and turn off the machine.')
            logger.log('Exception: \n{}'.format(str(e)))
        print('We got an exception! Check log and turn off the machine.')
        print('Exception: \n{}'.format(str(e)))
        print(traceback.format_exc())
    if log_for_telegram_group:
        end = datetime.now()
        logger.log('Hyper parameter search finished! Check results and turn off the machine. '
                   'End time:' + str(end) + '  Program duration:' + str(end - start))
    print('Hyper parameter search finished! Check results and turn off the machine.')
