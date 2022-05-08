import os
import traceback
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from Utils.Logger import Logger


def run_list_combination():
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')

    csv_a = pd.read_csv(DATASET_PATH + '/test.csv')

    csv_b = pd.read_csv(DATASET_PATH + '/P3alphaRP3betaItemKNN_CFCBF_HybridHybridRecommender-submission-final.csv')

    print(csv_a.size)
    print(csv_b.size)
    csv_a = csv_a.join(csv_a['prediction'].str.split(' ', expand=True).add_prefix('top'))
    csv_b = csv_b.join(csv_b['prediction'].str.split(' ', expand=True).add_prefix('top'))

    csv_a = csv_a.merge(csv_b, on=['customer_id'], how='left')
    csv_a['Top_6_x'] = csv_a['top0_x'].map(str) + ' ' + csv_a['top1_x'].map(str) + ' ' + \
                       csv_a['top2_x'].map(str) + ' ' + csv_a['top3_x'].map(str) + ' ' + \
                       csv_a['top4_x'].map(str) + ' ' + csv_a['top5_x'].map(str)
    csv_a['Top_7_12'] = csv_a[csv_a.columns[8:14]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    csv_a['Top_6_y'] = csv_a[csv_a.columns[16:22]].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    # csv_a = csv_a[['customer_id', 'Top_2_x', 'Top_3_12', 'Top_10_y']]
    csv_a.loc[csv_a['Top_6_y'].str.contains('nan'), 'Top_6_y'] = csv_a['Top_7_12']
    csv_a['prediction'] = csv_a['Top_6_x'].map(str) + ' ' + csv_a['Top_6_y'].map(str)
    csv_a = csv_a[['customer_id', 'prediction']]
    csv_a.to_csv(DATASET_PATH + "/new.csv", index=False)


if __name__ == '__main__':

    # current date and time
    start = datetime.now()

    log_for_telegram_group = True
    logger = Logger('Hybrid - Start time:' + str(start))
    if log_for_telegram_group:
        logger.log('Started Hyper-parameter tuning. Hybrid recsys')
    print('Started Hyper-parameter tuning')
    try:
        run_list_combination()
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
