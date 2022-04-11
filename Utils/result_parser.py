import glob
import os
import re

import pandas as pd
from dotenv import load_dotenv


def handle_best_config(bestLine):
    configure = re.search('{(.*)}', bestLine).group(1)
    config = ''
    for i in configure.split(', '):
        i1 = i.split(':')[0].replace('\'', '')
        i1 += '='
        i2 = i.split(':')[1].replace(' ', '')
        i = i1 + i2
        i += ','
        config += i
    return config[:-1]


def handle_MAP(bestLine):
    return re.search('MAP: (\\d+.\\d+)', bestLine).group(1)


def get_best_line(filename):
    with open(filename) as fh:
        for line in fh:
            if 'New best config found' in line:
                best_line = line

    return best_line


def handle_ICM_name(filename):
    # ICMName = re.search('ICM_.*_SearchBayesianSkopt', filename).group(0).rsplit('_', 2)[0]
    ICMName = re.search('ICM_(.*)_SearchBayesianSkopt', filename).group(1).rsplit('_', 1)[0]
    return '\'' + ICMName + '\','


if __name__ == '__main__':

    load_dotenv()

    RESULT_PATH = os.getenv('RESULT_PATH')
    result_name = '/ItemKNNCBF_CFCBF_URM_Train_2019-06-22_2019-09-23_Val_2019-09-23_2019-09-30'

    columns = ['ICM name', 'MAP', 'Best config']
    dataframe = pd.DataFrame(columns=columns)

    for file_name in list(glob.glob(RESULT_PATH + result_name + '/*.txt')):

        if 'CFCBF' in file_name.split('/')[-1]:
            continue

        best_line = get_best_line(file_name)
        config = handle_best_config(best_line)
        MAP = handle_MAP(best_line)
        ICM_name = handle_ICM_name(file_name)
        df_new = {'ICM name': ICM_name,
                  'MAP': MAP,
                  'Best config': config}

        df_new = pd.DataFrame(data=df_new, index=[file_name.split('/')[-1].replace('.txt', '')])

        if df_new['ICM name'].values not in dataframe['ICM name'].values:
            dataframe = dataframe.append(df_new)
        elif dataframe.loc[dataframe['ICM name'] == df_new.iloc[0]['ICM name']]['MAP'].values[0] < df_new['MAP'].values:
            dataframe.drop(dataframe.loc[dataframe['ICM name'] == df_new.iloc[0]['ICM name']].index, inplace=True)
            dataframe = dataframe.append(df_new)

    dataframe.sort_values(by=['MAP'], inplace=True, ascending=False)
    dataframe.to_excel(RESULT_PATH + result_name + '/output_CBF.xlsx')
    print("Completed!!!")
