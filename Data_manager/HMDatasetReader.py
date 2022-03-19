"""
Created on 19/03/2022
@author: Riccardo Pazzi
"""

import zipfile, shutil
import numpy as np
import pandas as pd
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL
from Data_manager.DatasetMapperManager import DatasetMapperManager

DATASET_SOURCE = "hm-temporal/"
DATASET_ZIP_NAME = "dataset_URM.zip"


class HMDatasetReader(DataReader):
    DATASET_SUBFOLDER = DATASET_SOURCE
    AVAILABLE_URM = ["URM_all", "URM_timestamp"]
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []
    DATASET_SPLIT_ROOT_FOLDER = "./processed/"

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original
        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + DATASET_ZIP_NAME)

        except (FileNotFoundError, zipfile.BadZipFile):
            print("No dataset in path: {}".format(zipFile_path + DATASET_ZIP_NAME))
            return

        # All URMs and ICMs must be extracted here
        # ICM_genre_path = dataFile.extract("ml-1m/movies.dat", path=zipFile_path + "decompressed/")
        # UCM_path = dataFile.extract("ml-1m/users.dat", path=zipFile_path + "decompressed/")
        """This part should be changed to use Luca scripts to correctly create the dataset folder"""
        URM_train_path = dataFile.extract(DATASET_ZIP_NAME + "/URM_train.npz", path=zipFile_path + "decompressed/")
        URM_test_path = dataFile.extract(DATASET_ZIP_NAME + "/URM_test.npz", path=zipFile_path + "decompressed/")

        self._print("Loading URMs...")
        URM_train = np.load(URM_train_path)
        URM_test = np.load(URM_test_path)

        self._print("Converting into dataframes...")
        train_df = pd.DataFrame(data=URM_train,
                                columns=["UserID", "ItemID", "Data"])
        test_df = pd.DataFrame(data=URM_test, columns=["UserID", "ItemID", "Data"])

        # For each user a list of features

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(train_df, "URM_train")
        dataset_manager.add_URM(test_df, "URM_test")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset
