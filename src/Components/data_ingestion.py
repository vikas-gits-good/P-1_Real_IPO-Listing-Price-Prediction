import os
import numpy as np
import pandas as pd
from pymongo import MongoClient

from sklearn.model_selection import train_test_split

from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Constants import common_constants
from src.Entity.config_entity import DataIngestionConfig, MongoDBConfig
from src.Entity.artifact_entity import DataIngestionArtifact
from src.Utils.main_utils import save_dataframe


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig(),
        db_config: MongoDBConfig = MongoDBConfig(),
    ):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.db_config = db_config
        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def get_dataframe(self) -> pd.DataFrame:
        try:
            db_database_name = self.db_config.database
            db_collection_name = self.db_config.collection_main
            self.mongo_client = MongoClient(self.db_config.mongo_db_url)
            collections = self.mongo_client[db_database_name][db_collection_name]
            df = pd.DataFrame(list(collections.find()))
            df.drop_duplicates(inplace=True, ignore_index=True)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def get_indices(self, df, X_split):
        try:
            return df[
                df.drop(columns=common_constants.TARGET_COLUMN)
                .apply(tuple, axis=1)
                .isin(
                    pd.DataFrame(
                        X_split, columns=df.columns.drop(common_constants.TARGET_COLUMN)
                    ).apply(tuple, axis=1)
                )
            ].index

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def initialise(self) -> DataIngestionArtifact:
        try:
            logging.info("Data Ingestion: Started")
            logging.info("Data Ingestion: Getting data from database")
            df_main = self.get_dataframe()

            logging.info("Data Ingestion: Performing train-valid-test split")
            df_train, df_valid = train_test_split(
                df_main,
                train_size=self.data_ingestion_config.train_size,
                random_state=self.data_ingestion_config.random_state,
                stratify=df_main[common_constants.TARGET_COLUMN],
            )
            df_valid, df_test = train_test_split(
                df_valid,
                test_size=self.data_ingestion_config.test_size,
                random_state=self.data_ingestion_config.random_state,
                stratify=df_valid[common_constants.TARGET_COLUMN],
            )

            logging.info("Data Ingestion: Saving ingested data to file")
            for data, path in [
                (df_main, self.data_ingestion_config.feature_store_dir),
                (df_train, self.data_ingestion_config.train_dir),
                (df_valid, self.data_ingestion_config.vald_dir),
                (df_test, self.data_ingestion_config.test_dir),
            ]:
                save_dataframe(data=data, path=path)

            logging.info("Data Ingestion: Exporting data ingestion artifact")
            di_artf = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_dir,
                vald_file_path=self.data_ingestion_config.vald_dir,
                test_file_path=self.data_ingestion_config.test_dir,
            )
            logging.info("Data Ingestion: Finished")
            return di_artf

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
