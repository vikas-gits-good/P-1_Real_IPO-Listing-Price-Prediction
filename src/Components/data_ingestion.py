import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Constants import mongo_db_dc, common_constants
from src.Entity.config_entity import DataIngestionConfig
from src.Entity.artifact_entity import DataIngestionArtifact

load_dotenv("src/Secrets/mongo_db.env")
MONGO_DB_UN = os.getenv("MONGO_DB_UN")
MONGO_DB_PW = os.getenv("MONGO_DB_PW")
MONGO_DB_DC = mongo_db_dc()
COMMON_CONSTANTS = common_constants()
MONGO_DB_URL = f"mongodb+srv://{MONGO_DB_UN}:{MONGO_DB_PW}@cluster0.8y5aipc.mongodb.net/?retryWrites=true&w=majority&appName={MONGO_DB_DC.CLUSTER_NAME}"


class DataIngestion:
    def __init__(
        self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def get_dataframe(self):
        try:
            db_database_name = self.data_ingestion_config.database_name
            db_collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = MongoClient(MONGO_DB_URL)
            collections = self.mongo_client[db_database_name][db_collection_name]
            df = pd.DataFrame(list(collections.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def save_data_to_file(self, data: pd.DataFrame = None, path: str = None):
        try:
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)
            data.to_csv(path, header=True, index=False)

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def initialise(self):
        try:
            logging.info("Data Ingestion: Started")
            logging.info("Data Ingestion: Getting data from database")
            df_main = self.get_dataframe()

            logging.info("Data Ingestion: Performing train-test split")
            df_train, df_test = train_test_split(
                df_main,
                test_size=self.data_ingestion_config.test_size,
                random_state=self.data_ingestion_config.random_state,
            )

            logging.info("Data Ingestion: Saving ingested data to file")
            for data, path in [
                (df_main, self.data_ingestion_config.feature_store_dir),
                (df_train, self.data_ingestion_config.train_dir),
                (df_test, self.data_ingestion_config.test_dir),
            ]:
                self.save_data_to_file(data=data, path=path)

            logging.info("Data Ingestion: Exporting data ingestion artifact")
            di_artf = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_dir,
                test_file_path=self.data_ingestion_config.test_dir,
            )
            logging.info("Data Ingestion: Finished")
            return di_artf

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
