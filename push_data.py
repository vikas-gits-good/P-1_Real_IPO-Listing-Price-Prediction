import json
from pymongo import MongoClient
import pandas as pd

from src.Logging.logger import log_etl
from src.Exception.exception import CustomException
from src.Constants import data_ingestion, common_constants
from src.Entity.config_entity import MongoDBConfig


class IPODataPusher:
    def __init__(self, db_config: MongoDBConfig = MongoDBConfig()):
        try:
            self.db_config = db_config
        except Exception as e:
            raise CustomException(e)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e)

    def insert_data_mongodb(self, records):
        try:
            self.database = self.db_config.database
            self.collection = self.db_config.collection_main  # <- Check before using
            self.records = records

            self.mongo_client = MongoClient(self.db_config.mongo_db_url)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    FILE_PATH = f"{data_ingestion.DATA_DIR}/{common_constants.DATA_FILE_NAME}"
    log_etl.info("ETL | Pushing Started")
    networkobj = IPODataPusher()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records)
    log_etl.info("ETL | Pushing Finished")
    print(no_of_records)
