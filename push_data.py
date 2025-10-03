import os
import json
import pymongo
import certifi
import pandas as pd
from dotenv import load_dotenv

from src.Logging.logger_etl import logging
from src.Exception.exception import CustomException
from src.Constants import mongo_db_dc, data_ingestion, common_constants

load_dotenv("src/Secrets/mongo_db.env")
MONGO_DB_UN = os.getenv("MONGO_DB_UN")
MONGO_DB_PW = os.getenv("MONGO_DB_PW")
MONGO_DB_URL = f"mongodb+srv://{MONGO_DB_UN}:{MONGO_DB_PW}@cluster0.8y5aipc.mongodb.net/?retryWrites=true&w=majority&appName={mongo_db_dc.CLUSTER_NAME}"
print(MONGO_DB_URL)
ca = certifi.where()


class IPODataExtract:
    def __init__(self):
        try:
            pass
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

    def insert_data_mongodb(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    FILE_PATH = f"{data_ingestion.DATA_DIR}/{common_constants.DATA_FILE_NAME}"
    DATABASE = mongo_db_dc.DATABASE_NAME
    Collection = mongo_db_dc.COLLECTION_NAME
    logging.info("ETL | Pushing Started")
    networkobj = IPODataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    logging.info("ETL | Pushing Finished")
    print(no_of_records)
