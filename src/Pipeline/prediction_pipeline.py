import json
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient, UpdateOne

from src.Logging.logger import log_prd
from src.Exception.exception import CustomException, LogException

from src.Utils.estimator import NetworkModel
from src.Utils.main_utils import read_model_object
from src.Entity.config_entity import MongoDBConfig
from src.Constants import common_constants


class MakeIPOPrediction:
    def __init__(self, db_config: MongoDBConfig = MongoDBConfig()):
        self.db_config = db_config
        self.today = datetime.today()  # datetime(2025, 9, 5)  #

    def df_to_json_convertor(self, data: pd.DataFrame):
        try:
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records

        except Exception as e:
            log_prd.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)

    def update_prediction_database(self, data: pd.DataFrame = None) -> None:
        try:
            log_prd.info("Prediction: Communicating with database")
            records = self.df_to_json_convertor(data)
            database = self.db_config.database
            collection = self.db_config.collection_pred_arcv  # <- Check before using

            mongo_client = MongoClient(self.db_config.mongo_db_url)
            upld_data = mongo_client[database][collection]

            log_prd.info("Prediction: Preparing data upload/update operations")
            operations = []
            for record in records:
                filter_query = {"IPO_company_name": record.get("IPO_company_name")}
                operations.append(
                    UpdateOne(filter_query, {"$set": record}, upsert=True)
                )

            if operations:
                log_prd.info(
                    f"Prediction: Uploading data to archive database: '{collection}'"
                )
                upld_data.bulk_write(operations)

        except Exception as e:
            log_prd.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)

    def get_current_data(self) -> pd.DataFrame:
        try:
            database = self.db_config.database
            collection = self.db_config.collection_main  # <- Check before using

            mongo_client = MongoClient(self.db_config.mongo_db_url)
            dwld_data = mongo_client[database][collection]

            df = pd.DataFrame(list(dwld_data.find()))
            df.drop_duplicates(inplace=True, ignore_index=True)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)

            # skip prediction for columns whose subscription data is unavailable
            # might extend this later to other columns like gmp as well
            df = df.loc[df["IPO_day2_qib"] != "error", :]

            df["IPO_open_date"] = pd.to_datetime(df["IPO_open_date"])
            month_filt = (df["IPO_open_date"].dt.year == self.today.year) & (
                df["IPO_open_date"].dt.month == self.today.month
            )

            df_filt = df.loc[month_filt, :]
            df_filt.reset_index(drop=True, inplace=True)
            return df_filt

        except Exception as e:
            log_prd.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)

    def predict(self, path: str = None):
        try:
            log_prd.info(f"Prediction: Reading '{path.split('/')[-1]}' object")
            trfm_obj: NetworkModel = read_model_object(file_path=path)

            log_prd.info(
                f"Prediction: Getting {self.today.strftime('%Y %B')}'s IPO company data to predict"
            )
            df_x_pred = self.get_current_data()
            X = df_x_pred.drop(columns=[common_constants.TARGET_COLUMN])

            log_prd.info("Prediction: Sending data to prediction pipeline")
            trfm_obj.log = log_prd  # change the log from log_trn to log_prd
            df_y_pred = trfm_obj.predict(X=X)

            log_prd.info("Prediction: Updating database with latest prediction")
            df_pred = pd.concat([df_x_pred, df_y_pred], axis=1, ignore_index=False)
            self.update_prediction_database(df_pred)

            log_prd.info("Prediction: Exporting prediction")
            return df_y_pred

        except Exception as e:
            log_prd.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)


if __name__ == "__main__":
    try:
        MakeIPOPrediction().predict(
            path="Artifacts/2025_10_25_22_22_33/model_trainer/trained_model/2025-10-25_22-22-33_model.pkl"
        )

    except Exception as e:
        log_prd.info(f"Error: {e}")
        LogException(e)
        raise CustomException(e)
