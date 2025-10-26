import os
import pandas as pd
from glob import glob
from datetime import datetime

from src.Logging.logger import log_prd
from src.Exception.exception import CustomException, LogException

from src.Utils.estimator import NetworkModel
from src.Utils.main_utils import (
    read_model_object,
    get_df_from_MongoDB,
    put_df_to_MongoDB,
)
from src.Constants import common_constants


class MakeIPOPrediction:
    def __init__(self):
        pass

    def predict(self, path: str = None):
        try:
            log_prd.info(f"Prediction: Reading '{path.split('/')[-1]}' object")
            trfm_obj: NetworkModel = read_model_object(file_path=path)

            log_prd.info("Prediction: Getting data from database")
            df_x_pred = get_df_from_MongoDB(
                collection="IPOPredMain",
                pipeline="predict",
                log=log_prd,
                prefix="Prediction",
            )
            X = df_x_pred.drop(columns=[common_constants.TARGET_COLUMN])

            log_prd.info("Prediction: Sending data to prediction pipeline")
            trfm_obj.log = log_prd  # change the log from log_trn to log_prd
            df_y_pred = trfm_obj.predict(X=X)

            log_prd.info("Prediction: Updating database with latest prediction")
            df_pred = pd.concat([df_x_pred, df_y_pred], axis=1, ignore_index=False)
            put_df_to_MongoDB(
                data=df_pred, collection="IPOPredArcv", log=log_prd, prefix="Prediction"
            )
            log_prd.info("Prediction: Exporting prediction")
            return df_pred

        except Exception as e:
            log_prd.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)


if __name__ == "__main__":
    try:
        model_paths = glob("Artifacts/*/model_trainer/trained_model/*_model.pkl")
        model_paths = sorted(
            model_paths, key=lambda x: os.path.basename(x)[:19], reverse=True
        )
        df_y_pred = MakeIPOPrediction().predict(path=model_paths[0])

    except Exception as e:
        log_prd.info(f"Error: {e}")
        LogException(e)
        raise CustomException(e)
