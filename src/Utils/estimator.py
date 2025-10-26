import numpy as np
import pandas as pd
from logging import Logger
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.Logging.logger import log_trn
from src.Exception.exception import CustomException


class NetworkModel:
    def __init__(
        self,
        preprocessor: Pipeline = None,
        model: RandomForestClassifier = None,
        log: Logger = log_trn,
    ):
        try:
            self.ppln_prpc = preprocessor
            self.model = model
            self.log = log

        except Exception as e:
            self.log.info(f"Error: {e}")
            raise CustomException(e)

    def to_dataframe(self, y: np.typing.NDArray = None) -> pd.DataFrame:
        try:
            df_pred = pd.DataFrame()
            df_pred = pd.DataFrame(y, columns=["Predicted_IPO_listing_gain_category"])

            def convert_pred_catg(row):
                mapping = {
                    "Cat_1": (-100, -20),
                    "Cat_2": (-20, 0),
                    "Cat_3": (0, 10),
                    "Cat_4": (10, 20),
                    "Cat_5": (20, 40),
                    "Cat_6": (40, 100),
                    0: (-100, -20),
                    1: (-20, 0),
                    2: (0, 10),
                    3: (10, 20),
                    4: (20, 40),
                    5: (40, 100),
                }
                if row in mapping:
                    low, high = mapping[row]
                    return f"{low} to {high}"

            df_pred["Predicted_IPO_listing_gain_category"] = df_pred[
                "Predicted_IPO_listing_gain_category"
            ].apply(convert_pred_catg)

            return df_pred

        except Exception as e:
            self.log(f"Error: {e}")
            raise CustomException(e)
            return df_pred

    def predict(
        self, X: pd.DataFrame = None, y: pd.DataFrame = None
    ) -> np.typing.NDArray:
        try:
            # Use net_model.log = log_prd and then net_model.predict() during Prediction Pipeline
            self.log.info("Prediction: Transforming input data")
            x_test_trfm = self.ppln_prpc.transform(X=X)

            self.log.info("Prediction: Predicting outcome")
            y_pred = self.model.predict(X=x_test_trfm)

            self.log.info("Prediction: Transforming prediction to required format")
            df_pred = self.to_dataframe(y=y_pred)
            return df_pred

        except Exception as e:
            self.log.info(f"Error: {e}")
            raise CustomException(e)
