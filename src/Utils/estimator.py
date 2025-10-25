import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.Logging.logger import log_prd
from src.Exception.exception import CustomException


class NetworkModel:
    def __init__(
        self, preprocessor: Pipeline = None, model: RandomForestClassifier = None
    ):
        try:
            self.ppln_prpc = preprocessor
            self.model = model

        except Exception as e:
            log_prd.info(f"Error: {e}")
            raise CustomException(e)

    def predict(
        self, X: np.typing.NDArray = None, y: np.typing.NDArray = None
    ) -> np.typing.NDArray:
        try:
            log_prd.info("User Prediction: Transforming input data")
            x_test_trfm = self.ppln_prpc.transform(X)

            log_prd.info("User Prediction: Predicting outcome")
            y_pred = self.model.predict(x_test_trfm)

            log_prd.info("User Prediction: Exporting prediction")
            return y_pred

        except Exception as e:
            log_prd.info(f"Error: {e}")
            raise CustomException(e)
