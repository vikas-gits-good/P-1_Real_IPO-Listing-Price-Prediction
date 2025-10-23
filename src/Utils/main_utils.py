import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Utils.ml_utils import get_model_scores
from src.Utils.estimator import NetworkModel


def log_exception(error: Exception = None) -> logging:
    _, _, exc_tb = sys.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    log_msg = f"Error: File - [{filename}], line - [{lineno}], error - [{str(error)}]"
    logging.info(log_msg)


def save_dataframe(data: pd.DataFrame = None, path: str = None) -> None:
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(path, header=True, index=False)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def read_dataframe(path: str = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def write_yaml_file(
    file_path: str = None, content: object = None, replace: bool = True
) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def read_yaml_file(file_path: str = None) -> dict:
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def save_numpy_array(file_path: str = None, array: np.typing.NDArray = None) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def read_numpy_array(file_path: str = None) -> np.typing.NDArray:
    try:
        with open(file_path, "rb") as file:
            return np.load(file, allow_pickle=True)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def save_transformation_object(file_path: str = None, object: Pipeline = None) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(object, file)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def read_transformation_object(file_path: str = None) -> Pipeline:
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def save_model_object(file_path: str = None, object: NetworkModel = None) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(object, file)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def read_model_object(file_path: str = None) -> NetworkModel:
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


def evaluate_models(
    x_train: np.typing.NDArray = None,
    y_train: np.typing.NDArray = None,
    x_vald: np.typing.NDArray = None,
    y_vald: np.typing.NDArray = None,
    models: dict = None,
    sort_by: Literal["f1_score", "precision_score", "recall_score"] = "f1_score",
) -> dict:
    try:
        report = {}
        for i in range(len(list(models.keys()))):
            model_name = list(models.keys())[i]
            model_object = models[model_name]["Model"]
            model_params = models[model_name]["Parameters"]

            logging.info(f"Model Training: Training '{model_name}' model")
            gs = GridSearchCV(
                estimator=model_object,
                param_grid=model_params,
                cv=3,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X=x_train, y=y_train)

            logging.info(f"Model Training: Scoring best fit '{model_name}' model")
            y_pred_train, y_pred_vald = [gs.predict(x) for x in [x_train, x_vald]]
            scores_train, scores_vald = [
                get_model_scores(y_true, y_pred)
                for y_true, y_pred in [
                    (y_train, y_pred_train),
                    (y_vald, y_pred_vald),
                ]
            ]
            logging.info(
                f"Model Training: train_{sort_by}={getattr(scores_train, sort_by):.4f} & vald_{sort_by}={getattr(scores_vald, sort_by):.4f}"
            )
            report[model_name] = {
                "Model_object": gs,
                "Model_score": [scores_train, scores_vald],
            }
        # reorder the dictionary in the descending order of validation set score
        logging.info("Model Training: Exporting evaluated best fit models")
        report_sorted = dict(
            sorted(
                report.items(),
                key=lambda item: getattr(item[1]["Model_score"][1], sort_by),
                reverse=True,
            )
        )
        return report_sorted

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)
