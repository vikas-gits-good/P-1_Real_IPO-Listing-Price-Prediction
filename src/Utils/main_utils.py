import os
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.Logging.logger_train import logging
from src.Exception.exception import CustomException


def save_dataframe(data: pd.DataFrame = None, path: str = None) -> None:
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(path, header=True, index=False)

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


def save_numpy_array(file_path: str = None, array: np.array = None) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)

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
