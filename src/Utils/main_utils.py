import os
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from glob import glob
from typing import Literal
from datetime import datetime
from pymongo import MongoClient, UpdateOne
from dateutil.relativedelta import relativedelta

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from logging import Logger
from src.Logging.logger import log_etl, log_trn, log_prd
from src.Exception.exception import CustomException, LogException
from src.Utils.ml_utils import get_model_scores
from src.Utils.estimator import NetworkModel
from src.Entity.config_entity import MongoDBConfig


def save_dataframe(
    data: pd.DataFrame = None, path: str = None, log_name: Logger = log_trn
) -> None:
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(path, header=True, index=False)

    except Exception as e:
        log_name.info(f"Error: {e}")
        raise CustomException(e)


def read_dataframe(path: str = None, log_name: Logger = log_trn) -> pd.DataFrame:
    try:
        return pd.read_csv(path)

    except Exception as e:
        log_name.info(f"Error: {e}")
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
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def read_yaml_file(file_path: str = None) -> dict:
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def save_numpy_array(file_path: str = None, array: np.typing.NDArray = None) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def read_numpy_array(file_path: str = None) -> np.typing.NDArray:
    try:
        with open(file_path, "rb") as file:
            return np.load(file, allow_pickle=True)

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def save_transformation_object(file_path: str = None, object: Pipeline = None) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(object, file)

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def read_transformation_object(file_path: str = None) -> Pipeline:
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def save_model_object(file_path: str = None, object: NetworkModel = None) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(object, file)

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def read_model_object(file_path: str = None) -> NetworkModel:
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def get_df_from_MongoDB(
    collection: Literal[
        "IPOPredMain", "IPOPredOrig", "IPOPredTest", "IPOPredArcv"
    ] = "IPOPredMain",
    pipeline: Literal["etl", "train", "predict", "archive", "latest"] = "train",
    db_config: MongoDBConfig = MongoDBConfig(),
    log: Logger = log_trn,
    prefix: Literal[
        "Data Ingestion", "Prediction", "Webpage", "Extraction"
    ] = "Data Ingestion",
) -> pd.DataFrame:
    try:
        database_name = db_config.database
        log.info(f"{prefix}: Communicating with MongoDB: {database_name}/{collection}")
        mongo_client = MongoClient(db_config.mongo_db_url)
        collections = mongo_client[database_name][collection]
        # today = datetime(2025, 9, 5)
        today = pd.to_datetime(datetime.today())

        log.info(f"{prefix}: Preprocessing dataframe")
        df = pd.DataFrame(list(collections.find()))
        df.drop_duplicates(inplace=True, ignore_index=True)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        df.replace({"na": np.nan}, inplace=True)

        if pipeline == "etl":
            pass

        elif pipeline == "train":
            log.info(f"{prefix}: Dropping unlisted company data")
            df = df.dropna(subset=["IPO_listing_price"], ignore_index=True)
            df = df.loc[df["IPO_day3_qib"] != "error", :]

            last_month = today - relativedelta(months=1)
            end_date = last_month.replace(day=1)

            log.info(f"{prefix}: Selecting data till {end_date.strftime('%Y %B')}.")
            df["IPO_open_date"] = pd.to_datetime(
                df["IPO_open_date"], errors="raise", format="%Y-%m-%d"
            )
            month_filt = df["IPO_open_date"] < end_date
            df = df.loc[month_filt, :]
            df["IPO_open_date"] = df["IPO_open_date"].dt.strftime("%Y-%m-%d")

        elif pipeline == "predict":
            log.info(f"{prefix}: Selecting {today.strftime('%Y %B')}'s company data")
            # skip prediction for columns whose subscription data is unavailable
            # might extend this later to other columns like gmp as well
            df = df.loc[df["IPO_day2_qib"] != "error", :]
            df["IPO_open_date"] = pd.to_datetime(df["IPO_open_date"], errors="coerce")
            month_filt = (df["IPO_open_date"].dt.year == today.year) & (
                df["IPO_open_date"].dt.month == today.month
            )
            df = df.loc[month_filt, :]
            df["IPO_open_date"] = df["IPO_open_date"].dt.strftime("%Y-%m-%d")

        elif pipeline == "archive":
            # only get data from 01-09-2025 to last months
            last_month = today - relativedelta(months=1)
            start_date = pd.to_datetime("2025-09-01")
            end_date = last_month.replace(day=1) + relativedelta(months=1)

            log.info(
                f"{prefix}: Selecting {start_date.strftime('%Y %B')} to {end_date.strftime('%Y %B')} company data"
            )
            df["IPO_open_date"] = pd.to_datetime(
                df["IPO_open_date"], errors="raise", format="%Y-%m-%d"
            )
            month_filt = (df["IPO_open_date"] >= start_date) & (
                df["IPO_open_date"] < end_date
            )
            df = df.loc[month_filt, :]
            df["IPO_open_date"] = df["IPO_open_date"].dt.strftime("%Y-%m-%d")

        elif pipeline == "latest":
            log.info(f"{prefix}: Selecting {today.strftime('%Y %B')}'s company data")
            df["IPO_open_date"] = pd.to_datetime(
                df["IPO_open_date"], errors="raise", format="%Y-%m-%d"
            )
            month_filt = (df["IPO_open_date"].dt.year == today.year) & (
                df["IPO_open_date"].dt.month == today.month
            )
            df = df.loc[month_filt, :]
            df["IPO_open_date"] = df["IPO_open_date"].dt.strftime("%Y-%m-%d")

        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        LogException(e, logger=log)
        raise CustomException(e)


def df_to_json(data: pd.DataFrame = None) -> json.JSONEncoder:
    try:
        df = data.copy()
        df.reset_index(drop=True, inplace=True)
        records = list(json.loads(df.T.to_json()).values())
        return records

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def put_df_to_MongoDB(
    data: pd.DataFrame = None,
    collection: Literal[
        "IPOPredMain", "IPOPredOrig", "IPOPredTest", "IPOPredArcv"
    ] = "IPOPredMain",
    db_config: MongoDBConfig = MongoDBConfig(),
    log: Logger = log_etl,
    prefix: Literal["Loading", "Prediction"] = "Loading",
) -> pd.DataFrame:
    try:
        log.info(f"{prefix}: Converting dataframe to required format")
        records = df_to_json(data)

        db_database_name = db_config.database
        log.info(
            f"{prefix}: Communicating with MongoDB: {db_database_name}/{collection}"
        )
        db_collection_name = collection
        mongo_client = MongoClient(db_config.mongo_db_url)
        collections = mongo_client[db_database_name][db_collection_name]

        log.info(f"{prefix}: Preparing data upload/update operations")
        operations = []
        for record in records:
            filter_query = {"IPO_company_name": record.get("IPO_company_name")}
            operations.append(UpdateOne(filter_query, {"$set": record}, upsert=True))

        if operations:
            log.info(f"{prefix}: Uploading data to '{db_database_name}/{collection}'")
            collections.bulk_write(operations)

    except Exception as e:
        LogException(e, logger=log)
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

            log_trn.info(f"Model Training: Training '{model_name}' model")
            gs = GridSearchCV(
                estimator=model_object,
                param_grid=model_params,
                cv=3,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X=x_train, y=y_train)

            log_trn.info(f"Model Training: Scoring best fit '{model_name}' model")
            y_pred_train, y_pred_vald = [gs.predict(x) for x in [x_train, x_vald]]
            scores_train, scores_vald = [
                get_model_scores(y_true, y_pred)
                for y_true, y_pred in [
                    (y_train, y_pred_train),
                    (y_vald, y_pred_vald),
                ]
            ]
            log_trn.info(
                f"Model Training: train_{sort_by}={getattr(scores_train, sort_by):.4f} & vald_{sort_by}={getattr(scores_vald, sort_by):.4f}"
            )
            report[model_name] = {
                "Model_object": gs,
                "Model_score": [scores_train, scores_vald],
            }
        # reorder the dictionary in the descending order of validation set score
        log_trn.info("Model Training: Exporting evaluated best fit models")
        report_sorted = dict(
            sorted(
                report.items(),
                key=lambda item: getattr(item[1]["Model_score"][1], sort_by),
                reverse=True,
            )
        )
        return report_sorted

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)


def s3_syncer(source: str = None, destination: str = None):
    try:
        log_trn.info("Model Pushing: Syncing data to/from s3 bucket")
        command = f"aws s3 sync {source} {destination}"
        os.system(command)

    except Exception as e:
        LogException(e)
        raise CustomException(e)


def get_model_paths(latest: bool = True) -> str | list[str]:
    try:
        model_paths = glob("Artifacts/*/model_trainer/trained_model/*_model.pkl")
        model_paths = sorted(
            model_paths, key=lambda x: os.path.basename(x)[:19], reverse=True
        )
        return model_paths[0] if latest else model_paths

    except Exception as e:
        LogException(e)
        raise CustomException(e)
