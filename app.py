import os
import json
import logging
import asyncio
import calendar
import threading
import pandas as pd
from glob import glob
from typing import Literal
from datetime import datetime
from flask import Flask, request, render_template, send_file, abort, jsonify

from src.Logging.logger import log_etl, log_trn, log_prd, log_flk
from src.Exception.exception import CustomException, LogException

from src.ETL.ETL_main import ETLPipeline
from src.Pipeline.training_pipeline import TrainIPOPrediction
from src.Pipeline.prediction_pipeline import MakeIPOPrediction


class AplcOps:
    def __init__(
        self, pipeline: Literal["etl", "train", "pred"] = "etl", pred_file_path=None
    ):
        self.log_path = f"./logs/{pipeline}"
        self.log_file = f"*_{pipeline}.log"
        self.logger = {"etl": log_etl, "train": log_trn, "pred": log_prd}.get(
            pipeline, log_etl
        )
        self.function = {
            "etl": ETLPipeline().run,
            "train": TrainIPOPrediction().train,  # <- Dont call here ()
            "pred": MakeIPOPrediction().predict,  # <- Use ops_prd.get_pred_func() to get callable
        }.get(pipeline, ETLPipeline().run)

    def get_latest_log(self):
        try:
            log_dir = os.path.abspath(self.log_path)
            log_files = glob(os.path.join(log_dir, self.log_file))
            if not log_files:
                return jsonify({"error": "No logs found"}), 404
            # Sort files by modified time descending
            log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_log = os.path.basename(log_files[0])
            return jsonify({"latest_log": latest_log})
        except Exception as e:
            LogException(e)
            # raise CustomException(e)
            return jsonify({"error": "Internal Server Error"}), 500

    def run_function_background(self, *args, **kwargs):
        try:
            self.logger.info("User Action: Started")
            if asyncio.iscoroutinefunction(self.function):
                asyncio.run(self.function(*args, **kwargs))
            else:
                self.function(*args, **kwargs)
            self.logger.info("User Action: Finished")

        except Exception as e:
            LogException(e)
            # raise CustomException(e)

    def run_function(self, *args, **kwargs):
        try:
            thread = threading.Thread(
                target=self.run_function_background, args=args, kwargs=kwargs
            )
            thread.start()
            return json.dumps({"status": "started"})

        except Exception as e:
            LogException(e)
            self.logger.info(f"Error: {e}")
            # raise CustomException(e)
            return abort(500, "Problem during retraining")


class UtilOps:
    def __init__(self):
        pass

    def get_log_content(self):
        try:
            filename = request.args.get("file")
            log_type = request.args.get("type", "train").lower()

            if not filename:
                return abort(400, "Missing file parameter")

            # Determine log directory based on type
            valid_log_dirs = {
                "train": "./logs/train",
                "pred": "./logs/pred",
                "etl": "./logs/etl",
            }
            if log_type not in valid_log_dirs:
                abort(400, "Invalid log type")

            log_dir = os.path.abspath(valid_log_dirs[log_type])

            file_path = os.path.abspath(os.path.join(log_dir, filename))
            # Security check: ensure the file is within the correct directory
            if not file_path.startswith(log_dir):
                return abort(403, "Access denied")

            if not os.path.exists(file_path):
                return abort(404, "File not found")

            return send_file(file_path, mimetype="text/plain")

        except Exception as e:
            LogException(e)
            return abort(404, "File not found")

    def home(self):
        log_prd.info(f"{'Initialise':-^{60}}")
        log_prd.info("Initialising data to display")
        df = pd.read_csv("src/Data/InitialData/ipo_scrn_gmp_EQ.csv")
        cols_list = [
            "IPO_company_name",
            "IPO_open_date",
            "IPO_close_date",
            "IPO_issue_price",
            "IPO_lot_size",
            "IPO_Broker_apply",
            "IPO_Member_apply",
            "IPO_day3_qib",
            "IPO_day3_nii",
            "IPO_day3_rtl",
        ]
        df_arch = df.iloc[:30, :].loc[:, cols_list]
        df_lats = df.iloc[-30:, :].loc[:, cols_list]

        archive_json = json.dumps(df_arch.to_dict(orient="records"))
        latest_json = json.dumps(df_lats.to_dict(orient="records"))

        def get_files():
            log_prd.info("Getting log files and model files")
            etl_log_files = glob("logs/etl/*_etl.log")
            trn_log_files = glob("logs/train/*_train.log")
            prd_log_files = glob("logs/pred/*_pred.log")
            trn_mdl_files = glob("Artifacts/*/model_trainer/trained_model/*_model.pkl")

            log_prd.info("Sorting files by date")
            # sort by date in name [:19] -> '%Y-%m-%d_%H-%M-%S'
            etl_log_files, trn_log_files, prd_log_files, trn_mdl_files = [
                sorted(file_list, key=lambda x: os.path.basename(x)[:19], reverse=True)
                for file_list in (
                    etl_log_files,
                    trn_log_files,
                    prd_log_files,
                    trn_mdl_files,
                )
            ]

            log_prd.info("Removing empty files from list")
            etl_log_files, trn_log_files, prd_log_files, trn_mdl_files = [
                [item for item in file_list if os.path.getsize(item) > 0]
                for file_list in (
                    etl_log_files,
                    trn_log_files,
                    prd_log_files,
                    trn_mdl_files,
                )
            ]
            return etl_log_files, trn_log_files, prd_log_files, trn_mdl_files

        etl_log_files, trn_log_files, prd_log_files, trn_mdl_files = get_files()

        # current month, year and previous month, year data as string
        today = datetime.today()
        this_month_name = today.strftime("%B")
        this_year_name = today.strftime("%Y")
        prev_month_num = today.month - 1 if today.month > 1 else 12
        prev_month_name = calendar.month_name[prev_month_num]
        prev_year_name = str(today.year if today.month > 1 else today.year - 1)

        data_to_html = {
            "LatestData": latest_json,
            "ArchiveData": archive_json,
            "LogFiles": trn_log_files,
            "EtlLogFiles": etl_log_files,
            "PrdLogFile": prd_log_files,
            "ModelFiles": trn_mdl_files,
            "CrntMonth": this_month_name,
            "CrntYear": this_year_name,
            "PrevMonth": prev_month_name,
            "PrevYear": prev_year_name,
        }
        return render_template("index.html", data_dict=data_to_html)


# I want flask's default logger to go to log_flk
werkzeug_logger = logging.getLogger("werkzeug")
for handler in werkzeug_logger.handlers[:]:
    werkzeug_logger.removeHandler(handler)
werkzeug_logger.addHandler(log_flk.handlers[0])

# initialise app
application = Flask(__name__)
app = application

# # initialise operations
ops_etl = AplcOps(pipeline="etl")
ops_trn = AplcOps(pipeline="train")
ops_prd = AplcOps(pipeline="pred")
ops_utl = UtilOps()


# ETL API calls
@app.route("/api/logs/latest_etl")
def latest_etl_log():
    return ops_etl.get_latest_log()


@app.route("/api/etl_update")
def etl_update():
    return ops_etl.run_function()


# Train API calls
@app.route("/api/logs/latest_train")
def latest_train_log():
    return ops_trn.get_latest_log()


@app.route("/api/trn_retrain")
def retrain():
    return ops_trn.run_function()


# Predict API calls
@app.route("/api/logs/latest_pred")
def latest_pred_log():
    return ops_prd.get_latest_log()


@app.route("/api/prd_predict")
def predict():
    model_path = request.args.get("model")
    return ops_prd.run_function(path=model_path)


# Util API calls
@app.route("/api/logs/content")
def get_log_text():
    return ops_utl.get_log_content()


@app.route("/")
def home_page():
    return ops_utl.home()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
