import os
import json
import threading
import pandas as pd
from glob import glob
from flask import Flask, request, render_template, send_file, abort, jsonify

from src.Logging.logger_pred import logging
from src.Pipeline.training_pipeline import TrainIPOPrediction

application = Flask(__name__)
app = application


@app.route("/api/logs/latest_pred")
def get_latest_pred_log():
    try:
        pred_log_dir = os.path.abspath("./logs/pred")
        log_files = glob(os.path.join(pred_log_dir, "*_pred.log"))
        if not log_files:
            return jsonify({"error": "No logs found"}), 404
        # Sort files by modified time descending
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_log = os.path.basename(log_files[0])
        return jsonify({"latest_log": latest_log})
    except Exception as e:
        logging.error(f"Error getting latest pred log: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


def run_training_background():
    try:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        TrainIPOPrediction().train()
        logging.getLogger("werkzeug").setLevel(logging.INFO)
    except Exception as e:
        logging.error(f"Error in background retrain: {e}")


@app.route("/api/retrain")
def retrain():
    try:
        logging.info("User Retrain: Started")
        thread = threading.Thread(target=run_training_background)
        thread.start()
        return json.dumps({"status": "started"})
    except Exception as e:
        logging.error(f"Error in retrain(): {e}")
        return abort(500, "Problem during retraining")


@app.route("/api/logs/content")
def get_log_content():
    try:
        filename = request.args.get("file")
        pred = request.args.get("pred", "false").lower() == "true"
        if not filename:
            return abort(400, "Missing file parameter")

        log_dir = os.path.abspath("./logs/pred" if pred else "./logs/train")
        file_path = os.path.abspath(os.path.join(log_dir, filename))

        # Security check: file must be in log_dir
        if not file_path.startswith(log_dir):
            return abort(403, "Access denied")

        if not os.path.exists(file_path):
            return abort(404, "File not found")

        return send_file(file_path, mimetype="text/plain")

    except Exception as e:
        logging.info(f"Error in get_log_content(): {e}")
        return abort(404, "File not found")


@app.route("/")
def home():
    df = pd.read_csv("./src/Data/InitialData/ipo_scrn_gmp_EQ.csv")
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
    log_files = glob("logs/train/*.log")
    model_files = glob("Artifacts/*/model_trainer/trained_model/*_model.pkl")

    data_to_html = {
        "LatestData": latest_json,
        "ArchiveData": archive_json,
        "LogFiles": log_files,
        "ModelFiles": model_files,
    }
    return render_template("index.html", data_dict=data_to_html)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
