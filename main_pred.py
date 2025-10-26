import os
from glob import glob
from src.Logging.logger import log_trn
from src.Exception.exception import CustomException
from src.Pipeline.prediction_pipeline import MakeIPOPrediction


if __name__ == "__main__":
    try:
        model_paths = glob("Artifacts/*/model_trainer/trained_model/*_model.pkl")
        model_paths = sorted(
            model_paths, key=lambda x: os.path.basename(x)[:19], reverse=True
        )
        df_y_pred = MakeIPOPrediction().predict(path=model_paths[0])

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)
