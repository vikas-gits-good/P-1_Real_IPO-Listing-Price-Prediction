from src.Logging.logger import log_trn
from src.Exception.exception import CustomException
from src.Pipeline.prediction_pipeline import MakeIPOPrediction


if __name__ == "__main__":
    try:
        df_y_pred = MakeIPOPrediction().predict(
            path="Artifacts/2025_10_25_22_22_33/model_trainer/trained_model/2025-10-25_22-22-33_model.pkl"
        )

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)
