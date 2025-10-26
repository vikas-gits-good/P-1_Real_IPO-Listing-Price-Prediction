from src.Logging.logger import log_trn
from src.Exception.exception import CustomException
from src.Pipeline.training_pipeline import TrainIPOPrediction


if __name__ == "__main__":
    try:
        TrainIPOPrediction().train()

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)
