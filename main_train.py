from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Pipeline.training_pipeline import TrainIPOPrediction


if __name__ == "__main__":
    try:
        TrainIPOPrediction().train()

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)
