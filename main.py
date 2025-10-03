from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Components.data_ingestion import DataIngestion
from src.Components.data_validation import DataValidation


class TrainIPOPrediction:
    def __init__(self):
        pass

    def train(self):
        try:
            logging.info(f"{'Data Ingestion':-^{60}}")
            di_artf = DataIngestion().initialise()

            logging.info(f"{'Data Validation':-^{60}}")
            dv_artf = DataValidation(di_artf).initialise()

            logging.info(f"{'Data Transformation':-^{60}}")

            logging.info(f"{'Model Training':-^{60}}")

            logging.info(f"{'Model Pushing':-^{60}}")

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        TrainIPOPrediction().train()

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)
