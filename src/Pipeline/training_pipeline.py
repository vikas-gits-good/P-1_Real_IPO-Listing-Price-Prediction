from src.Logging.logger import log_trn
from src.Exception.exception import CustomException
from src.Components.data_ingestion import DataIngestion
from src.Components.data_validation import DataValidation
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer


class TrainIPOPrediction:
    def __init__(self):
        pass

    def train(self):
        try:
            log_trn.info(f"{'Data Ingestion':-^{60}}")
            di_artf = DataIngestion().initialise()

            log_trn.info(f"{'Data Validation':-^{60}}")
            dv_artf = DataValidation(di_artf).initialise()

            log_trn.info(f"{'Data Transformation':-^{60}}")
            dt_artf = DataTransformation(dv_artf).initialise()

            log_trn.info(f"{'Model Training':-^{60}}")
            mt_artf = ModelTrainer(dt_artf).initialise()

            log_trn.info(f"{'Model Pushing':-^{60}}")

        except Exception as e:
            log_trn.info(f"Error: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        TrainIPOPrediction().train()

    except Exception as e:
        log_trn.info(f"Error: {e}")
        raise CustomException(e)
