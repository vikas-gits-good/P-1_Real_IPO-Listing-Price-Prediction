import asyncio
from src.Logging.logger import log_ful
from src.Exception.exception import CustomException, LogException
from src.Utils.main_utils import get_model_paths

from src.Pipeline.ETL_pipeline import ETLPipeline
from src.Pipeline.training_pipeline import TrainIPOPrediction
from src.Pipeline.prediction_pipeline import MakeIPOPrediction


class FullPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            log_ful.info(f"{'ETL Pipeline'::^{60}}")
            asyncio.run(ETLPipeline().scrape())

            log_ful.info(f"{'Training Pipeline'::^{60}}")
            TrainIPOPrediction().train()

            log_ful.info(f"{'Prediction Pipeline'::^{60}}")
            MakeIPOPrediction().predict(path=get_model_paths(latest=True))

        except Exception as e:
            LogException(e, logger=log_ful)
            raise CustomException(e)


if __name__ == "__main__":
    try:
        FullPipeline().run()

    except Exception as e:
        LogException(e, logger=log_ful)
        raise CustomException(e)
