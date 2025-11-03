from src.Logging.logger import log_ful
from src.Exception.exception import CustomException, LogException
from src.Pipeline.full_pipeline import FullPipeline

if __name__ == "__main__":
    try:
        FullPipeline().run()

    except Exception as e:
        LogException(e, logger=log_ful)
        raise CustomException(e)
