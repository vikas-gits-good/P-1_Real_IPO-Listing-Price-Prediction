import asyncio
from src.Logging.logger import log_etl
from src.Exception.exception import CustomException
from src.ETL.ETL_main import ETLPipeline


if __name__ == "__main__":
    try:
        asyncio.run(ETLPipeline().run())

    except Exception as e:
        log_etl.info(f"Error: {e}")
        raise CustomException(e)
