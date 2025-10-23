import asyncio
from src.Logging.logger_etl import logging
from src.Exception.exception import CustomException
from src.ETL.ETL_main import ETLPipeline


if __name__ == "__main__":
    try:
        asyncio.run(ETLPipeline().run())

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)
