import asyncio

from src.ETL.etl_components.ETL_Extraction import ExtractData
from src.ETL.etl_components.ETL_Transformation import TransformData
from src.ETL.etl_components.ETL_Loading import LoadData

from src.Logging.logger import log_etl
from src.Exception.exception import CustomException, LogException


class ETLPipeline:
    def __init__(self):
        pass

    async def scrape(self):
        try:
            log_etl.info(f"{'Extraction':-^{60}}")
            ext_artf = await ExtractData().initiate()

            log_etl.info(f"{'Transformation':-^{60}}")
            tfm_artf = TransformData(ext_artf).initiate()

            log_etl.info(f"{'Loading':-^{60}}")
            LoadData(tfm_artf).initiate()

        except Exception as e:
            LogException(e, logger=log_etl)
            raise CustomException(e)


if __name__ == "__main__":
    try:
        asyncio.run(ETLPipeline().run())

    except Exception as e:
        LogException(e, logger=log_etl)
        raise CustomException(e)
