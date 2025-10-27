import asyncio

from src.ETL.etl_components.ETL_Extraction import ExtractData
from src.ETL.etl_components.ETL_Transformation import TransformData
from src.ETL.etl_components.ETL_Loading import LoadData

from src.Logging.logger import log_etl
from src.Exception.exception import CustomException


class ETLPipeline:
    def __init__(self):
        pass

    async def run(self):
        try:
            log_etl.info(f"{'Extraction':-^{60}}")
            ext_artf = await ExtractData().initiate()

            log_etl.info(f"{'Transformation':-^{60}}")
            tfm_artf = TransformData(ext_artf).initiate()

            log_etl.info(f"{'Loading':-^{60}}")
            LoadData(tfm_artf).initiate()

        except Exception as e:
            log_etl.info(f"Error in ETL_Pipeline(): {e}")
            raise CustomException(e)


if __name__ == "__main__":
    asyncio.run(ETLPipeline().run())
