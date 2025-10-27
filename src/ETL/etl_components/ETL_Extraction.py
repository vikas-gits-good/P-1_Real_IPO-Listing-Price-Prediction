from src.ETL.etl_utils.Extract_func import (
    CompanyListExtractor,
    CheckDatabase,
    IPODataExtractor,
    ScreenerExtractor,
    ListingPriceExtractor,
)
from src.ETL.etl_config.ETL_config import ExtractionConfig, ExtractionArtifact
from src.ETL.etl_config import etl_constants
from src.Utils.main_utils import save_dataframe
from src.Logging.logger import log_etl
from src.Exception.exception import CustomException


class ExtractData:
    def __init__(self, extraction_config: ExtractionConfig = ExtractionConfig()):
        self.extraction_config = extraction_config

    async def initiate(self):
        try:
            log_etl.info("Extraction: Started")
            log_etl.info("Extraction: Getting latest IPO company names")
            df_main = await CompanyListExtractor(
                start_year=self.extraction_config.right_now.year,
                end_year=self.extraction_config.right_now.year,
            ).extract()
            # df_main = df_main.iloc[:20, :]

            log_etl.info("Extraction: Filtering data to scrape")
            df_main = CheckDatabase(Data=df_main, drop_sme=True).filter()

            log_etl.info("Extraction: Getting companies' details")
            df_main = await IPODataExtractor(Data=df_main).extract()

            log_etl.info("Extraction: Getting companies' screener data")
            df_main = await ScreenerExtractor(
                Data=df_main, screener_crawl_method="html"
            ).extract()

            log_etl.info("Extraction: Getting companies' listing price")
            df_main = ListingPriceExtractor(Data=df_main).extract()

            log_etl.info("Extraction: Saving data to file")
            save_path = f"{self.extraction_config.data_dir_path}/{etl_constants.EXTRACTED_DATA_FILE_NAME}"
            save_dataframe(data=df_main, path=save_path, log_name=log_etl)
            ext_artf = ExtractionArtifact(extracted_data_file_path=save_path)

            log_etl.info("Extraction: Exporting etl extraction artifact")
            log_etl.info("Extraction: Finished")
            return ext_artf

        except Exception as e:
            log_etl.info(f"Error in ExtractData(): {e}")
            raise CustomException(e)
