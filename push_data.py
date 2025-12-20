import pandas as pd

from src.Logging.logger import log_etl
from src.Exception.exception import LogException, CustomException

from src.Utils.main_utils import put_df_to_MongoDB
from src.Constants import data_ingestion, common_constants


class InitialiseMongoDBData:
    def __init__(self) -> None:
        pass

    def upsert(self, path: str):
        try:
            df = pd.read_csv(path)
            _ = put_df_to_MongoDB(
                data=df,
                collection="IPOPredMain",
                log=log_etl,
                prefix="Loading",
            )

        except Exception as e:
            LogException(e, logger=log_etl)
            raise CustomException(e)


if __name__ == "__main__":
    FILE_PATH = f"{data_ingestion.DATA_DIR}/{common_constants.DATA_FILE_NAME}"
    log_etl.info("ETL | Pushing Started")
    _ = InitialiseMongoDBData().upsert(path=FILE_PATH)
    log_etl.info("ETL | Pushing Finished")
