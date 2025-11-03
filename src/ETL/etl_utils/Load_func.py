import pandas as pd

from src.Logging.logger import log_etl
from src.Exception.exception import CustomException, LogException
from src.Utils.main_utils import put_df_to_MongoDB


class DataLoader:
    def __init__(self, Data: pd.DataFrame = None) -> None:
        try:
            self.data = Data

        except Exception as e:
            LogException(e)
            raise CustomException(e)

    def load(self):
        try:
            log_etl.info("Loading: Preparing loading operations")
            put_df_to_MongoDB(
                data=self.data, collection="IPOPredMain", log=log_etl, prefix="Loading"
            )

        except Exception as e:
            LogException(e)
            raise CustomException(e)
