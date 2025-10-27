from src.ETL.etl_config.ETL_config import TransformationArtifact
from src.ETL.etl_utils.Load_func import DataLoader
from src.Utils.main_utils import read_dataframe
from src.Logging.logger import log_etl
from src.Exception.exception import CustomException


class LoadData:
    def __init__(self, transformation_artifact: TransformationArtifact = None):
        self.transformation_artifact = transformation_artifact

    def initiate(self):
        try:
            log_etl.info("Loading: Started")
            log_etl.info("Loading: Reading transformed data")
            df = read_dataframe(
                path=self.transformation_artifact.transformed_data_file_path,
                log_name=log_etl,
            )

            log_etl.info("Loading: Performing Loading operations")
            DataLoader(Data=df).load()

            log_etl.info("Loading: Finished")

        except Exception as e:
            log_etl.info(f"Error in ExtractData(): {e}")
            raise CustomException(e)
