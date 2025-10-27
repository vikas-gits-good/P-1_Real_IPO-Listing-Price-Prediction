import os

from src.ETL.etl_config.ETL_config import (
    ExtractionArtifact,
    TransformationConfig,
    TransformationArtifact,
)
from src.ETL.etl_utils.Transform_func import DataTransformer
from src.Utils.main_utils import read_dataframe, save_dataframe
from src.Logging.logger import log_etl
from src.Exception.exception import CustomException


class TransformData:
    def __init__(
        self,
        extraction_artifact: ExtractionArtifact = None,
        transformation_config: TransformationConfig = TransformationConfig(),
    ):
        self.extraction_artifact = extraction_artifact
        self.transformation_config = transformation_config

    def initiate(self):
        try:
            log_etl.info("Transformation: Started")
            log_etl.info("Transformation: Reading extracted data")
            df = read_dataframe(
                path=self.extraction_artifact.extracted_data_file_path, log_name=log_etl
            )

            log_etl.info("Transformation: Performing transformation operations")
            df_trfm = DataTransformer(data=df).transform()

            log_etl.info("Transformation: Saving transformed data to file")
            save_path = f"{os.path.dirname(self.extraction_artifact.extracted_data_file_path)}/{self.transformation_config.transformation_file_name}"
            save_dataframe(data=df_trfm, path=save_path, log_name=log_etl)

            tfm_artf = TransformationArtifact(transformed_data_file_path=save_path)
            log_etl.info("Transformation: Exporting etl transformation artifact")
            log_etl.info("Transformation: Finished")
            return tfm_artf

        except Exception as e:
            log_etl.info(f"Error in ExtractData(): {e}")
            raise CustomException(e)
