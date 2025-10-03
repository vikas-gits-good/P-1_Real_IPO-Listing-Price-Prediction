import os
from datetime import datetime

from src.Constants import common_constants, mongo_db_dc, data_ingestion, data_validation


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        self.timestamp = timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        self.pipeline_name = common_constants.PIPELINE_NAME
        self.artifact_dir_name = common_constants.ARTIFACT_DIR
        self.artifact_dir_path = os.path.join(self.artifact_dir_name, self.timestamp)


class DataIngestionConfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig(),
    ):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir_path,
            data_ingestion.DATA_INGESTION_DIR_NAME,
        )
        self.feature_store_dir = os.path.join(
            self.data_ingestion_dir,
            data_ingestion.FEATURE_STORE_DIR_NAME,
            common_constants.DATA_FILE_NAME,
        )
        self.train_dir = os.path.join(
            self.data_ingestion_dir,
            data_ingestion.INGESTED_DIR_NAME,
            common_constants.TRAIN_FILE_NAME,
        )
        self.vald_dir = os.path.join(
            self.data_ingestion_dir,
            data_ingestion.INGESTED_DIR_NAME,
            common_constants.VALD_FILE_NAME,
        )
        self.test_dir = os.path.join(
            self.data_ingestion_dir,
            data_ingestion.INGESTED_DIR_NAME,
            common_constants.TEST_FILE_NAME,
        )
        self.train_size = data_ingestion.TRAIN_SPLIT_RATIO
        self.test_size = data_ingestion.TEST_SPLIT_RATIO
        self.collection_name = mongo_db_dc.COLLECTION_NAME
        self.database_name = mongo_db_dc.DATABASE_NAME
        self.random_state = common_constants.RANDOM


# for var, val in vars(DataIngestionConfig()).items():
#     print(f"{var}: {val}")


class DataValidationConfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig(),
    ):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir_path,
            data_validation.DATA_VALIDATION_DIR_NAME,
        )
        self.valid_train_data_path = os.path.join(
            self.data_validation_dir,
            data_validation.VALID_DATA_DIR_NAME,
            common_constants.TRAIN_FILE_NAME,
        )
        self.valid_vald_data_path = os.path.join(
            self.data_validation_dir,
            data_validation.VALID_DATA_DIR_NAME,
            common_constants.VALD_FILE_NAME,
        )
        self.valid_test_data_path = os.path.join(
            self.data_validation_dir,
            data_validation.VALID_DATA_DIR_NAME,
            common_constants.TEST_FILE_NAME,
        )
        self.invalid_train_data_path = os.path.join(
            self.data_validation_dir,
            data_validation.INVALID_DATA_DIR_NAME,
            common_constants.TRAIN_FILE_NAME,
        )
        self.invalid_vald_data_path = os.path.join(
            self.data_validation_dir,
            data_validation.INVALID_DATA_DIR_NAME,
            common_constants.VALD_FILE_NAME,
        )
        self.invalid_test_data_path = os.path.join(
            self.data_validation_dir,
            data_validation.INVALID_DATA_DIR_NAME,
            common_constants.TEST_FILE_NAME,
        )
        self.drift_report_path_vald = os.path.join(
            self.data_validation_dir,
            data_validation.DATA_DRIFT_REPORT_DIR,
            data_validation.DRIFT_REPORT_FILE_NAME_VALD,
        )
        self.drift_report_path_test = os.path.join(
            self.data_validation_dir,
            data_validation.DATA_DRIFT_REPORT_DIR,
            data_validation.DRIFT_REPORT_FILE_NAME_TEST,
        )


# for var, val in vars(DataValidationConfig()).items():
#     print(f"{var}: {val}")
