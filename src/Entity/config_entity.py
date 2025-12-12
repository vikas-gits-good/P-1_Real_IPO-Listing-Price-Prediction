import os
from datetime import datetime
from dotenv import load_dotenv

from src.Constants import (
    common_constants,
    mongo_db_dc,
    data_ingestion,
    data_validation,
    data_transformation,
    model_trainer,
    model_pusher,
    s3_constants,
)


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        self.timestamp = timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        self.pipeline_name = common_constants.PIPELINE_NAME
        self.artifact_dir_name = common_constants.ARTIFACT_DIR
        self.artifact_dir_path = os.path.join(self.artifact_dir_name, self.timestamp)
        self.model_dir = os.path.join(model_pusher.FINAL_ARTIFACTS_DIR_PATH)


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


class DataTransformationConfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig(),
    ):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir_path,
            data_transformation.DATA_TRANSFORMATION_DIR_NAME,
        )
        self.trfm_train_file_path = os.path.join(
            self.data_transformation_dir,
            data_transformation.TRANSFORMED_DATA_DIR_NAME,
            common_constants.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.trfm_vald_file_path = os.path.join(
            self.data_transformation_dir,
            data_transformation.TRANSFORMED_DATA_DIR_NAME,
            common_constants.VALD_FILE_NAME.replace("csv", "npy"),
        )
        self.trfm_test_file_path = os.path.join(
            self.data_transformation_dir,
            data_transformation.TRANSFORMED_DATA_DIR_NAME,
            common_constants.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.trfm_object_file_path = os.path.join(
            self.data_transformation_dir,
            data_transformation.TRANSFORMED_DATA_OBJECT_DIR_NAME,
            data_transformation.TRANSFORMATION_OBJECT_NAME,
        )


# for var, val in vars(DataTransformationConfig()).items():
#     print(f"{var}: {val}")


class ModelTrainerConfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig(),
    ):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir_path,
            model_trainer.MODEL_TRAINER_DIR_NAME,
        )
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            model_trainer.TRAINED_MODEL_DIR_NAME,
            model_trainer.TRAINED_MODEL_NAME,
        )
        self.expected_model_score = model_trainer.TRAINED_MODEL_EXPECTED_SCORE
        self.bad_fit_threshold = model_trainer.TRAINED_MODEL_BAD_FIT_THRESHOLD


# for var, val in vars(ModelTrainerConfig()).items():
#     print(f"{var}: {val}")


class ModelPusherConfig:
    def __init__(
        self,
        training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig(),
    ):
        self.url_artifact = f"s3://{s3_constants.TRAINING_BUCKET_NAME}/artifact/{training_pipeline_config.timestamp}"
        self.url_models = f"s3://{s3_constants.TRAINING_BUCKET_NAME}/final_models/{training_pipeline_config.timestamp}"
        self.lcl_artifact_dir = training_pipeline_config.artifact_dir_path
        self.lcl_model_dir = training_pipeline_config.model_dir


# for var, val in vars(ModelPusherConfig()).items():
#     print(f"{var}: {val}")


class AngelOneConfig:
    def __init__(self):
        load_dotenv("src/Secrets/angel_one_historic.env")
        self.ao_token_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        self.ao_api_key = os.getenv("ANGEL_ONE_API_KEY")
        self.ao_client_id = os.getenv("ANGEL_ONE_CLIENT_ID")
        self.ao_pin = os.getenv("ANGEL_ONE_PIN")
        self.ao_qr_token = os.getenv("ANGEL_ONE_QR_TOKEN")


# for var, val in vars(AngelOneConfig()).items():
#     print(f"{var}: {val}")


class MongoDBConfig:
    def __init__(self):
        load_dotenv("src/Secrets/mongo_db.env")
        MONGO_DB_UN = os.getenv("MONGO_DB_UN")
        MONGO_DB_PW = os.getenv("MONGO_DB_PW")
        self.mongo_db_url = f"mongodb+srv://{MONGO_DB_UN}:{MONGO_DB_PW}@cluster0.8y5aipc.mongodb.net/?retryWrites=true&w=majority&appName={mongo_db_dc.CLUSTER_NAME}"
        self.database = mongo_db_dc.DATABASE_NAME
        self.collection_main = mongo_db_dc.COLLECTION_NAME_MAIN
        self.collection_orig = mongo_db_dc.COLLECTION_NAME_ORIGINAL
        self.collection_test = mongo_db_dc.COLLECTION_NAME_TESTING
        self.collection_pred_arcv = mongo_db_dc.COLLECTION_NAME_PREDICT_ARCHIVE


# for var, val in vars(MongoDBConfig()).items():
#     print(f"{var}: {val}")
