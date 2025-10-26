import numpy as np
from datetime import datetime
from dataclasses import dataclass


@dataclass
class common_constants:
    TARGET_COLUMN = "IPO_listing_gain_category"
    PIPELINE_NAME = "src"
    ARTIFACT_DIR = "Artifacts"
    DATA_FILE_NAME = "ipo_scrn_gmp_EQ.csv"
    TRAIN_FILE_NAME = "train.csv"
    VALD_FILE_NAME = "valid.csv"
    TEST_FILE_NAME = "test.csv"
    RANDOM = 565
    SAVED_MODEL_DIR_NAME = "saved_models"
    SAVED_MODEL_FILE_NAME = "model.pkl"


@dataclass
class mongo_db_dc:
    CLUSTER_NAME = "Cluster0"
    DATABASE_NAME = "RealWorldProjects"
    COLLECTION_NAME_MAIN = "IPOPredMain"
    COLLECTION_NAME_ORIGINAL = "IPOPredOrig"
    COLLECTION_NAME_TESTING = "IPOPredTest"
    COLLECTION_NAME_PREDICT_ARCHIVE = "IPOPredArcv"


@dataclass
class data_ingestion:
    DATA_DIR = "src/Data/InitialData"
    DATA_INGESTION_DIR_NAME = "data_ingestion"
    FEATURE_STORE_DIR_NAME = "feature_store"
    INGESTED_DIR_NAME = "ingested"
    # This is to split data into 70 : 15: 15 - Train: Validation: Test sets
    TRAIN_SPLIT_RATIO: float = 0.7
    TEST_SPLIT_RATIO: float = 0.5


@dataclass
class data_validation:
    DATA_VALIDATION_DIR_NAME = "data_validation"
    VALID_DATA_DIR_NAME = "validated"
    INVALID_DATA_DIR_NAME = "invalid"
    DATA_DRIFT_REPORT_DIR = "drift_report"
    DRIFT_REPORT_FILE_NAME_VALD = "drift_report_vald.yaml"
    DRIFT_REPORT_FILE_NAME_TEST = "drift_report_test.yaml"
    SCHEMA_FILE_PATH = "src/Constants/validation_schema.yaml"


@dataclass
class data_transformation:
    DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
    TRANSFORMED_DATA_DIR_NAME = "transformed_data"
    TRANSFORMED_DATA_OBJECT_DIR_NAME = "transformed_object"
    TRANSFORMATION_OBJECT_NAME = "ppln_prpc.pkl"
    TRANSFORMATION_IMPUTER_PARAMS = {
        "missing_values": np.nan,
        "n_neighbors": 5,
        "weights": "uniform",
    }


@dataclass
class model_trainer:
    MODEL_TRAINER_DIR_NAME = "model_trainer"
    TRAINED_MODEL_DIR_NAME = "trained_model"
    TRAINED_MODEL_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_model.pkl"
    TRAINED_MODEL_EXPECTED_SCORE = 0.7
    TRAINED_MODEL_BAD_FIT_THRESHOLD = 0.05


@dataclass
class dagshub_constants:
    REPO_OWNER_NAME = "vikas-gits-good"
    REPO_NAME = "P-1_Real_IPO-Listing-Price-Prediction"


@dataclass
class model_pusher:
    FINAL_ARTIFACTS_DIR_NAME = "Final_Artifacts"
    FINAL_ARTIFACTS_DIR_PATH = "src/Final_Artifacts"


@dataclass
class s3_constants:
    TRAINING_BUCKET_NAME = "s3-bct-ipo-price-prediction"
