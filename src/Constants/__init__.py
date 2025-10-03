from dataclasses import dataclass


@dataclass
class common_constants:
    TARGET_COLUMN = "IPO_listing_price"
    PIPELINE_NAME = "src"
    ARTIFACT_DIR = "Artifacts"
    DATA_FILE_NAME = "ipo_scrn_gmp_EQ.csv"
    TRAIN_FILE_NAME = "train.csv"
    VALD_FILE_NAME = "valid.csv"
    TEST_FILE_NAME = "test.csv"

    RANDOM = 257


@dataclass
class mongo_db_dc:
    CLUSTER_NAME = "Cluster0"
    DATABASE_NAME = "RealWorldProjects"
    COLLECTION_NAME = "IPOPred"


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
