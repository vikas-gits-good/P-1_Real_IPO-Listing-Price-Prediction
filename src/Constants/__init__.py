from dataclasses import dataclass


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
    TEST_SPLIT_RATIO: float = 0.2


@dataclass
class common_constants:
    TARGET_COLUMN = "IPO_listing_price"
    PIPELINE_NAME = "src"
    ARTIFACT_DIR = "Artifacts"
    DATA_FILE_NAME = "ipo_scrn_gmp_EQ.csv"
    TRAIN_FILE_NAME = "train.csv"
    TEST_FILE_NAME = "test.csv"
