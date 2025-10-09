from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    vald_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_vald_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_vald_file_path: str
    invalid_test_file_path: str
    drift_report_file_path_vald: str
    drift_report_file_path_test: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_valid_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artf_train_set: ClassificationMetricArtifact
    metric_artf_vald_set: ClassificationMetricArtifact
    metric_artf_test_set: ClassificationMetricArtifact
