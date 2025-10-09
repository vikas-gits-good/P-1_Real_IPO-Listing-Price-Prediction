import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Entity.artifact_entity import ClassificationMetricArtifact


def get_model_scores(
    y_true: np.typing.NDArray = None, y_pred: np.typing.NDArray = None
) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred, average="micro")
        model_precision_score = precision_score(y_true, y_pred, average="micro")
        model_recall_score = recall_score(y_true, y_pred, average="micro")
        cm_artf = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
        )
        return cm_artf

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)
