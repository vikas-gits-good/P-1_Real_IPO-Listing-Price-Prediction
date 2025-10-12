import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Entity.artifact_entity import ClassificationMetricArtifact


def get_model_scores(
    y_true: np.typing.NDArray = None, y_pred: np.typing.NDArray = None
) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred, average="macro")
        model_precision_score = precision_score(y_true, y_pred, average="macro")
        model_recall_score = recall_score(y_true, y_pred, average="macro")
        cm_artf = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
        )
        return cm_artf

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


# y_train_true = [1, 1, 1, 1, 1]
# y_train_pred = [1, 0, 1, 0, 0]
# y_vald_true = [1, 1, 0]
# y_vald_pred = [1, 1, 1]
# y_test_true = [0, 1, 0]
# y_test_pred = [0, 1, 1]

# metrics_list = [
#     get_model_scores(y_true, y_pred)
#     for y_true, y_pred in [
#         (y_train_true, y_train_pred),
#         (y_vald_true, y_vald_pred),
#         (y_test_true, y_test_pred),
#     ]
# ]

# metrics_data = [
#     score
#     for metric in metrics_list
#     for score in (metric.f1_score, metric.precision_score, metric.recall_score)
# ]
# print(metrics_data)

# metrics_data = {
#     f"{dataset}_{score}": getattr(metric, score)
#     for dataset, metric in zip(["train", "vald", "test"], metrics_list)
#     for score in ["f1_score", "precision_score", "recall_score"]
# }
# print(metrics_data)
