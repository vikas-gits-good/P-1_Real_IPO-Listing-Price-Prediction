import mlflow
import dagshub
import functools
import numpy as np
from typing import List
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import RandomForestClassifier

from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Constants import common_constants, dagshub_constants
from src.Constants.model_constants import model_dict, create_model
from src.Entity.config_entity import ModelTrainerConfig
from src.Entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.Utils.main_utils import (
    read_numpy_array,
    read_transformation_object,
    save_model_object,
    evaluate_models,
)
from src.Utils.estimator import NetworkModel
from src.Utils.ml_utils import get_model_scores, ClassificationMetricArtifact


class ModelTrainer:
    def __init__(
        self,
        artifact: DataTransformationArtifact = None,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
    ):
        try:
            self.data_transformation_artifact = artifact
            self.model_trainer_config = model_trainer_config
            dagshub.init(
                repo_owner=dagshub_constants.REPO_OWNER_NAME,
                repo_name=dagshub_constants.REPO_NAME,
                mlflow=True,
            )

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def track_mlflow(
        self,
        model: RandomForestClassifier = None,
        metrics: List[ClassificationMetricArtifact] = None,
    ) -> None:
        try:
            mlflow.set_tracking_uri(
                f"https://dagshub.com/{dagshub_constants.REPO_OWNER_NAME}/{dagshub_constants.REPO_NAME}.mlflow"
            )
            with mlflow.start_run():
                metrics_data = {
                    f"{dataset}_{score}": getattr(metric, score)
                    for dataset, metric in zip(["train", "vald", "test"], metrics)
                    for score in ["f1_score", "precision_score", "recall_score"]
                }
                mlflow.log_metrics(metrics=metrics_data)
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        except Exception as e:
            logging.info(f"Error in model_trainer.py: {e}")
            # raise CustomException(e)

    def train_model(
        self,
        x_train: np.typing.NDArray = None,
        y_train: np.typing.NDArray = None,
        x_vald: np.typing.NDArray = None,
        y_vald: np.typing.NDArray = None,
        x_test: np.typing.NDArray = None,
        y_test: np.typing.NDArray = None,
    ) -> dict:
        try:
            if "TFNeuralNetwork" in model_dict.keys():
                model_dict["TFNeuralNetwork"]["Model"] = KerasClassifier(
                    model=functools.partial(
                        create_model, input_shape=(x_train.shape[1],)
                    ),
                    index=0,
                    random_state=common_constants.RANDOM,
                )

            models_report = evaluate_models(
                x_train, y_train, x_vald, y_vald, models=model_dict, sort_by="f1_score"
            )
            best_model_name = list(models_report.keys())[0]
            best_model_object = models_report[best_model_name]["Model_object"]
            best_model_score = models_report[best_model_name]["Model_score"]

            logging.info(
                f"Model Training: Scoring best performing model '{best_model_name}' on test set"
            )
            y_pred_test = best_model_object.predict(x_test)
            best_model_score += [get_model_scores(y_true=y_test, y_pred=y_pred_test)]

            logging.info(
                f"Model Training: Using MLFlow to track {best_model_name}'s metrics"
            )
            self.track_mlflow(best_model_object, best_model_score)

            logging.info(
                f"Model Training: {best_model_name}'s test set scores: f1_score={best_model_score[2].f1_score:.4f}, precision_score={best_model_score[2].precision_score:.4f}, recall_score={best_model_score[2].recall_score:.4f}"
            )

            logging.info(
                f"Model Training: Saving best fit '{best_model_name}' model to file"
            )
            ppln_prpc = read_transformation_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            nm_object = NetworkModel(preprocessor=ppln_prpc, model=best_model_object)
            save_model_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=nm_object,
            )
            save_model_object(
                file_path="src/Final_Artifacts/ppln_prpc.pkl", object=ppln_prpc
            )
            save_model_object(
                file_path="src/Final_Artifacts/final_model.pkl",
                object=best_model_object,
            )

            best_model_dict = {
                "Model_name": best_model_name,
                "Model_object": best_model_object,
                "Model_score": best_model_score,
            }
            return best_model_dict

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def initialise(self) -> ModelTrainerArtifact:
        try:
            logging.info("Model Training: Started")
            logging.info("Model Training: Getting transformed data from file")
            ary_train, ary_vald, ary_test = [
                read_numpy_array(path)
                for path in [
                    self.data_transformation_artifact.transformed_train_file_path,
                    self.data_transformation_artifact.transformed_valid_file_path,
                    self.data_transformation_artifact.transformed_test_file_path,
                ]
            ]
            (x_train, y_train), (x_vald, y_vald), (x_test, y_test) = [
                (ary[:, :-1], ary[:, -1]) for ary in [ary_train, ary_vald, ary_test]
            ]

            bm_dict = self.train_model(x_train, y_train, x_vald, y_vald, x_test, y_test)

            mt_artf = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artf_train_set=bm_dict["Model_score"][0],
                metric_artf_vald_set=bm_dict["Model_score"][1],
                metric_artf_test_set=bm_dict["Model_score"][2],
            )
            logging.info("Model Training: Exporting model trainer artifact")
            logging.info("Model Training: Finished")
            return mt_artf

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
