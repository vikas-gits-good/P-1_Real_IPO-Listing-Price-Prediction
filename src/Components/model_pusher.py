from src.Logging.logger import log_trn
from src.Exception.exception import CustomException, LogException

from src.Utils.main_utils import s3_syncer
from src.Entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
from src.Entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(
        self,
        artifact: ModelTrainerArtifact = None,
        model_pusher_config: ModelPusherConfig = ModelPusherConfig(),
    ):
        try:
            self.model_train_artf = artifact
            self.model_pusher_config = model_pusher_config

        except Exception as e:
            LogException(e)
            raise CustomException(e)

    def initialise(self):
        try:
            log_trn.info(
                "Model Pushing: Preparing to sync artifacts and models to AWS s3 bucket"
            )
            s3_syncer(
                self.model_pusher_config.lcl_artifact_dir,
                self.model_pusher_config.url_artifact,
            )
            s3_syncer(
                self.model_pusher_config.lcl_model_dir,
                self.model_pusher_config.url_models,
            )
            mp_artf = ModelPusherArtifact(
                local_artifact_dir=self.model_pusher_config.lcl_artifact_dir,
                local_model_dir=self.model_pusher_config.lcl_model_dir,
                cloud_artifact_dir=self.model_pusher_config.url_artifact,
                cloud_model_dir=self.model_pusher_config.url_models,
            )
            log_trn.info("Model Pushing: Exporting model pusher artifact")
            log_trn.info("Model Pushing: Finished")
            return mp_artf

        except Exception as e:
            LogException(e)
            raise CustomException(e)
