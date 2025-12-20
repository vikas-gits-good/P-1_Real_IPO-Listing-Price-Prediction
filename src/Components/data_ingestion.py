import pandas as pd

from sklearn.model_selection import train_test_split

from src.Logging.logger import log_trn
from src.Exception.exception import CustomException, LogException
from src.Constants import common_constants
from src.Entity.config_entity import DataIngestionConfig
from src.Entity.artifact_entity import DataIngestionArtifact
from src.Utils.main_utils import save_dataframe, get_df_from_MongoDB


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig(),
    ):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            log_trn.info(f"Error: {e}")
            raise CustomException(e)

    def get_indices(self, df, X_split):
        try:
            return df[
                df.drop(columns=common_constants.TARGET_COLUMN)
                .apply(tuple, axis=1)
                .isin(
                    pd.DataFrame(
                        X_split, columns=df.columns.drop(common_constants.TARGET_COLUMN)
                    ).apply(tuple, axis=1)
                )
            ].index

        except Exception as e:
            log_trn.info(f"Error: {e}")
            raise CustomException(e)

    def initialise(self) -> DataIngestionArtifact:
        try:
            log_trn.info("Data Ingestion: Started")
            log_trn.info("Data Ingestion: Getting data from database")
            df_main = get_df_from_MongoDB(
                collection="IPOPredMain",
                pipeline="train",
                log=log_trn,
                prefix="Data Ingestion",
            )

            log_trn.info("Data Ingestion: Performing train-valid-test split")
            df_train, df_valid = train_test_split(
                df_main,
                train_size=self.data_ingestion_config.train_size,
                random_state=common_constants.RANDOM,
                stratify=df_main[common_constants.TARGET_COLUMN],
            )
            df_train.to_pickle("./df_train.pkl")
            df_valid.to_pickle("./df_valid_0.pkl")
            df_valid, df_test = train_test_split(
                df_valid,
                test_size=self.data_ingestion_config.test_size,
                random_state=common_constants.RANDOM,
                # stratify=df_valid[common_constants.TARGET_COLUMN],
            )  # cat_1 has only 1 value causing issues. so disabling stratify
            df_valid.to_pickle("./df_valid_1.pkl")
            df_test.to_pickle("./df_test.pkl")

            log_trn.info("Data Ingestion: Saving ingested data to file")
            for data, path in [
                (df_main, self.data_ingestion_config.feature_store_dir),
                (df_train, self.data_ingestion_config.train_dir),
                (df_valid, self.data_ingestion_config.vald_dir),
                (df_test, self.data_ingestion_config.test_dir),
            ]:
                save_dataframe(data=data, path=path)

            log_trn.info("Data Ingestion: Exporting data ingestion artifact")
            di_artf = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_dir,
                vald_file_path=self.data_ingestion_config.vald_dir,
                test_file_path=self.data_ingestion_config.test_dir,
            )
            log_trn.info("Data Ingestion: Finished")
            return di_artf

        except Exception as e:
            LogException(e, logger=log_trn)
            raise CustomException(e)
