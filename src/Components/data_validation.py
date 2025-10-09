import pandas as pd
from scipy.stats import ks_2samp

from src.Logging.logger_train import logging
from src.Exception.exception import CustomException
from src.Constants import data_validation
from src.Entity.config_entity import DataValidationConfig
from src.Entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.Utils.main_utils import (
    read_dataframe,
    read_yaml_file,
    write_yaml_file,
    save_dataframe,
)


class DataValidation:
    def __init__(
        self,
        artifact: DataIngestionArtifact = None,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ingestion_artifact = artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(data_validation.SCHEMA_FILE_PATH)

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def validate_num_of_cols(self, data: pd.DataFrame = None) -> bool:
        try:
            if len(data.columns) == len(self._schema_config["columns"]):
                return True
            else:
                logging.info(
                    "Data Validation: Number of columns between dataset and schema are mismatched."
                )
                return False

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def validate_dtype_of_cols(self, data: pd.DataFrame = None) -> bool:
        try:
            data_numr_cols = [col for col in data.columns if data[col].dtypes != "O"]
            schema_numr_cols = self._schema_config["numerical_columns"]
            missing_cols = set(schema_numr_cols) - set(data_numr_cols)
            if missing_cols:
                logging.info(
                    f"Data Validation: These columns are missing from dataset. {missing_cols}"
                )
                return False
            return True

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def detect_data_drift(
        self,
        base_df: pd.DataFrame = None,
        current_df: pd.DataFrame = None,
        report_path: str = None,
        threshold: float = 0.05,
    ) -> bool:
        try:
            status = True
            report = {}
            numr_cols = [col for col in base_df.columns if base_df[col].dtypes != "O"]
            for col in numr_cols:
                d1 = base_df[col]
                d2 = current_df[col]
                # Skip ks_2samp if any NaNs in either column
                if d1.hasnans or d2.hasnans:
                    report[col] = {
                        "p_value": None,
                        "drift_status": None,
                        "note": "Skipped due to NaN values",
                    }
                    continue

                sampl_test = ks_2samp(d1, d2)
                if threshold <= sampl_test.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        col: {
                            "p_value": float(sampl_test.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )

            logging.info("Data Validation: Writing data drift report to file")
            write_yaml_file(
                file_path=report_path,
                content=report,
                replace=True,
            )
            return status

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def initialise(self) -> DataValidationArtifact:
        try:
            logging.info("Data Validation: Started")
            logging.info("Data Validation: Getting ingested data from file")
            df_train, df_vald, df_test = [
                read_dataframe(path)
                for path in [
                    self.data_ingestion_artifact.train_file_path,
                    self.data_ingestion_artifact.vald_file_path,
                    self.data_ingestion_artifact.test_file_path,
                ]
            ]

            logging.info("Data Validation: Running validation scheme")
            (train_num, train_dtype), (vald_num, vald_dtype), (test_num, test_dtype) = [
                (
                    self.validate_num_of_cols(data=df),
                    self.validate_dtype_of_cols(data=df),
                )
                for df in [df_train, df_vald, df_test]
            ]

            logging.info("Data Validation: Checking data drift")
            vald_drift, test_drift = [
                self.detect_data_drift(
                    base_df=df_train, current_df=df, report_path=path, threshold=0.05
                )
                for (df, path) in [
                    (df_vald, self.data_validation_config.drift_report_path_vald),
                    (df_test, self.data_validation_config.drift_report_path_test),
                ]
            ]
            final_status = [
                True
                if all(
                    [
                        train_num,
                        train_dtype,
                        vald_num,
                        vald_dtype,
                        test_num,
                        test_dtype,
                        vald_drift,
                        test_drift,
                    ]
                )
                else False
            ][0]

            dv_artf = DataValidationArtifact(
                validation_status=final_status,
                valid_train_file_path=self.data_validation_config.valid_train_data_path,
                valid_vald_file_path=self.data_validation_config.valid_vald_data_path,
                valid_test_file_path=self.data_validation_config.valid_test_data_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_data_path,
                invalid_vald_file_path=self.data_validation_config.invalid_vald_data_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_data_path,
                drift_report_file_path_vald=self.data_validation_config.drift_report_path_vald,
                drift_report_file_path_test=self.data_validation_config.drift_report_path_test,
            )

            if not final_status:
                logging.info("Data Validation: Split data is INVALID. Please recheck")
                logging.info("Data Validation: Saving INVALID data to file")
                for data, path in (
                    (df_train, self.data_validation_config.invalid_train_data_path),
                    (df_vald, self.data_validation_config.invalid_vald_data_path),
                    (df_test, self.data_validation_config.invalid_test_data_path),
                ):
                    save_dataframe(data=data, path=path)

            else:
                logging.info("Data Validation: Saving validated data to file")
                for data, path in (
                    (df_train, self.data_validation_config.valid_train_data_path),
                    (df_vald, self.data_validation_config.valid_vald_data_path),
                    (df_test, self.data_validation_config.valid_test_data_path),
                ):
                    save_dataframe(data=data, path=path)

            logging.info("Data Validation: Exporting data validation artifact")
            logging.info("Data Validation: Finished")
            return dv_artf

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
