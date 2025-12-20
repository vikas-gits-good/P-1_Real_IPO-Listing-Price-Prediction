import numpy as np
import pandas as pd
from typing import List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.impute import KNNImputer


from src.Logging.logger import log_trn
from src.Exception.exception import CustomException, LogException
from src.Constants import common_constants, data_transformation
from src.Entity.config_entity import DataTransformationConfig
from src.Entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.Utils.main_utils import (
    read_dataframe,
    save_numpy_array,
    save_transformation_object,
)


class DataTransformation:
    def __init__(
        self,
        artifact: DataValidationArtifact = None,
        data_transformation_config: DataTransformationConfig = DataTransformationConfig(),
    ):
        try:
            self.data_validation_artifact = artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            log_trn.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)

    def create_gmp_columns(self, data: pd.DataFrame = None) -> None:
        try:
            df = data.copy()
            strn_cols = [col for col in df.columns if df[col].dtypes == "O"]
            manu_cols = [
                "IPO_listing_price",
                "IPO_listing_gain_percentage",
                "IPO_day1_qib",
                "IPO_day1_nii",
                "IPO_day1_rtl",
                "IPO_day2_rtl",
            ]
            date_cols = [
                col
                for col in df.columns
                if pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
                .notna()
                .any()
            ]
            days_cols = [col for col in df.columns if col.startswith("day")]
            subs_cols = [col for col in df.columns if col.startswith("IPO_day3")]
            rcmd_cols = [
                col
                for col in df.columns
                if col.endswith("_neutral") or col.endswith("_avoid")
            ]

            df["IPO_open_date"] = pd.to_datetime(df["IPO_open_date"], errors="coerce")
            df["IPO_close_date"] = pd.to_datetime(df["IPO_close_date"], errors="coerce")

            # create a bunch of new columns where only gmp_prices until 1 day b4 closing day is available
            for i in range(1, 34):
                date_col = f"day{i}_date"
                price_col = f"day{i}_price"
                gmp_col = f"gmp_day{i}"
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[gmp_col] = df.apply(
                    lambda row: row[price_col]
                    if pd.notna(row[date_col])
                    and (
                        row[date_col] <= (row["IPO_close_date"] - pd.Timedelta(days=1))
                    )
                    else np.nan,
                    axis=1,
                )

            # create a new feature which is the sum of rate of change in gmp each day
            def analyze_row_gmp(row: pd.Series):
                vals = row.dropna().values
                if len(vals) < 2:
                    return np.nan, np.nan
                pct_changes = [
                    (vals[i] - vals[i - 1]) / vals[i - 1] * 100
                    for i in range(1, len(vals))
                ]
                pct_change_sum = sum(pct_changes)
                return pct_change_sum, vals[-1]

            gmp_cols = [f"gmp_day{i}" for i in range(1, 34)]
            results = df[gmp_cols].apply(analyze_row_gmp, axis=1, result_type="expand")
            results.columns = ["gmp_sum_roc", "gmp_pu_day"]
            df = df.join(results)
            drop_cols = set(
                strn_cols
                + date_cols
                + gmp_cols
                + days_cols
                + subs_cols
                + rcmd_cols
                + manu_cols
            )
            items_to_remove = ["IPO_day2_qib", "IPO_day2_nii"]
            drop_cols = [item for item in drop_cols if item not in items_to_remove]
            df.drop(columns=drop_cols, inplace=True)
            df["IPO_issue_size"] = df["IPO_issue_size"].replace(0, np.nan)
            return df

        except Exception as e:
            log_trn.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)

    def create_pipeline_object(self, columns: List[str]) -> Pipeline:
        try:
            ppln_prpc = Pipeline(
                [
                    ("Preprocessing", FunctionTransformer(self.create_gmp_columns)),
                    (
                        "Imputing",
                        ColumnTransformer(
                            [
                                (
                                    "KNNImputer",
                                    KNNImputer(
                                        **data_transformation.TRANSFORMATION_IMPUTER_PARAMS
                                    ),
                                    columns,
                                ),
                            ],
                            verbose_feature_names_out=False,
                            remainder="passthrough",
                            n_jobs=-1,
                        ),
                    ),
                    (
                        "Transforming",
                        PowerTransformer(method="yeo-johnson", standardize=True),
                    ),
                ]
            )  # .set_output(transform="pandas")
            return ppln_prpc

        except Exception as e:
            log_trn.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)

    def initialise(self) -> DataTransformationArtifact:
        try:
            log_trn.info("Data Transformation: Started")
            log_trn.info("Data Transformation: Getting validated data from file")
            df_train, df_vald, df_test = [
                read_dataframe(path)
                for path in [
                    self.data_validation_artifact.valid_train_file_path,
                    self.data_validation_artifact.valid_vald_file_path,
                    self.data_validation_artifact.valid_test_file_path,
                ]
            ]
            (x_train, y_train), (x_vald, y_vald), (x_test, y_test) = [
                (
                    df.drop(columns=[common_constants.TARGET_COLUMN]),
                    df[common_constants.TARGET_COLUMN],
                )
                for df in [df_train, df_vald, df_test]
            ]
            # converting 'Cat_1','Cat_2',...,'Cat_6' to 0,1,...,5
            y_train, y_vald, y_test = [
                series.apply(lambda x: int(x.split("_")[1]) - 1)
                for series in [y_train, y_vald, y_test]
            ]

            log_trn.info("Data Transformation: Creating transformation pipeline object")
            impt_cols = [
                "IPO_face_value",
                "IPO_issue_price",
                "IPO_lot_size",
                "IPO_issue_size",
                "IPO_Broker_apply",
                "IPO_Member_apply",
                "IPO_day2_qib",
                "IPO_day2_nii",
                "gmp_sum_roc",
                "gmp_pu_day",
            ]
            ppln_prpc = self.create_pipeline_object(columns=impt_cols)

            log_trn.info("Data Transformation: Transforming datasets")
            ppln_prpc.fit(x_train)
            x_train_1, x_vald_1, x_test_1 = [
                ppln_prpc.transform(x) for x in [x_train, x_vald, x_test]
            ]
            df_train_trfm, df_vald_trfm, df_test_trfm = [
                np.c_[np.array(X), np.array(Y)]
                for X, Y in [
                    (x_train_1, y_train),
                    (x_vald_1, y_vald),
                    (x_test_1, y_test),
                ]
            ]

            log_trn.info("Data Transformation: Saving fitted pipeline object to file")
            save_transformation_object(
                file_path=self.data_transformation_config.trfm_object_file_path,
                object=ppln_prpc,
            )

            log_trn.info("Data Transformation: Saving transformed datasets to file")
            [
                save_numpy_array(file_path=path, array=data)
                for data, path in [
                    (
                        df_train_trfm,
                        self.data_transformation_config.trfm_train_file_path,
                    ),
                    (df_vald_trfm, self.data_transformation_config.trfm_vald_file_path),
                    (df_test_trfm, self.data_transformation_config.trfm_test_file_path),
                ]
            ]

            dt_artf = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.trfm_object_file_path,
                transformed_train_file_path=self.data_transformation_config.trfm_train_file_path,
                transformed_valid_file_path=self.data_transformation_config.trfm_vald_file_path,
                transformed_test_file_path=self.data_transformation_config.trfm_test_file_path,
            )

            log_trn.info("Data Transformation: Exporting data transformation artifact")
            log_trn.info("Data Transformation: Finished")
            return dt_artf

        except Exception as e:
            log_trn.info(f"Error: {e}")
            LogException(e)
            raise CustomException(e)
