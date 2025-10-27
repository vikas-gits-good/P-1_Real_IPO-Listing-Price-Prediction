import pandas as pd
import numpy as np

from src.ETL.etl_config import etl_constants
from src.Utils.main_utils import read_dataframe
from src.Logging.logger import log_etl
from src.Exception.exception import CustomException, LogException


class DataTransformer:
    def __init__(self, data: pd.DataFrame = None, drop_sme: bool = True):
        try:
            self.data = data
            self.drop_sme = drop_sme

        except Exception as e:
            LogException(e, "Transformation")
            raise CustomException(e)

    def reorder_dataframe(self, Data: pd.DataFrame = None) -> pd.DataFrame:
        try:
            df = Data.copy()
            df_req = read_dataframe(
                path=etl_constants.INITIAL_DATA_DIR, log_name=log_etl
            )

            # reorder columns
            cols_ordr = etl_constants.EXTRACTED_DATA_COLS_ORDER
            df = df.reindex(columns=cols_ordr)

            # rename columns
            repl_dict = {
                key: val
                for key, val in zip(df.columns.to_list(), df_req.columns.to_list())
            }
            # log_etl.info(f"{repl_dict}\n")
            df.rename(columns=repl_dict, inplace=True, errors="ignore")
            return df

        except Exception as e:
            LogException(e, "Transformation")
            # return Data
            raise CustomException(e)

    def create_missing_columns(self, Data: pd.DataFrame = None) -> pd.DataFrame:
        try:
            df = Data.copy()

            def calc_name(row) -> str:
                return row["company_long_name"]

            def calc_perc(row) -> float:
                return round(
                    (row["IPO_listing_price"] - row["IPO_issue_price"])
                    / row["IPO_listing_price"]
                    * 100,
                    2,
                )

            def calc_catg(row) -> str:
                try:
                    if pd.isna(row["IPO_listing_gain_percentage"]):
                        return np.nan
                    else:
                        return next(
                            cat
                            for cat, (low, high) in {
                                "Cat_1": (-100, -20),
                                "Cat_2": (-20, 0),
                                "Cat_3": (0, 10),
                                "Cat_4": (10, 20),
                                "Cat_5": (20, 40),
                                "Cat_6": (40, 100),
                            }.items()
                            if low <= row["IPO_listing_gain_percentage"] <= high
                        )
                except Exception as e:
                    LogException(e)
                    return np.nan
                    # raise CustomException(e)

            df["gmp_company_name"] = df.apply(calc_name, axis=1)
            df["IPO_listing_gain_percentage"] = df.apply(calc_perc, axis=1)
            df["IPO_listing_gain_category"] = df.apply(calc_catg, axis=1)

            return df

        except Exception as e:
            LogException(e, "Transformation")
            # return Data
            raise CustomException(e)

    def update_subscription(self, Data: pd.DataFrame = None) -> pd.DataFrame:
        try:
            df = Data.copy()
            subsc_cols = [col for col in df.columns if "IPO_day" in col]
            subsc_len = int(len(subsc_cols) / 3)
            days = list(range(1, subsc_len + 1))
            metrics = ["qib", "nii", "rtl"]

            # Prepare container for new df of day1, day2, day3 columns
            new_data = {f"IPO_day{d}_{m}": [] for d in range(1, 4) for m in metrics}

            for _, row in df.iterrows():
                metric_values = {
                    metric: [row[f"IPO_day{day}_{metric}"] for day in days]
                    for metric in metrics
                }

                # Day 1 data remains as is
                for metric in metrics:
                    new_data[f"IPO_day1_{metric}"].append(row[f"IPO_day1_{metric}"])

                # For day 2 and day 3 replacements
                for metric in metrics:
                    values_from_day2 = metric_values[metric][1:]  # from day2 onwards
                    non_nan_vals = [v for v in values_from_day2 if not pd.isna(v)]

                    if len(non_nan_vals) >= 2:
                        new_data[f"IPO_day2_{metric}"].append(
                            non_nan_vals[-2]
                        )  # 2nd last non NaN
                        new_data[f"IPO_day3_{metric}"].append(
                            non_nan_vals[-1]
                        )  # last non NaN
                    elif len(non_nan_vals) == 1:
                        new_data[f"IPO_day2_{metric}"].append(non_nan_vals[0])
                        new_data[f"IPO_day3_{metric}"].append(non_nan_vals[0])
                    else:
                        new_data[f"IPO_day2_{metric}"].append(np.nan)
                        new_data[f"IPO_day3_{metric}"].append(np.nan)

            # Create new dataframe with only day1, day2, day3 columns
            new_cols = [f"IPO_day{d}_{m}" for d in range(1, 4) for m in metrics]
            new_df = pd.DataFrame({col: new_data[col] for col in new_cols})

            df.drop(columns=subsc_cols, inplace=True)
            df = pd.concat([df, new_df], axis=1, ignore_index=False)
            return df

        except Exception as e:
            LogException(e, "Transformation")
            # return Data
            raise CustomException(e)

    def reorder_gmp(self, Data: pd.DataFrame = None) -> pd.DataFrame:
        try:
            df = Data.copy()

            def gmp_day1_zero_shift(row):
                price_cols = df.filter(regex=r"^day\d+_price$").columns.to_list()
                date_cols = df.filter(regex=r"^day\d+_date$").columns.to_list()
                # Find first non-zero price index
                prices = row[price_cols].values
                nonzero_indices = np.where(prices != 0)[0]
                if len(nonzero_indices) == 0:
                    return np.nan  # All zero, just return NaN
                first_idx = nonzero_indices[0]
                # Slice date and price arrays from first_idx to end
                shifted_prices = prices[first_idx:]
                shifted_dates = row[date_cols].values[first_idx:]
                # Fill from day1 with sliced data & Pad the rest with np.nan to keep length same
                new_prices = np.concatenate(
                    [shifted_prices, np.full(first_idx, np.nan)]
                )
                new_dates = np.concatenate([shifted_dates, np.full(first_idx, np.nan)])
                # Assign back the shifted and padded arrays
                for i, col in enumerate(price_cols):
                    row[col] = new_prices[i]
                for i, col in enumerate(date_cols):
                    row[col] = new_dates[i]
                return row

            df = df.apply(gmp_day1_zero_shift, axis=1)
            # log_etl.info(f"{df.columns}\n")
            return df

        except Exception as e:
            LogException(e, "Transformation")
            # return Data
            raise CustomException(e)

    def transform(self) -> pd.DataFrame:
        try:
            log_etl.info("Transformation: Updating IPO subscriptions")
            df_subs = self.update_subscription(Data=self.data)

            log_etl.info("Transformation: Handling absent columns")
            df_miss = self.create_missing_columns(Data=df_subs)

            log_etl.info("Transformation: Reordering and renaming columns")
            df_rodr = self.reorder_dataframe(Data=df_miss)

            log_etl.info("Transformation: Checking GMP values")
            df_gmp = self.reorder_gmp(Data=df_rodr)

            if self.drop_sme:
                log_etl.info("Transformation: Dropping SME data")
                df_gmp = df_gmp.loc[df_gmp["Security_type"] != "SME", :]

            return df_gmp

        except Exception as e:
            LogException(e, "Transformation")
            # return Data
            raise CustomException(e)
