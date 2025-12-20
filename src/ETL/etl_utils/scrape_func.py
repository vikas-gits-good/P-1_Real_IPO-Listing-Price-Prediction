import re
import pandas as pd
from typing import Literal
from crawl4ai.async_webcrawler import AsyncWebCrawler

from src.ETL.etl_config.ETL_config import ScrapeConfig
from src.Logging.logger import log_etl
from src.Exception.exception import LogException, CustomException


class IPOScraper:
    def __init__(
        self,
        urls: list[str],
        purpose: Literal["detail", "review", "sbscrp", "grmkpt"],
        config: ScrapeConfig = ScrapeConfig(),
    ):
        try:
            self.urls = urls
            self.purpose = purpose
            self.config = config
            self.run_config = {
                "detail": self.config.rncf_ipo_details,
                "review": self.config.rncf_ipo_reviews,
                "sbscrp": self.config.rncf_ipo_sbscrps,
                "grmkpt": self.config.rncf_ipo_grmkpts,
            }.get(self.purpose, self.config.rncf_ipo_details)

        except Exception as e:
            LogException(e, logger=log_etl)
            # raise CustomException(e)

    async def scrape(
        self,
    ):
        async with AsyncWebCrawler(config=self.config.browser_config) as crawler:
            results = await crawler.arun_many(
                urls=self.urls,
                config=self.run_config,
                dispatcher=self.config.mem_ada_dispatcher,
            )
            df = pd.DataFrame()
            try:
                df = self._results_to_dataframe(results)

            except Exception as e:
                LogException(e, logger=log_etl)
                # raise CustomException(e)

            return df

    def _results_to_dataframe(self, results) -> pd.DataFrame:
        df = pd.DataFrame()
        for result in results:
            try:
                data_dict = {f"{self.purpose}_url": [result.url]}
                if self.purpose == "detail":
                    for table in result.tables:
                        data_dict.update(table["data"])
                    df = pd.concat(
                        [df, pd.DataFrame(data_dict)], axis=0, ignore_index=True
                    )

                elif self.purpose == "review":
                    for table in result.tables:
                        reviewer = table["data"]["Review By"][0]
                        dict_1 = {
                            f"{reviewer} {key}": table["data"][key]
                            for key in table["data"]
                            if key != "Review By"
                        }
                        data_dict.update(dict_1)
                    df = pd.concat(
                        [df, pd.DataFrame(data_dict)], axis=0, ignore_index=True
                    )

                elif self.purpose == "sbscrp":
                    for table in result.tables:
                        days = [
                            # re.match(r"(Day \d+)", item).group(1)
                            re.match(r"(?i).*?(Day \d+).*?", item).group(1)
                            for item in table["data"]["Date"]
                        ]

                        for i, day in enumerate(days):
                            day_dict = {}
                            for key in table["data"]:
                                if key == "Date":
                                    full_date = table["data"]["Date"][i]
                                    date_only = re.sub(
                                        r"(Day \d+)", "", full_date
                                    ).strip()
                                    day_dict[f"{day}_Date"] = [date_only]
                                else:
                                    day_dict[f"{day}_{key}"] = [table["data"][key][i]]
                            data_dict.update(day_dict)
                    df = pd.concat(
                        [df, pd.DataFrame(data_dict)], axis=0, ignore_index=True
                    )

                elif self.purpose == "grmkpt":
                    for table in result.tables:
                        dates = [
                            re.match(r"(\d{1,2}-\d{1,2}-\d{4})", item).group(1)
                            for item in table["data"]["GMP Date"]
                        ]
                        for i, date in enumerate(dates):
                            date_dict = {}
                            for key in table["data"]:
                                if key == "GMP Date":
                                    date_dict[f"Day_{i + 1}_Date"] = [date]
                                else:
                                    date_dict[f"Day_{i + 1}_{key}"] = [
                                        table["data"][key][i]
                                    ]
                            data_dict.update(date_dict)
                    df = pd.concat(
                        [df, pd.DataFrame(data_dict)], axis=0, ignore_index=True
                    )

            except Exception as e:
                log_etl.info(f"Error while processing '{result.url}'")
                LogException(e, logger=log_etl)
                continue
                # raise CustomException(e)

        return df
