import re
import json
import time
import requests
import numpy as np
import pandas as pd
from lxml import html
from pyotp import TOTP
from bs4 import BeautifulSoup
from datetime import datetime
from pymongo import MongoClient
from SmartApi import SmartConnect
from dateutil.parser import parse
from typing import List, Literal, Tuple

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy


from src.Logging.logger import log_etl
from src.Exception.exception import CustomException, LogException
from src.Entity.config_entity import AngelOneConfig
from src.ETL.etl_config.ETL_config import (
    GMPCrawlerConfig,
    ScreenerCSSCrawlerConfig,
    ScreenerHTMLCrawlerConfig,
    BSECrawlerConfig,
)
from src.Entity.config_entity import MongoDBConfig
from src.Utils.main_utils import get_df_from_MongoDB


class CompanyListExtractor:
    def __init__(
        self,
        start_year: int = 2012,
        end_year: int = datetime.now().year,
        headless: bool = True,
    ):
        try:
            log_etl.info("Extraction: Initializing Web Scraping")
            self.start_year = start_year
            self.end_year = end_year
            self.browser_config = BrowserConfig(
                browser_type="chromium", headless=headless, verbose=False
            )
            self.crawler_config = CrawlerRunConfig(
                extraction_strategy=JsonCssExtractionStrategy(
                    schema={
                        "name": "Company Info",
                        "baseSelector": "div > table > tbody > tr:not(:has(td:first-child div a > del))",  # "div > table > tbody > tr",
                        "fields": [
                            {
                                "name": "Company_name",
                                "selector": "a",
                                "type": "text",
                            },
                            {
                                "name": "Company_info_url",
                                "selector": "a",
                                "type": "attribute",
                                "attribute": "href",
                            },
                        ],
                    }
                )
            )

        except Exception as e:
            raise CustomException(e)

    async def extract(self):
        try:
            log_etl.info("Extraction: Extracting company names")
            url_list = [
                f"https://www.chittorgarh.com/report/ipo-in-india-list-main-board-sme/82/{category}/?year={yr}"
                for category in ("mainboard", "sme")
                for yr in range(self.start_year, self.end_year + 1)
            ]
            df = pd.DataFrame()

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                for url in url_list:
                    result = await crawler.arun(url=url, config=self.crawler_config)
                    if result and result.extracted_content:
                        products = json.loads(result.extracted_content)
                        df_temp = pd.DataFrame(products)
                        df_temp["IPO_category"] = (
                            "SME" if "sme" in url.split("/")[-2] else "EQ"
                        )
                        df = pd.concat([df, df_temp], ignore_index=True)

            df["Company_name"] = (
                df["Company_name"]
                .str.replace("Corp.", "Corporation ", regex=False)
                .str.replace("Ltd.", "Limited", regex=False)
                .str.replace("Co.", "Company ", regex=False)
                .str.replace(" IPO", "", regex=False)
            )

            return df

        except Exception as e:
            raise CustomException(e)


class IPODataExtractor:
    def __init__(
        self,
        Data: pd.DataFrame = None,
        CrawlerConfig: GMPCrawlerConfig = None,
    ):
        self.urls = Data["Company_info_url"].to_list()
        self.data = Data
        CrawlerConfig = (
            GMPCrawlerConfig(max_parallel=10, len_list=len(self.urls))
            if CrawlerConfig is None
            else CrawlerConfig
        )
        self.browser_config = CrawlerConfig.browser_config
        self.crawler_run_config = CrawlerConfig.crawler_run_config
        self.rate_limiter = CrawlerConfig.rate_limiter
        self.memory_adap_dispatcher = CrawlerConfig.memory_adap_dispatcher

    def safe_calc(self, func):
        try:
            return func()
        except Exception as e:
            LogException(e)
            return "error"

    def get_ipo_detail(self, results):
        df_ipo = pd.DataFrame()
        for result in results:
            details = {
                "result_link": [result.url],
                "IPO_open_date": ["error"],
                "IPO_close_date": ["error"],
                "IPO_list_date": ["error"],
                "IPO_face_value": ["error"],
                "IPO_issue_price": ["error"],
                "IPO_lot_size": ["error"],
                "IPO_issue_size": ["error"],
            }
            soup = BeautifulSoup(result.cleaned_html, "html.parser")
            tables = soup.find_all("table")
            target_table = None
            # print(tables)

            for table in tables:
                header_cells = table.find_all(["th", "td"])
                if any(
                    (cell.get_text(strip=True) == "IPO Date") for cell in header_cells
                ):
                    target_table = table
                    break

            if not target_table:
                log_etl.info(f"Extraction: IPO details table not found in {result.url}")
                df_ipo = pd.concat(
                    [df_ipo, pd.DataFrame(details)], axis=0, ignore_index=True
                )
                continue
            # print(target_table)

            rows = target_table.find_all("tr")
            cols = [row.find_all("td") for row in rows]

            details["IPO_open_date"] = [
                self.safe_calc(
                    lambda: datetime.strptime(
                        cols[0][-1].get_text(strip=True).split(" to ")[0],
                        "%B %d, %Y",
                    ).strftime("%Y-%m-%d")
                )
            ]

            details["IPO_close_date"] = [
                self.safe_calc(
                    lambda: datetime.strptime(
                        cols[0][-1].get_text(strip=True).split(" to ")[1],
                        "%B %d, %Y",
                    ).strftime("%Y-%m-%d")
                )
            ]

            def extract_ipo_list_date(cols):
                date = "error"
                try:
                    date_string = cols[1][-1].get_text(strip=True)
                    date = datetime.strptime(date_string, "%B %d, %Y").strftime(
                        "%Y-%m-%d"
                    )
                except Exception as e:
                    log_etl.info(
                        f"Error from {result.url}: {e}. Using tentative listing date."
                    )
                    LogException(e)
                    for table in tables:
                        header_cells = table.find_all(["th", "td"])
                        if any(
                            (cell.get_text(strip=True) == "Tentative Listing Date")
                            for cell in header_cells
                        ):
                            target_table = table
                            break
                    try:
                        rows = target_table.find_all("tr")
                        for row in rows:
                            cols_inner = row.find_all("td")
                            if (
                                cols_inner
                                and cols_inner[0].get_text(strip=True)
                                == "Tentative Listing Date"
                            ):
                                date_str = cols_inner[1].get_text(strip=True)
                                date = datetime.strptime(
                                    date_str, "%a, %b %d, %Y"
                                ).strftime("%Y-%m-%d")
                    except Exception as e_inner:
                        log_etl.info(f"Error parsing tentative listing date: {e_inner}")
                        LogException(e)
                        date = "error"
                return date

            details["IPO_list_date"] = [
                self.safe_calc(lambda: extract_ipo_list_date(cols))
            ]

            details["IPO_face_value"] = [
                self.safe_calc(
                    lambda: int(
                        re.findall(
                            r"\d{1,3}(?:,\d{2})*(?:,\d{3})(?:\.\d+)?|\d+(?:\.\d+)?",
                            cols[2][-1].get_text(strip=True).replace(",", ""),
                        )[-1]
                    )
                )
            ]
            details["IPO_issue_price"] = [
                self.safe_calc(
                    lambda: int(
                        re.findall(
                            r"\d{1,3}(?:,\d{2})*(?:,\d{3})(?:\.\d+)?|\d+(?:\.\d+)?",
                            cols[3][-1].get_text(strip=True).replace(",", ""),
                        )[-1]
                    )
                )
            ]

            def extract_lot_size(cols):
                regex = r"\d{1,3}(?:,\d{2})*(?:,\d{3})(?:\.\d+)?|\d+(?:\.\d+)?"
                data = (
                    cols[5][-1].get_text(strip=True).replace(",", "")
                    if "per share" in cols[4][-1].get_text(strip=True).replace(",", "")
                    else cols[4][-1].get_text(strip=True).replace(",", "")
                )
                size = int(re.findall(regex, data)[-1])
                return size

            details["IPO_lot_size"] = [self.safe_calc(lambda: extract_lot_size(cols))]

            def extract_issue_size(cols):
                regex_pattern = r"\d{1,3}(?:,\d{2})*(?:,\d{3})(?:\.\d+)?|\d+(?:\.\d+)?"
                text6 = cols[6][-1].get_text(strip=True)
                matches6 = re.findall(regex_pattern, text6)
                if matches6:
                    return float(matches6[-1].replace(",", "")) * 10**7
                else:
                    text7 = cols[7][-1].get_text(strip=True)
                    matches7 = re.findall(regex_pattern, text7)
                    return float(matches7[-1].replace(",", "")) * 10**7

            details["IPO_issue_size"] = [
                self.safe_calc(lambda: extract_issue_size(cols))
            ]

            df_ipo = pd.concat(
                [df_ipo, pd.DataFrame(details)], axis=0, ignore_index=True
            )
        return df_ipo

    def get_rhp_doc(self, urls: list[str] = None):
        try:
            df_rhp = pd.DataFrame(
                {"rhp_link": [link for link in urls]},
            )
            # This function is supposed to process pdf and extract some company financial
            # and risk factor data but I'll do that later
            return df_rhp
        except Exception as e:
            LogException(e)
            return df_rhp
            # raise CustomException(e)

    async def get_gmp_link(self, urls: list[str]):
        try:
            data = pd.DataFrame(
                {
                    **{"gmp_link": [link for link in urls]},
                    **{
                        f"day{i}_{m}": [np.nan] * len(urls)
                        for i in range(1, 4)
                        for m in ["date", "price"]
                    },
                }
            )
            df_gmp = await GMPDataExtractor(reorder_dates=True).extract(urls=urls)
            return df_gmp

        except Exception as e:
            LogException(e)
            return data
            # raise CustomException(e)

    async def get_ipo_review(self, urls: list[str] = None):
        try:
            df_rvw = pd.DataFrame()
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                results = await crawler.arun_many(
                    urls=urls,
                    config=self.crawler_run_config,
                    dispatcher=self.memory_adap_dispatcher,
                )

                for result in results:
                    try:
                        review = {
                            **{"review_link": [result.url]},
                            **{
                                f"IPO_{i}_{m}": ["error"]
                                for i in ["Broker", "Member"]
                                for m in ["apply", "neutral", "avoid"]
                            },
                        }
                        # reviews = {key: val * len(urls) for key, val in review.items()}
                        soup = BeautifulSoup(result.cleaned_html, "html.parser")
                        tables = soup.find_all("table")
                        matching_tables = []

                        for table in tables:
                            header_cells = table.find_all(["th", "td"])
                            if any(
                                cell.get_text(strip=True) == "Review By"
                                for cell in header_cells
                            ):
                                matching_tables.append(table)

                        if not matching_tables:
                            log_etl.info(
                                f"Extraction: Review table not found in {result.url}"
                            )
                            df_rvw = pd.concat(
                                [df_rvw, pd.DataFrame(review)],
                                axis=0,
                                ignore_index=True,
                            )
                            continue
                        # print(matching_tables)

                        def get_reviews(table, prefix: str):
                            rows = table.find_all("tr")
                            cols = rows[1].find_all("td")
                            rvw = {}

                            rvw[f"IPO_{prefix}_apply"] = [
                                self.safe_calc(
                                    lambda: int(
                                        cols[1].get_text(strip=True).replace(",", "")
                                    )
                                    + int(cols[2].get_text(strip=True).replace(",", ""))
                                )
                            ]
                            rvw[f"IPO_{prefix}_neutral"] = [
                                self.safe_calc(
                                    lambda: int(
                                        cols[3].get_text(strip=True).replace(",", "")
                                    )
                                )
                            ]
                            rvw[f"IPO_{prefix}_avoid"] = [
                                self.safe_calc(
                                    lambda: int(
                                        cols[4].get_text(strip=True).replace(",", "")
                                    )
                                )
                            ]
                            return rvw

                        _ = [
                            review.update(
                                get_reviews(table=matching_tables[i], prefix=prf)
                            )
                            for i, prf in enumerate(["Broker", "Member"])
                        ]
                        df_rvw = pd.concat(
                            [df_rvw, pd.DataFrame(review)], axis=0, ignore_index=True
                        )
                        # print(reviews)
                    except Exception as e:
                        log_etl.info(
                            f"Error in get_ipo_review() from {result.url}. Error: {e}"
                        )
                        df_rvw = pd.concat(
                            [df_rvw, pd.DataFrame(review)], axis=0, ignore_index=True
                        )
                        # raise CustomException(e)
            return df_rvw

        except Exception as e:
            LogException(e)
            return df_rvw
            # raise CustomException(e)

    async def get_ipo_subscription(self, urls):
        try:
            df_subsc = pd.DataFrame()
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                results = await crawler.arun_many(
                    urls=urls,
                    config=self.crawler_run_config,
                    dispatcher=self.memory_adap_dispatcher,
                )

                for result in results:
                    try:
                        subsc = {
                            **{"subsc_link": [result.url]},
                            **{
                                f"IPO_day{i}_{m}": ["error"]
                                for i in range(1, 4)
                                for m in ["qib", "nii", "rtl"]
                            },
                        }
                        soup = BeautifulSoup(result.cleaned_html, "html.parser")
                        tables = soup.find_all("table")
                        target_table = None
                        # print(tables)

                        for table in tables:
                            header_cells = table.find_all(["th", "td"])
                            if any(
                                (cell.get_text(strip=True) == "Date")
                                for cell in header_cells
                            ):
                                target_table = table
                                break

                        if not target_table:
                            log_etl.info(
                                f"Extraction: Subscription table not found in {result.url}"
                            )
                            df_subsc = pd.concat(
                                [df_subsc, pd.DataFrame(subsc)],
                                axis=0,
                                ignore_index=True,
                            )
                            continue
                        # print(target_table)

                        rows = target_table.find_all("tr")
                        header_cells = rows[0].find_all(["th", "td"])
                        header_indices = {}
                        for idx, cell in enumerate(header_cells):
                            text = cell.get_text(strip=True)
                            if text in [
                                "QIB",
                                "NII",
                                "Retail",
                                "QIB (Ex Anchor)",
                                "NII*",
                                "Individual Investors",
                            ]:
                                header_indices[text] = idx

                        for i, row in enumerate(rows[1:], start=1):
                            cols = row.find_all("td")
                            idx_qib = header_indices.get(
                                "QIB", header_indices.get("QIB (Ex Anchor)")
                            )
                            idx_nii = header_indices.get(
                                "NII", header_indices.get("NII*")
                            )
                            idx_rtl = header_indices.get(
                                "Retail", header_indices.get("Individual Investors")
                            )

                            subsc[f"IPO_day{i}_qib"] = [
                                self.safe_calc(
                                    lambda: float(
                                        cols[idx_qib]
                                        .get_text(strip=True)
                                        .replace(",", "")
                                    )
                                )
                            ]
                            subsc[f"IPO_day{i}_nii"] = [
                                self.safe_calc(
                                    lambda: float(
                                        cols[idx_nii]
                                        .get_text(strip=True)
                                        .replace(",", "")
                                    )
                                )
                            ]
                            subsc[f"IPO_day{i}_rtl"] = [
                                self.safe_calc(
                                    lambda: float(
                                        cols[idx_rtl]
                                        .get_text(strip=True)
                                        .replace(",", "")
                                    )
                                )
                            ]
                        df_subsc = pd.concat(
                            [df_subsc, pd.DataFrame(subsc)], axis=0, ignore_index=True
                        )
                        # print(subsc)
                    except Exception as e:
                        log_etl.info(
                            f"Error in get_ipo_subscription() from {result.url}. Error: {e}"
                        )
                        df_subsc = pd.concat(
                            [df_subsc, pd.DataFrame(subsc)], axis=0, ignore_index=True
                        )
                        # raise CustomException(e)
            # print(df_subsc)
            return df_subsc

        except Exception as e:
            LogException(e)
            return df_subsc
            # raise CustomException(e)

    async def extract(self):
        df_lnk = self.data.copy()

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            results = await crawler.arun_many(
                self.urls,
                config=self.crawler_run_config,
                dispatcher=self.memory_adap_dispatcher,
            )

            keys = ["result_link", "rhp_link", "review_link", "subsc_link", "gmp_link"]
            urls_dict = {key: [] for key in keys}

            for result in results:
                # ipo data link
                result_link = result.url

                # RHP document link
                rhp_pattern = r"\* \[!\[RHP External link\]\(.*?\)RHP\]\((.*?)\)|\* \[!\[DRHP External link\]\(.*?\)DRHP\]\((.*?)\)"
                rhp_match = re.search(rhp_pattern, result.markdown)
                rhp_link = (
                    next((group for group in rhp_match.groups() if group), None)
                    if rhp_match
                    else None
                )

                # review link
                review_pattern = (
                    r'\[Review\]\((.*?) "(?:IPO Reviews & Recommendation)"\)'
                )
                review_match = re.search(review_pattern, result.markdown)
                review_link = review_match.group(1) if review_match else None

                # subscription link
                subscription_pattern = (
                    r'\[Subscription\]\((.*?) "(?:IPO Live Subscription)"\)'
                )
                subscription_match = re.search(subscription_pattern, result.markdown)
                subscription_link = (
                    subscription_match.group(1) if subscription_match else None
                )

                # gmp link
                gmp_pattern = r'\[GMP\]\((.*?) "(?:IPO GMP)"\)'
                gmp_match = re.search(gmp_pattern, result.markdown)
                gmp_link = gmp_match.group(1) if gmp_match else None

                # combine and update
                data_links = [
                    result_link,
                    rhp_link,
                    review_link,
                    subscription_link,
                    gmp_link,
                ]
                _ = [
                    urls_dict[key].append(val)
                    for key, val in zip(urls_dict.keys(), data_links)
                ]

            try:
                log_etl.info("Extraction: Scraping IPO details")
                df_ipo = self.get_ipo_detail(results=results)

                log_etl.info("Extraction: Scraping RHP docs")
                df_rhp = self.get_rhp_doc(urls=urls_dict["rhp_link"])

                log_etl.info("Extraction: Scraping reviews")
                df_rvw = await self.get_ipo_review(urls=urls_dict["review_link"])

                log_etl.info("Extraction: Scraping subscriptions")
                df_sub = await self.get_ipo_subscription(urls=urls_dict["subsc_link"])

                log_etl.info("Extraction: Scraping GMP")
                df_gmp = await self.get_gmp_link(urls=urls_dict["gmp_link"])

                # reorder the parallel scraped data to correct format
                # reorder links
                order_map = {link: idx for idx, link in enumerate(self.urls)}
                sort_idx = sorted(
                    range(len(urls_dict["result_link"])),
                    key=lambda i: order_map[urls_dict["result_link"][i]],
                )
                urls_dict = {
                    key: [urls_dict[key][idx] for idx in sort_idx]
                    for key in urls_dict.keys()
                }
                # print(urls_dict)

                # reorder function
                def reorder_df(df, link_column, link_order):
                    df[link_column] = pd.Categorical(
                        df[link_column], categories=link_order, ordered=True
                    )
                    df_sorted = df.sort_values(by=link_column).reset_index(drop=True)
                    return df_sorted

                # reorder dataframes
                df_list = [df_ipo, df_rhp, df_rvw, df_sub, df_gmp]
                df_ipo, df_rhp, df_rvw, df_sub, df_gmp = [
                    reorder_df(df, col, urls)
                    for df, col, urls in zip(
                        df_list,
                        keys,
                        list(urls_dict.values()),
                    )
                ]

                # drop link columns that arent needed
                df_ipo, df_rvw, df_sub = [
                    df.drop(columns=[col])
                    for df, col in zip(
                        (df_ipo, df_rvw, df_sub),
                        ("result_link", "review_link", "subsc_link"),
                    )
                ]

                # rename columns as per requirement
                df_rhp, df_gmp = [
                    df.rename(columns=repl)
                    for df, repl in zip(
                        (df_rhp, df_gmp),
                        ({"rhp_link": "IPO_RHP_doc"}, {"gmp_link": "IPO_gmp_link"}),
                    )
                ]
                df_list = [df_lnk, df_ipo, df_rhp, df_rvw, df_sub, df_gmp]
                # print(df_list)
                df_main = pd.concat(
                    df_list,
                    axis=1,
                    ignore_index=False,
                )
                # print(df_main.columns.to_list())
                return df_main

            except Exception as e:
                LogException(e)
                # raise CustomException(e)


class InvestorGainCrawler:
    def __init__(
        self,
        template: pd.DataFrame = None,
        urls: List = None,
        CrawlerConfig: GMPCrawlerConfig = None,
    ):
        CrawlerConfig = (
            GMPCrawlerConfig(max_parallel=10, len_list=len(urls))
            if CrawlerConfig is None
            else CrawlerConfig
        )
        self.urls = urls
        self.df_tmpl = template
        self.browser_config = CrawlerConfig.browser_config
        self.crawler_run_config = CrawlerConfig.crawler_run_config
        self.rate_limiter = CrawlerConfig.rate_limiter
        self.memory_adap_dispatcher = CrawlerConfig.memory_adap_dispatcher

    def clean(self, data: pd.DataFrame = None) -> pd.DataFrame:
        df = data.copy()
        date_cols = [col for col in df.columns if "date" in col]
        price_cols = [col for col in df.columns if "price" in col]
        mask = ~df["day1_date"].apply(lambda x: pd.isna(x))  # df["day1_date"].notna()
        # print(df["day1_date"], mask)
        for col in date_cols:
            df.loc[mask, col] = pd.to_datetime(
                df.loc[mask, col], format="%d-%m-%Y", errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        for col in price_cols:
            df.loc[mask, col] = df.loc[mask, col].apply(
                lambda x: x if pd.isna(x) else int(re.sub(r"[^\d]", "", x))
            )
        # print(df)
        return df

    async def extract_table_from_html(self, html: str, url: str) -> pd.DataFrame:
        try:
            extracted_values = self.df_tmpl.copy()
            extracted_values["gmp_link"] = [url]

            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")
            target_table = None

            for table in tables:
                header_cells = table.find_all(["th", "td"])
                if any(
                    (cell.get_text(strip=True) == "GMP Date") for cell in header_cells
                ):
                    target_table = table
                    break
            if not target_table:
                log_etl.info(f"Extraction: GMP table not found in {url}")
                return extracted_values
            # print(target_table)
            rows = target_table.find_all("tr")
            # print(rows)
            for i, row in enumerate(rows[1:], start=1):
                cols = row.find_all("td")
                if len(cols) >= 2:
                    extracted_values[f"day{i}_date"] = [
                        re.match(
                            r"\d{1,2}-\d{1,2}-\d{4}", cols[0].get_text(strip=True)
                        ).group()
                    ]
                    extracted_values[f"day{i}_price"] = [cols[2].get_text(strip=True)]
            # print(extracted_values)
            return extracted_values
        except Exception as e:
            LogException(e)
            return extracted_values
            # raise CustomException(e)

    async def extract(self) -> pd.DataFrame:
        df_ig = pd.DataFrame()
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            responses = await crawler.arun_many(
                self.urls,
                config=self.crawler_run_config,
                dispatcher=self.memory_adap_dispatcher,
            )
            for result in responses:
                if result.success and result.cleaned_html:
                    df = await self.extract_table_from_html(
                        result.cleaned_html, url=result.url
                    )
                    # print(df)
                    df_ig = pd.concat([df_ig, df], axis=0, ignore_index=True)
                else:
                    backup = self.df_tmpl.copy()
                    backup["gmp_link"] = [result.url]
                    df_ig = pd.concat(
                        [df_ig, backup],
                        axis=0,
                        ignore_index=True,
                    )
        # print(df_ig)
        df_ig = self.clean(data=df_ig)
        # print(df_ig)
        return df_ig


class IPOCentralCrawler:
    def __init__(
        self,
        template: pd.DataFrame = None,
        urls: List = None,
        CrawlerConfig: GMPCrawlerConfig = None,
    ):
        if CrawlerConfig is None:
            CrawlerConfig = GMPCrawlerConfig(max_parallel=10, len_list=len(urls))

        self.urls = urls
        self.df_tmpl = template
        self.browser_config = CrawlerConfig.browser_config
        self.crawler_run_config = CrawlerConfig.crawler_run_config
        self.rate_limiter = CrawlerConfig.rate_limiter
        self.memory_adap_dispatcher = CrawlerConfig.memory_adap_dispatcher

    def clean(self, data: pd.DataFrame = None) -> pd.DataFrame:
        df = data.copy()
        cols_price = [col for col in df.columns if "price" in col]

        def clean_price(val):
            # print(val)
            if pd.isna(val):  # leave NaNs as it is
                return np.nan

            val = str(val).strip()
            if val == "–" or val == "":  # convert '–' and '' to NaN
                return np.nan

            # Extract number after currency like INR, Rs., etc.
            currency_match = re.search(r"(?:INR|Rs\.?)\s*([\d,]+)", val, re.IGNORECASE)
            if currency_match:
                num_str = currency_match.group(1).replace(",", "")
                try:
                    return int(num_str)
                except ValueError:
                    return np.nan

            if " – " in val:  # picking highest number in a range
                # print(val)
                parts = val.split(" – ")
                nums = []
                for p in parts:
                    p = p.replace(",", "").strip()
                    try:
                        nums.append(int(p))
                    except ValueError:
                        continue
                return max(nums) if nums else np.nan

            elif "->" in val:
                parts = val.split("->")
                nums = []
                for p in parts:
                    p = p.replace(",", "").strip()
                    try:
                        nums.append(int(p))
                    except ValueError:
                        continue
                return max(nums) if nums else np.nan

            paren_match = re.search(r"\((\-?\d+)\)", val)  # number inside parentheses
            if paren_match:
                return int(paren_match.group(1))

            elif val.startswith("–"):  # convert negative numbers properly
                num = val.replace("–", "").replace(",", "").strip()
                try:
                    num = -int(num)
                    return num
                except Exception as e:
                    LogException(e)
                    return np.nan

            else:  # remove ',' from 10,000
                val = val.replace(",", "").replace(" ", "")
                try:
                    return int(val)
                except ValueError:
                    return np.nan

        for col in cols_price:
            df[col] = df[col].apply(clean_price)

        return df

    async def extract_table_from_html(self, html: str, url: str) -> pd.DataFrame:
        try:
            extracted_values = self.df_tmpl.copy()
            extracted_values["gmp_link"] = [url]

            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")
            target_table = None

            # Find table with header cell 'Date'
            for table in tables:
                header_cells = table.find_all(["th", "td"])
                if any((cell.get_text(strip=True) == "Date") for cell in header_cells):
                    target_table = table
                    break
            if not target_table:
                log_etl.info(f"Extraction: GMP table not found in {url}")
                return extracted_values
            # print(target_table)
            rows = target_table.find_all("tr")
            # print(rows)
            for i, row in enumerate(rows[1:], start=1):
                cols = row.find_all("td")
                if len(cols) >= 2:
                    extracted_values[f"day{i}_date"] = [
                        parse(cols[0].get_text(strip=True), dayfirst=True)
                        .date()
                        .strftime("%Y-%m-%d")
                    ]
                    extracted_values[f"day{i}_price"] = [cols[1].get_text(strip=True)]
            # print(extracted_values)
            return extracted_values
        except Exception as e:
            LogException(e)
            return extracted_values

    async def extract(self) -> pd.DataFrame:
        df_ic = pd.DataFrame()
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            responses = await crawler.arun_many(
                self.urls,
                config=self.crawler_run_config,
                dispatcher=self.memory_adap_dispatcher,
            )
            for result in responses:
                if result.success and result.cleaned_html:
                    df = await self.extract_table_from_html(
                        result.cleaned_html, url=result.url
                    )
                    # print(df)
                    df_ic = pd.concat([df_ic, df], axis=0, ignore_index=True)
                else:
                    backup = self.df_tmpl.copy()
                    backup["gmp_link"] = [result.url]
                    df_ic = pd.concat(
                        [df_ic, backup],
                        axis=0,
                        ignore_index=True,
                    )
        # print(df_ic)
        df_ic = self.clean(data=df_ic)
        # print(df_ic)
        return df_ic


class GMPDataExtractor:
    def __init__(self, reorder_dates: bool = True):
        self.reorder = reorder_dates

    def reorder_dates(self, data: pd.DataFrame = None) -> pd.DataFrame:
        df = data.copy()
        iters = range(1, int((len(df.columns) - 1) / 2 + 1))
        date_cols = [f"day{i}_date" for i in iters]
        price_cols = [f"day{i}_price" for i in iters]

        for day in date_cols:
            df[day] = pd.to_datetime(df[day], errors="coerce")

        dates_array = df[date_cols].to_numpy()
        prices_array = df[price_cols].to_numpy()

        def sort_dates_prices(dates, prices):
            idx_sorted = np.argsort(dates, axis=1, kind="stable")
            sorted_dates = np.take_along_axis(dates, idx_sorted, axis=1)
            sorted_prices = np.take_along_axis(prices, idx_sorted, axis=1)
            return sorted_dates, sorted_prices

        sorted_dates_array, sorted_prices_array = sort_dates_prices(
            dates_array, prices_array
        )

        df[date_cols] = sorted_dates_array
        df[price_cols] = sorted_prices_array

        return df

    async def extract(self, urls: List = None) -> pd.DataFrame:
        df = pd.DataFrame()
        df_ic = None
        df_ig = None

        url_gmp_ig = [link for link in urls if "investorgain.com" in link]  # [:4]
        url_gmp_ic = [link for link in urls if "ipocentral.in" in link]  # [:4]
        df_tmpl = pd.DataFrame(
            {
                **{"gmp_link": [np.nan]},
                **{
                    f"day{i}_{m}": [np.nan]
                    for i in range(1, 4)
                    for m in ["date", "price"]
                },
            }
        )
        if url_gmp_ig:
            cc_ig = GMPCrawlerConfig(max_parallel=10, len_list=len(url_gmp_ig))
            df_ig = await InvestorGainCrawler(
                template=df_tmpl, urls=url_gmp_ig, CrawlerConfig=cc_ig
            ).extract()
        if url_gmp_ic:
            cc_ic = GMPCrawlerConfig(max_parallel=10, len_list=len(url_gmp_ic))
            df_ic = await IPOCentralCrawler(
                template=df_tmpl, urls=url_gmp_ic, CrawlerConfig=cc_ic
            ).extract()
        df_list = [df] + [
            df for df in [df_ig, df_ic] if df is not None and not df.empty
        ]
        df = pd.concat(df_list, axis=0, ignore_index=True)
        df = self.reorder_dates(data=df) if self.reorder_dates else df
        # print(df)
        return df


class ScreenerExtractor:
    def __init__(
        self,
        Data: pd.DataFrame = None,
        screener_crawl_method: Literal["css", "html"] = "css",
    ):
        self.data = Data.dropna(inplace=False, ignore_index=True, how="all")
        self.company_names = self.data["Company_name"].to_list()
        self.company_names = [s.replace("&", "and") for s in self.company_names]
        self.screener_crawl_method = screener_crawl_method

        CrawlerConfig = GMPCrawlerConfig(
            max_parallel=10, len_list=len(self.company_names)
        )
        self.se_browser_config = CrawlerConfig.browser_config
        self.se_crawler_config = CrawlerConfig.crawler_run_config
        # self.se_rate_limiter = CrawlerConfig.rate_limiter
        self.se_dispatcher = CrawlerConfig.memory_adap_dispatcher

        if self.screener_crawl_method == "css":
            ScrnCrwlConf = ScreenerCSSCrawlerConfig(
                max_parallel=5,  # <- keep it low. Might get ip blocked. Use proxy rotation
                len_list=len(self.company_names),
            )
            self.srcn_browser_config = ScrnCrwlConf.browser_config
            self.srcn_crawler_config = ScrnCrwlConf.run_config
            # self.srcn_rate_limiter = ScrnCrwlConf.rate_limiter
            self.srcn_dispatcher = ScrnCrwlConf.dispatcher

        elif self.screener_crawl_method == "html":
            ScrnCrwlConf = ScreenerHTMLCrawlerConfig(
                max_parallel=5,  # <- keep it low. Might get ip blocked. Use proxy rotation
                len_list=len(self.company_names),
            )
            self.srcn_browser_config = ScrnCrwlConf.browser_config
            self.srcn_crawler_config = ScrnCrwlConf.run_config
            # self.srcn_rate_limiter = ScrnCrwlConf.rate_limiter
            self.srcn_dispatcher = ScrnCrwlConf.dispatcher

        BSECrwlConf = BSECrawlerConfig(
            max_parallel=10,  # <- keep it low. Might get ip blocked. Use proxy rotation
            len_list=len(self.company_names),
        )
        self.bse_browser_config = BSECrwlConf.browser_config
        self.bse_crawler_config = BSECrwlConf.run_config
        # self.bse_rate_limiter = BSECrwlConf.rate_limiter
        self.bse_dispatcher = BSECrwlConf.dispatcher

    def safe_calc(self, func):
        try:
            return func()
        except Exception as e:
            LogException(e)
            return "error"

    async def get_bse_symbol(self, urls: List[str] = None) -> pd.DataFrame:
        df_bse = pd.DataFrame()
        async with AsyncWebCrawler(config=self.bse_browser_config) as crawler:
            results = await crawler.arun_many(
                urls,
                config=self.bse_crawler_config,
                dispatcher=self.bse_dispatcher,
            )
            for result in results:
                try:
                    bse_data = {"bse_link": [result.url], "bse_symbol": [None]}

                    # HTML Extraction strat
                    tree = html.fromstring(result.html)
                    xp = [
                        "/html/body/div[1]/div[4]/div[6]/div[3]/div/div[1]/div[1]/div[1]/div[2]/div/div[2]/text()",
                    ]

                    xpath_result = tree.xpath(xp[0])
                    full_text = xpath_result[0] if xpath_result else None

                    # process string
                    if full_text:
                        text = full_text.strip().strip("()")
                        bse_data["bse_symbol"] = [text.split("|")[0].strip()]
                    else:
                        log_etl.info(
                            f"Extraction: BSE symbol not found at {result.url}"
                        )

                    # log_etl.info(f"{bse_data = }")
                    df_bse = pd.concat(
                        [df_bse, pd.DataFrame(bse_data)],
                        axis=0,
                        ignore_index=True,
                    )

                except Exception as e:
                    log_etl.info(
                        f"Error in get_bse_symbol(). {bse_data = }, Error: {e}"
                    )
                    LogException(e)
                    df_bse = pd.concat(
                        [df_bse, pd.DataFrame(bse_data)], axis=0, ignore_index=True
                    )
                    # raise CustomException(e)
                    continue

            # log_etl.info(f"{df_bse = }")
            return df_bse

    async def get_screener_data(self, urls: list[str] = None) -> pd.DataFrame:
        df_data = pd.DataFrame()
        async with AsyncWebCrawler(config=self.srcn_browser_config) as crawler:
            results = await crawler.arun_many(
                urls,
                config=self.srcn_crawler_config,
                dispatcher=self.srcn_dispatcher,
            )
            for result in results:
                try:
                    data = {
                        "screener_link": [result.url],
                        "company_long_name": [None],
                        "company_website": [None],
                        "bse_link": [None],
                        "bse_symbol": [None],
                        "nse_link": [None],
                        "nse_symbol": [None],
                    }

                    if self.screener_crawl_method == "css":
                        details = json.loads(result.extracted_content)
                        if details[0]:
                            key_list = [
                                "company_long_name",
                                "link_1",
                                "link_2",
                                "link_3",
                            ]
                            name_c, link_1, link_2, link_3 = [
                                details[0].get(key) for key in key_list
                            ]
                            # details[0].get(key) will return data or None.
                            # details[0][key] will return data or raise error

                            # log_etl.info(
                            #     f"{result.url, name_c, link_1, link_2, link_3}"
                            # )

                    elif self.screener_crawl_method == "html":
                        tree = html.fromstring(result.html)
                        xpath_list = [
                            "/html/body/div/div[1]/h1/text()",
                            "/html/body/main/div[3]/div[3]/div[1]/div[2]/a[1]/@href",
                            "/html/body/main/div[3]/div[3]/div[1]/div[2]/a[2]/@href",
                            "/html/body/main/div[3]/div[3]/div[1]/div[2]/a[3]/@href",
                        ]
                        # parse html to get data
                        name_c, link_1, link_2, link_3 = [
                            (result[0] if result else None)
                            for result in (tree.xpath(xp) for xp in xpath_list)
                        ]
                        # log_etl.info(f"{result.url, name_c, link_1, link_2, link_3}")

                    # process extracted data to required format
                    data["company_long_name"] = [
                        self.safe_calc(lambda: name_c) if name_c else None
                    ]
                    data["company_website"] = [
                        self.safe_calc(lambda: link_1)
                        if (link_1 and "www.google.co.in/" not in link_1)
                        else None
                    ]
                    data["bse_link"] = [
                        self.safe_calc(lambda: link_2)
                        if (link_2 and "bseindia.com" in link_2)
                        else None
                    ]
                    data["bse_symbol"] = [
                        self.safe_calc(
                            lambda: (
                                link_2.rstrip("/").split("/")[-1]
                                if "-" in link_2.rstrip("/").split("/")[-2]
                                else link_2.rstrip("/").split("/")[-2]
                            )
                        )
                        if (link_2 and "bseindia.com" in link_2)
                        else None
                    ]
                    data["nse_link"] = [
                        self.safe_calc(lambda: link_3)
                        if (link_3 and "nseindia.com" in link_3)
                        else self.safe_calc(lambda: link_2)
                        if (link_2 and "nseindia.com" in link_2)
                        else None
                    ]
                    data["nse_symbol"] = [
                        self.safe_calc(lambda: link_3.split("?symbol=")[-1])
                        if (link_3 and "nseindia.com" in link_3)
                        else self.safe_calc(lambda: link_2.split("?symbol=")[-1])
                        if (link_2 and "nseindia.com" in link_2)
                        else None
                    ]

                    df_data = pd.concat(
                        [df_data, pd.DataFrame(data)], axis=0, ignore_index=True
                    )

                except Exception as e:
                    log_etl.info(
                        f"Error in get_screener_data() from {result.url}. Data: {data}. Error: {e}"
                    )
                    LogException(e)
                    df_data = pd.concat(
                        [df_data, pd.DataFrame(data)], axis=0, ignore_index=True
                    )
                    continue
                    # raise CustomException(e)

            return df_data

    async def get_screener_links(self, urls: List[str] = None) -> pd.DataFrame:
        df_link = pd.DataFrame()
        async with AsyncWebCrawler(config=self.se_browser_config) as crawler:
            results = await crawler.arun_many(
                urls,
                config=self.se_crawler_config,
                dispatcher=self.se_dispatcher,
            )
            for result in results:
                try:
                    scrn_dict = {
                        "search_url": [result.url],
                        "company_name": [None],
                        "screener_link": [None],
                    }
                    scrn_dict["company_name"] = [
                        self.safe_calc(
                            lambda: re.search(
                                # r"mainpage\+(.+?)",
                                # r"mainpage\+(.+?)&t=",
                                r"mainpage\+(.+?)&[^=]*=",
                                result.url,
                            )
                            .group(1)
                            .replace("+", " ")
                        )
                    ]
                    # log_etl.info(
                    #     f"{result.url}, {scrn_dict['company_name']} , {result.links['external']}"
                    # )
                    scrn_dict["screener_link"] = [  # Dont use safe_calc for this
                        item["href"]
                        for item in result.links["external"]
                        if "www.screener.in/company/" in item["href"]
                    ][0]  # <- extract first item incase multiple are found

                    df_link = pd.concat(
                        [df_link, pd.DataFrame(scrn_dict)],
                        axis=0,
                        ignore_index=True,
                    )

                except Exception as e:
                    log_etl.info(
                        f"Extraction: Failed to get screener link for {scrn_dict['company_name']} @ {result.url}. Error: {e}"
                    )
                    LogException(e)
                    df_link = pd.concat(
                        [df_link, pd.DataFrame(scrn_dict)],
                        axis=0,
                        ignore_index=True,
                    )
                    continue
                    # raise CustomException(e)
        return df_link

    async def while_loop_function(
        self,
        func: callable = None,
        data_column: str = None,
        urls_column: str = None,
        urls: List[str] = None,
        threshold: int = 10,
    ) -> pd.DataFrame:
        counter = 1
        null_urls = int(pd.isnull(urls).sum())
        urls = [x for x in urls if not pd.isnull(x)]
        data = await func(urls)  # <- Dont send list with NaN or None
        while True:
            try:
                null_cnt = data[data_column].isna().sum()
                totl_cnt = data[data_column].shape[0]
                if null_cnt != 0:
                    log_etl.info(
                        f"Extraction: Failed to scrape {null_cnt:03d} of {totl_cnt:03d} sites. Retry attempt: {counter:02d} of {threshold:02d}"
                    )
                    failed_urls = data.loc[
                        data[data_column].isna(), urls_column
                    ].tolist()
                    retry_data = await func(failed_urls)
                    data = data.dropna(subset=[data_column], ignore_index=True)
                    data = pd.concat([data, retry_data], axis=0, ignore_index=True)
                    counter += 1
                    if counter > threshold:
                        null_cnt = data[data_column].isna().sum()
                        log_etl.info(
                            f"Extraction: Retry attempt threshold crossed. Moving on. Failed: {null_cnt:03d}, Total: {totl_cnt:03d}"
                        )
                        break
                elif null_cnt == 0:
                    log_etl.info(
                        f"Extraction: Successfully scraped {totl_cnt:03d} sites."
                    )
                    break
            except Exception as e:
                log_etl.info(f"Error in while_loop_function(). {e}")
                LogException(e)
                raise CustomException(e)

        # returning Null data
        if null_urls != 0:
            null_data = pd.DataFrame(
                data=[[None] * len(data.columns)] * null_urls, columns=data.columns
            )
            data = pd.concat([data, null_data], axis=0, ignore_index=True)

        return data

    async def extract(self):
        try:
            log_etl.info("Extraction: Scraping for screener links")
            search_tmpl = {
                "google.com": "https://www.google.com/search?q=screener.in+mainpage+{company}&pws=0&gl=IN&hl=en",
                "google.co.in": "https://www.google.co.in/search?q=screener.in+mainpage+{company}&pws=0&hl=en",
                "duckduckgo.com": "https://duckduckgo.com/?q=screener.in+mainpage+{company}&t=h_&ia=web",
            }
            search_engine = list(search_tmpl.keys())[1]
            urls_srch = [
                search_tmpl[search_engine].format(company=company.replace(" ", "+"))
                for company in self.company_names
            ]

            df_link = await self.while_loop_function(
                func=self.get_screener_links,
                data_column="screener_link",
                urls_column="search_url",
                urls=urls_srch,
            )

            log_etl.info("Extraction: Scraping for screener data")
            urls_data = df_link["screener_link"].to_list()
            df_data = await self.while_loop_function(
                func=self.get_screener_data,
                data_column="company_long_name",
                urls_column="screener_link",
                urls=urls_data,
            )

            # get bse_symbols
            log_etl.info("Extraction: Scraping for BSE symbols")
            urls_bse = df_data["bse_link"].to_list()  # some items will be None
            df_bse = await self.while_loop_function(
                func=self.get_bse_symbol,
                data_column="bse_symbol",
                urls_column="bse_link",
                urls=urls_bse,
            )

            # reorder parallel scraped data
            def reorder_df(df: pd.DataFrame, column: str, order: list):
                df_copy = df.copy()
                # create unique null placeholders
                order = [
                    f"__null__{i}" if pd.isnull(item) else item
                    for i, item in enumerate(order)
                ]
                null_order = [item for item in order if "__null__" in item]

                # assign these null placeholders to df_copy
                null_iter = iter(null_order)
                df_copy[column] = [
                    next(null_iter) if pd.isnull(row) else row
                    for row in df_copy[column]
                ]

                # Use order as categories
                df_copy[column] = pd.Categorical(
                    df_copy[column], categories=order, ordered=True
                )

                # Sort dataframe
                df_sorted = df_copy.sort_values(by=column).reset_index(drop=True)

                # Convert placeholder back to None after sorting
                filt = df_sorted[column].str.contains("__null__", na=False)
                df_sorted.loc[filt, column] = None
                return df_sorted

            # log_etl.info(f"\n{self.company_names}")
            df_link = reorder_df(df_link, "company_name", self.company_names)
            # Then reorder df_data based on df_link["screener_link"]
            # log_etl.info(f"\n{df_link['screener_link'].to_list()}")
            df_data = reorder_df(
                df_data, "screener_link", df_link["screener_link"].to_list()
            )
            # log_etl.info(f"\n{df_data['bse_link'].to_list()}")
            df_bse = reorder_df(df_bse, "bse_link", df_data["bse_link"].to_list())
            log_etl.info("Extraction: Successfully reordered parallel scraped data")

            # reassign sorted, updated symbols
            df_data["bse_symbol"] = df_bse["bse_symbol"]

            for idx, row in df_data.iterrows():
                if (not row["bse_symbol"] or row["bse_symbol"] == "error") and (
                    not row["nse_symbol"] or row["nse_symbol"] == "error"
                ):
                    log_etl.info(
                        f"Extraction: Unable to find either exchange symbol for '{df_data.iloc[idx, 1]}' @ '{df_data.iloc[idx, 0]}'"
                    )

            # combine data
            df_scrn = pd.concat([self.data, df_data], axis=1, ignore_index=False)
            return df_scrn

        except Exception as e:
            LogException(e, logger=log_etl)
            data = {
                "screener_link": [np.nan] * len(self.company_names),
                "company_long_name": [np.nan] * len(self.company_names),
                "company_website": [np.nan] * len(self.company_names),
                "bse_link": [np.nan] * len(self.company_names),
                "bse_symbol": [np.nan] * len(self.company_names),
                "nse_link": [np.nan] * len(self.company_names),
                "nse_symbol": [np.nan] * len(self.company_names),
            }

            # raise CustomException(e)
            return pd.concat(
                [self.data, pd.DataFrame(data)], axis=1, ignore_index=False
            )


class ListingPriceExtractor:
    def __init__(
        self, Data: pd.DataFrame = None, ao_config: AngelOneConfig = AngelOneConfig()
    ):
        try:
            self.data = Data
            self.ao_config = ao_config
            log_etl.info("Extraction: Scraping Angel One token data")
            self.df_token = pd.DataFrame.from_dict(
                requests.get(self.ao_config.ao_token_url).json()
            )

        except Exception as e:
            LogException(e)
            raise CustomException(e)

    def get_company_token(self, symbol: str = None) -> Tuple[str, str]:
        try:
            filt = self.df_token["symbol"] == symbol
            token = self.df_token.loc[filt, "token"].values[0]
            exchange = self.df_token.loc[filt, "exch_seg"].values[0]
            return token, exchange

        except Exception as e:
            log_etl.info(
                f"Error in get_company_token(). Data: {token = }, {exchange = }. Error: {e}"
            )
            LogException(e)
            return np.nan, np.nan
            # raise CustomException(e)

    def get_listing_price(self, row):
        try:
            symbol = (
                row["nse_symbol"] if pd.notna(row["nse_symbol"]) else row["bse_symbol"]
            )
            if not symbol:  # <- in the case of unlisted company
                log_etl.info(
                    f"Extraction: Both exchange symbols are missing for '{row['Company_name']}' @ '{row['screener_link']}'"
                )
                return np.nan

            token, exchange = self.get_company_token(symbol=symbol)
            if pd.notna(token) or pd.notna(exchange):
                params = {
                    "exchange": exchange,
                    "symboltoken": token,
                    "interval": "ONE_HOUR",
                    "fromdate": f"{row['IPO_list_date']} 09:00",
                    "todate": f"{row['IPO_list_date']} 10:00",
                }
                hist_data = self.ao_smart_api.getCandleData(params)
                time.sleep(0.40)  # <- keep after api call

                if not hist_data["data"]:  # <- in the case of no data found
                    log_etl.info(
                        f"Extraction: Data Issue with Symbol: {symbol}, Data: {hist_data['data']}"
                    )
                    return np.nan

                return hist_data["data"][0][1]  #  <- in the case of data found

            else:
                log_etl.info(f"Extraction: Issue with {token = } or {exchange = }")
                return np.nan  # <- in the case of error in token or exchange

        except Exception as e:
            log_etl.info(f"Error in get_listing_price(). Symbol: {symbol}, Error: {e}")
            LogException(e)
            return np.nan

    def extract(self) -> pd.DataFrame:
        try:
            df_lp = self.data.copy()

            log_etl.info("Extraction: Initialising Angel One API")
            self.ao_totp = TOTP(self.ao_config.ao_qr_token).now()
            self.ao_smart_api = SmartConnect(self.ao_config.ao_api_key)
            data = self.ao_smart_api.generateSession(
                self.ao_config.ao_client_id, self.ao_config.ao_pin, self.ao_totp
            )
            refreshToken = data["data"]["refreshToken"]
            res = self.ao_smart_api.getProfile(refreshToken)
            self.ao_smart_api.generateToken(refreshToken)
            res = res["data"]["exchanges"]

            log_etl.info("Extraction: Making API calls")
            df_lp["IPO_listing_price"] = df_lp.apply(self.get_listing_price, axis=1)
            log_etl.info("Extraction: Successfully acquired listing price data")

            log_etl.info("Extraction: Logging out of Angel One API")
            _ = self.ao_smart_api.terminateSession(self.ao_config.ao_client_id)

            return df_lp

        except Exception as e:
            LogException(e)
            return df_lp
            # raise CustomException(e)


class CheckDatabase:
    def __init__(
        self,
        Data: pd.DataFrame = None,
        drop_sme: bool = True,
    ):
        try:
            self.data = Data
            self.drop_sme = drop_sme

        except Exception as e:
            LogException(e, "Extraction")
            raise CustomException(e)

    def filter(self):
        try:
            df_scrp = self.data.copy()
            log_etl.info("Extraction: Getting data from database for comparison")
            df_mgdb = get_df_from_MongoDB(
                collection="IPOPredMain",
                pipeline="etl",
                log=log_etl,
                prefix="Extraction",
            )

            if self.drop_sme:
                log_etl.info("Extraction: Dropping SME data from scraping")
                df_scrp = df_scrp.loc[df_scrp["IPO_category"] != "SME", :]

            log_etl.info("Extraction: Filtering out data already present in database")
            # select rows where df_scrp["Company_name"] is not in df_mgdb["IPO_company_name"]
            # select rows where df_scrp["Company_name"] is in df_mgdb["IPO_company_name"]
            # but those rows' df_mgdb["IPO_listing_price"] contains NaNs
            filt = (~df_scrp["Company_name"].isin(df_mgdb["IPO_company_name"])) | (
                df_scrp["Company_name"].isin(
                    df_mgdb.loc[df_mgdb["IPO_listing_price"].isna(), "IPO_company_name"]
                )
            )
            df_scrp = df_scrp.loc[filt, :]
            df_scrp.dropna(inplace=True, ignore_index=True)

            log_etl.info(
                f"Extraction: Using {df_scrp.shape[0]:03d} of {self.data.shape[0]:03d} data for scraping."
            )
            # log_etl.info(f"{df_scrp = }")
            return df_scrp

        except Exception as e:
            LogException(e, "Extraction")
            return self.data
            # raise CustomException(e)
