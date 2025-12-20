import re
import json
import time
import requests
import numpy as np
import pandas as pd
from lxml import html
from pyotp import TOTP
from datetime import datetime
from SmartApi import SmartConnect
from typing import List, Literal, Tuple

from crawl4ai.async_webcrawler import AsyncWebCrawler

from src.Logging.logger import log_etl
from src.Exception.exception import CustomException, LogException
from src.Entity.config_entity import AngelOneConfig
from src.ETL.etl_config.ETL_config import (
    ScreenerCSSCrawlerConfig,
    ScreenerHTMLCrawlerConfig,
    BSECrawlerConfig,
    CompanyCrawlConfig,
    GMPCrawlerConfig,
    ScrapeConfig,
)
from src.ETL.etl_utils.scrape_func import IPOScraper
from src.Utils.main_utils import get_df_from_MongoDB


class CompanyListExtractor:
    def __init__(
        self,
        start_year: int = 2012,
        end_year: int = datetime.now().year,
        crawl_config: CompanyCrawlConfig = CompanyCrawlConfig(),
    ):
        try:
            log_etl.info("Extraction: Initializing Web Scraping")
            self.start_year = start_year
            self.end_year = end_year
            self.crawl_config = crawl_config

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

            async with AsyncWebCrawler(
                config=self.crawl_config.browser_config
            ) as crawler:
                for url in url_list:
                    result = await crawler.arun(
                        url=url, config=self.crawl_config.crawler_run_config
                    )
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
    def __init__(self, Data: pd.DataFrame):
        self.urls = Data["Company_info_url"].to_list()
        self.data = Data

    async def get_ipo_detail(self, results):
        urls = [result.url for result in results]
        df_ipo = pd.DataFrame(
            {
                "result_link": urls,
                "IPO_open_date": [np.nan] * len(urls),
                "IPO_close_date": [np.nan] * len(urls),
                "IPO_list_date": [np.nan] * len(urls),
                "IPO_face_value": [np.nan] * len(urls),
                "IPO_issue_price": [np.nan] * len(urls),
                "IPO_lot_size": [np.nan] * len(urls),
                "IPO_issue_size": [np.nan] * len(urls),
            }
        )
        try:
            # scrape data
            df_ipo = await IPOScraper(
                urls=urls,
                purpose="detail",
                config=ScrapeConfig(len_list=len(urls)),
            ).scrape()

            # filter columns
            list_ipo_details = [
                "detail_url",
                "IPO Open Date",
                "IPO Close Date",
                "Listed on",
                "Tentative Listing Date",
                "Face Value",
                "Issue Price",
                "Price Band",
                "Lot Size",
                "Total Issue Size",
            ]
            df_ipo = df_ipo.loc[:, list_ipo_details]

            # datatype map
            dict_fmt = {
                "IPO Open Date": lambda x: np.nan
                if pd.isna(x)
                else datetime.strptime(x, "%a, %b %d, %Y").strftime("%Y-%m-%d"),
                "IPO Close Date": lambda x: np.nan
                if pd.isna(x)
                else datetime.strptime(x, "%a, %b %d, %Y").strftime("%Y-%m-%d"),
                "Listed on": lambda x: np.nan
                if pd.isna(x) or x == "[.]"
                else datetime.strptime(x, "%a %b %d %Y").strftime("%Y-%m-%d"),
                "Tentative Listing Date": lambda x: np.nan
                if pd.isna(x)
                else datetime.strptime(x, "%a, %b %d, %Y").strftime("%Y-%m-%d"),
                "Face Value": lambda x: int(re.findall(r"\d+", x)[0]),
                "Issue Price": lambda x: np.nan
                if pd.isna(x) or x == "[.]"
                else int(re.findall(r"\d+", x)[0]),
                "Price Band": lambda x: int(re.findall(r"\d+", x)[1]),
                "Lot Size": lambda x: int(re.findall(r"\d+", x)[0]),
                "Total Issue Size": lambda x: float(re.findall(r"₹([^C]+?)Cr", x)[0]),
            }

            # clean and typecast
            for col, formatter in dict_fmt.items():
                if col in df_ipo.columns:
                    df_ipo[col] = df_ipo[col].map(formatter)

            # replace missing values
            df_ipo["Listed on"] = df_ipo["Listed on"].fillna(
                df_ipo["Tentative Listing Date"]
            )
            df_ipo["Issue Price"] = df_ipo["Issue Price"].fillna(df_ipo["Price Band"])

            # rename columns
            repl_cols = {
                "detail_url": "result_link",
                "IPO Open Date": "IPO_open_date",
                "IPO Close Date": "IPO_close_date",
                "Listed on": "IPO_list_date",
                "Face Value": "IPO_face_value",
                "Issue Price": "IPO_issue_price",
                "Lot Size": "IPO_lot_size",
                "Total Issue Size": "IPO_issue_size",
            }
            df_ipo.rename(columns=repl_cols, inplace=True)

            # drop remaining columns
            df_ipo.drop(columns=["Tentative Listing Date", "Price Band"], inplace=True)

        except Exception as e:
            LogException(e, logger=log_etl)
            # raise CustomException(e)

        # df_ipo.to_pickle("./z_check_data/ipo_details.pkl")
        return df_ipo

    def get_rhp_doc(self, urls: list[str]):
        df_rhp = pd.DataFrame(
            {"rhp_link": [link for link in urls]},
        )
        try:
            # This function is supposed to process pdf and extract some company financial
            # and risk factor data but I'll do that later
            pass

        except Exception as e:
            LogException(e, logger=log_etl)
            # raise CustomException(e)

        return df_rhp

    async def get_gmp_link(self, urls: list[str]):
        df_gmp = pd.DataFrame(
            {
                **{"gmp_link": [link for link in urls]},
                **{
                    f"day{i}_{m}": [np.nan] * len(urls)
                    for i in range(1, 4)
                    for m in ["date", "price"]
                },
            }
        )
        try:
            # scrape
            df_gmp = await IPOScraper(
                urls=urls,
                purpose="grmkpt",
                config=ScrapeConfig(len_list=len(urls)),
            ).scrape()

            # filter data
            metrics = ["Date", "GMP"]
            gmp_cols_list = ["grmkpt_url"] + [
                col
                for col in df_gmp.columns  # dynamic selection
                if re.match(
                    r"Day_\d+_(?:" + "|".join(re.escape(m) for m in metrics) + r")", col
                )
            ]
            df_gmp = df_gmp.loc[:, gmp_cols_list]

            # convert dtypes
            date_cols = [col for col in gmp_cols_list if "Date" in col]
            gmp_cols = [col for col in gmp_cols_list if "GMP" in col]
            for col in date_cols:
                df_gmp[col] = pd.to_datetime(
                    df_gmp[col], format="%d-%m-%Y", errors="coerce"
                ).dt.strftime("%Y-%m-%d")
            for col in gmp_cols:
                df_gmp[col] = (
                    df_gmp[col].str.replace("₹", "", regex=False).astype("float64")
                )

            # rename columns
            repl_cols = {"grmkpt_url": "gmp_link"}
            for i in range(len(date_cols)):  # dynamic rename
                for m, n in zip(["date", "price"], ["Date", "GMP"]):
                    old_col = f"day{i + 1}_{m}"
                    new_col = f"Day_{i + 1}_{n}"
                    repl_cols[new_col] = old_col
            df_gmp.rename(columns=repl_cols, inplace=True)

            # rearrange columns
            def reorder_dates(data: pd.DataFrame) -> pd.DataFrame:
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

            df_gmp = reorder_dates(df_gmp)

        except Exception as e:
            LogException(e, logger=log_etl)
            # raise CustomException(e)

        # df_gmp.to_pickle("./z_check_data/gmp_details.pkl")
        return df_gmp

    async def get_ipo_review(self, urls: list[str]):
        df_rvw = pd.DataFrame(
            {
                **{"review_link": [url for url in urls]},
                **{
                    f"IPO_{i}_{m}": [np.nan] * len(urls)
                    for i in ["Broker", "Member"]
                    for m in ["apply", "neutral", "avoid"]
                },
            }
        )
        try:
            # scrape data
            df_rvw = await IPOScraper(
                urls=urls,
                purpose="review",
                config=ScrapeConfig(len_list=len(urls)),
            ).scrape()

            # filter columns
            rvw_cols_list = [
                "review_url",
                "Broker Apply",
                "Broker May Apply",
                "Broker Neutral",
                "Broker Avoid",
                "Broker Not Rated",
                "Member Apply",
                "Member May Apply",
                "Member Not Rated",
                "Member Avoid",
            ]
            df_rvw = df_rvw.loc[:, rvw_cols_list]

            # convert dtypes
            int_cols = df_rvw.columns[1:]
            df_rvw[int_cols] = (
                df_rvw[int_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
            )

            # sum columns
            df_rvw["Broker Apply"] += df_rvw["Broker May Apply"]
            df_rvw["Member Apply"] += df_rvw["Member May Apply"]

            # add column
            df_rvw["Member Neutral"] = (
                pd.Series([0] * df_rvw.shape[0])
                if "Member Neutral" not in df_rvw.columns
                else df_rvw["Member Neutral"]
            )

            # rename columns
            repl_cols = {
                "review_url": "review_link",
                "Broker Apply": "IPO_Broker_apply",
                "Broker Neutral": "IPO_Broker_neutral",
                "Broker Avoid": "IPO_Broker_avoid",
                "Member Apply": "IPO_Member_apply",
                "Member Neutral": "IPO_Member_neutral",
                "Member Avoid": "IPO_Member_avoid",
            }
            df_rvw.rename(columns=repl_cols, inplace=True)

            # drop remaining columns
            drop_cols = [
                "Broker May Apply",
                "Broker Not Rated",
                "Member May Apply",
                "Member Not Rated",
            ]
            df_rvw.drop(columns=drop_cols, inplace=True)

        except Exception as e:
            LogException(e, logger=log_etl)
            # raise CustomException(e)

        # df_rvw.to_pickle("./z_check_data/rvw_details.pkl")
        return df_rvw

    async def get_ipo_subscription(self, urls):
        df_subsc = pd.DataFrame(
            {
                **{"subsc_link": [url for url in urls]},
                **{
                    f"IPO_day{i}_{m}": [np.nan] * len(urls)
                    for i in range(1, 4)
                    for m in ["qib", "nii", "rtl"]
                },
            }
        )

        try:
            # scrape data
            df_subsc = await IPOScraper(
                urls=urls,
                purpose="sbscrp",
                config=ScrapeConfig(len_list=len(urls)),
            ).scrape()

            # filter data
            metrics = [
                "Date",
                "QIB",
                "QIB (Ex Anchor)",
                "NII",
                "NII*",
                "Retail",
                "Individual Investors",
            ]
            sbs_cols_list = ["sbscrp_url"] + [  # Dynamic method to capture data
                col
                for col in df_subsc.columns
                if re.match(
                    r"Day \d+_(?:"
                    + "|".join(re.escape(m) for m in metrics)
                    + r")(?!\s*\()",
                    col,
                )
            ]
            df_subsc = df_subsc.loc[:, sbs_cols_list]

            # convert dtypes
            date_cols = [col for col in df_subsc.columns if metrics[0] in col]
            numeric_cols = [
                col
                for col in df_subsc.columns
                if any(metric in col for metric in metrics[1:])
            ]
            for col in date_cols:
                df_subsc[col] = pd.to_datetime(
                    df_subsc[col], format="%b %d %Y", errors="coerce"
                ).dt.strftime("%Y-%m-%d")
            for col in numeric_cols:
                df_subsc[col] = pd.to_numeric(df_subsc[col], errors="coerce").astype(
                    "float64"
                )

            # rename cols
            itr = int((len(sbs_cols_list) - 1) / 3)
            repl_cols = {"sbscrp_url": "subsc_link"}
            day_cols = [col for col in sbs_cols_list[1:] if "Date" not in col]
            for i in range(itr):
                start_idx = i * 3  # Dynamic method for column remapper
                for j, m in enumerate(["qib", "nii", "rtl"]):
                    if start_idx + j < len(day_cols):
                        old_col = day_cols[start_idx + j]
                        new_col = f"IPO_day{i + 1}_{m}"
                        repl_cols[old_col] = new_col
            # repl_cols
            df_subsc.rename(columns=repl_cols, inplace=True)

            # drop columns
            drop_cols = [col for col in sbs_cols_list[1:] if "Date" in col]
            df_subsc.drop(columns=drop_cols, inplace=True)

        except Exception as e:
            LogException(e, logger=log_etl)
            # raise CustomException(e)

        # df_subsc.to_pickle("./z_check_data/sbs_details.pkl")
        return df_subsc

    async def extract(self):
        df_lnk = self.data.copy()
        config = ScrapeConfig(len_list=len(self.urls))

        async with AsyncWebCrawler(config=config.browser_config) as crawler:
            results = await crawler.arun_many(
                urls=self.urls,
                config=config.rncf_ipo_links,
                dispatcher=config.mem_ada_dispatcher,
            )

            def _get_relevant_links(results):
                map_dict = {
                    "Reviews": "review_link",
                    "Subscription": "subsc_link",
                    "DRHP": "rhp_link",
                    "GMP": "gmp_link",
                }
                urls_dict = {
                    "result_link": [result.url for result in results],
                    **{key: [] * len(map_dict.keys()) for key in map_dict.values()},
                }
                links_int = [result.links.get("internal", []) for result in results]
                links_ext = [result.links.get("external", []) for result in results]

                for key in list(map_dict.keys())[:2]:
                    for website in links_int:
                        for link in website:
                            if link["text"] == key:
                                urls_dict[map_dict[key]].append(link["href"])

                for key in list(map_dict.keys())[2:]:
                    for website in links_ext:
                        for link in website:
                            if link["text"] == key:
                                urls_dict[map_dict[key]].append(link["href"])

                keys = [
                    "result_link",
                    "rhp_link",
                    "review_link",
                    "subsc_link",
                    "gmp_link",
                ]
                urls_dict = {key: urls_dict[key] for key in keys}
                return urls_dict

            urls_dict = _get_relevant_links(results)
            keys = list(urls_dict.keys())

            try:
                log_etl.info("Extraction: Scraping IPO details")
                df_ipo = await self.get_ipo_detail(results=results)

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
                df_comb = pd.concat(
                    df_list,
                    axis=1,
                    ignore_index=False,
                )

            except Exception as e:
                LogException(e, logger=log_etl)
                # raise CustomException(e)

        # df_comb.to_pickle("./z_check_data/comb_details.pkl")
        return df_comb


class ScreenerExtractor:
    def __init__(
        self,
        Data: pd.DataFrame,
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
            LogException(e, logger=log_etl)
            return "error"

    async def get_bse_symbol(self, urls: List[str]) -> pd.DataFrame:
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
                    LogException(e, logger=log_etl)
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
                    LogException(e, logger=log_etl)
                    df_data = pd.concat(
                        [df_data, pd.DataFrame(data)], axis=0, ignore_index=True
                    )
                    continue
                    # raise CustomException(e)

            return df_data

    async def get_screener_links(self, urls: List[str]) -> pd.DataFrame:
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
                    LogException(e, logger=log_etl)
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
                LogException(e, logger=log_etl)
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
            # df_scrn.to_pickle("./z_check_data/srcn_details.pkl")
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
            LogException(e, logger=log_etl)
            raise CustomException(e)

    def get_company_token(self, symbol: str) -> Tuple[str, str]:
        try:
            filt = self.df_token["symbol"] == symbol
            token = self.df_token.loc[filt, "token"].values[0]
            exchange = self.df_token.loc[filt, "exch_seg"].values[0]
            return token, exchange

        except Exception as e:
            log_etl.info(
                f"Error in get_company_token(). Data: {token = }, {exchange = }. Error: {e}"
            )
            LogException(e, logger=log_etl)
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
            LogException(e, logger=log_etl)
            return np.nan

    def extract(self) -> pd.DataFrame:
        try:
            df_lspr = self.data.copy()

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
            df_lspr["IPO_listing_price"] = df_lspr.apply(self.get_listing_price, axis=1)
            log_etl.info("Extraction: Successfully acquired listing price data")

            log_etl.info("Extraction: Logging out of Angel One API")
            _ = self.ao_smart_api.terminateSession(self.ao_config.ao_client_id)

            # df_lspr.to_pickle("./z_check_data/lspr_details.pkl")
            return df_lspr

        except Exception as e:
            LogException(e, logger=log_etl)
            return df_lspr
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
