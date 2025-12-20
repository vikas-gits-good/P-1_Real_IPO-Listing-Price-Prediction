import os
import re
from typing import Literal, List
from datetime import datetime
from dataclasses import dataclass
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai import (
    CacheMode,
    RateLimiter,
    CrawlerMonitor,
    DefaultTableExtraction,
    MemoryAdaptiveDispatcher,
    JsonCssExtractionStrategy,
    TableExtractionStrategy,
    ProxyConfig,
    RoundRobinProxyStrategy,
)

from dotenv import load_dotenv
from src.ETL.etl_config import etl_constants


class ExtractionConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        self.right_now = timestamp
        self.timestamp = timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        self.data_dir = etl_constants.ETL_DATA_DIR
        self.data_dir_path = os.path.join(self.data_dir, self.timestamp)


@dataclass
class ExtractionArtifact:
    extracted_data_file_path: str


class TransformationConfig:
    def __init__(self):
        self.transformation_file_name = etl_constants.TRANSFORMED_DATA_FILE_NAME


@dataclass
class TransformationArtifact:
    transformed_data_file_path: str


class CompanyCrawlConfig:
    def __init__(self):
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,  # False,  #
            verbose=False,
        )
        JSON_SCHEMA = {
            "name": "Company Info",
            "baseSelector": "div > table > tbody > tr:not(:has(td:first-child div a > del))",
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

        extract_strat = JsonCssExtractionStrategy(schema=JSON_SCHEMA)
        self.crawler_run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extract_strat,
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
        )


class GMPCrawlerConfig:
    def __init__(self, max_parallel: int = 10, len_list: int = 0):
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
        )
        table_strategy = DefaultTableExtraction(
            table_score_threshold=5, min_rows=2, min_cols=2
        )
        self.crawler_run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_extraction=table_strategy,
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
        )
        self.rate_limiter = RateLimiter(
            base_delay=(1.0, 3.0),
            max_delay=10.0,
            max_retries=3,
            rate_limit_codes=[429, 503],
        )
        self.crawl_monitor = CrawlerMonitor(
            urls_total=len_list,
            refresh_rate=0.1,  # 0.5,  #
            enable_ui=True,
            max_width=120,
        )
        self.memory_adap_dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            max_session_permit=max_parallel,
            check_interval=1.0,
            rate_limiter=self.rate_limiter,
            monitor=self.crawl_monitor,
        )


class WebShareConfig:
    def __init__(self, instances: int = 4):
        load_dotenv("src/Secrets/proxy.env")
        domain = os.getenv("DOMAIN")
        port = os.getenv("PORT")
        countries = os.getenv("COUNTRIES").split("-")
        username = f"{os.getenv('PROXY_USERNAME')}{''.join([f'-{x}' for x in countries])}-rotate"
        password = os.getenv("PROXY_PASSWORD")
        server = f"http://{domain}:{port}"

        self.proxy_url = f"http://{username}:{password}@{domain}:{port}"
        self.proxy_configs = [
            ProxyConfig(server=server, username=username, password=password)
            for _ in range(instances)
        ]
        self.proxy_rotation_strat = RoundRobinProxyStrategy(proxies=self.proxy_configs)


class ScrapeConfig:
    def __init__(self, max_parallel: int = 6, len_list: int = 0) -> None:
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,  # False,  #
            verbose=False,
            enable_stealth=True,
        )

        # strategy to get links
        self.rncf_ipo_links = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
            wait_for=".d-flex.flex-wrap.gap-2",
        )

        # strategy to get ipo_details
        self.rncf_ipo_details = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_extraction=IPOTableExtractor(purpose="detail"),
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
            wait_for=".col-md-9",
        )

        # strategy to get ipo_reviews
        self.rncf_ipo_reviews = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_extraction=IPOTableExtractor(purpose="review"),
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
            wait_for=".col-12.col-md-9",
        )

        # strategy to get ipo_subscriptions
        self.rncf_ipo_sbscrps = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_extraction=IPOTableExtractor(purpose="sbscrp"),
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
            wait_for=".col-12.col-md-9",
        )

        # strategy to get ipo_gmp
        self.rncf_ipo_grmkpts = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_extraction=IPOTableExtractor(purpose="grmkpt"),
            js_code=[
                """await new Promise(r=>setTimeout(r,Math.random()*10000+5000));"""
            ],
            wait_for=".col-lg-8.col-md-8.col-sm-12.px-0",
            proxy_rotation_strategy=WebShareConfig().proxy_rotation_strat,
        )

        rate_limiter = RateLimiter(
            base_delay=(10, 15),
            max_delay=60.0,
            max_retries=3,
            rate_limit_codes=[429, 503, 403],
        )
        crawl_monitor = CrawlerMonitor(
            urls_total=len_list,
            refresh_rate=0.1,
            enable_ui=True,
            max_width=120,
        )

        self.mem_ada_dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            max_session_permit=max_parallel,
            check_interval=1.0,
            rate_limiter=rate_limiter,
            monitor=crawl_monitor,
        )


class IPOTableExtractor(TableExtractionStrategy):
    """Extract tables containing IPO data."""

    def __init__(
        self,
        purpose: Literal["detail", "review", "sbscrp", "grmkpt"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.number_pattern = re.compile(r"\d+[,.]?\d*")
        self.purpose = purpose
        self.table_identifier = {
            "detail": [
                "IPO Date",
                "Total Issue Size",
            ],
            "review": ["Not Rated"],  # <- Don't use 'Review By'
            "sbscrp": ["Date"],
            "grmkpt": ["GMP Date"],
        }.get(self.purpose, ["Not Rated"])

    def extract_tables(self, element, **kwargs):
        self.tables_data = []
        for table in element.xpath(".//table"):
            table_text = "".join(table.itertext())

            # Must contain numbers
            numbers = self.number_pattern.findall(table_text)
            if len(numbers) < 1:
                continue

            # Identify table as per purpose
            if any(
                col in self.table_identifier
                for col in [
                    th.text_content().strip()
                    for th in table.xpath(
                        ".//thead//th | .//tr[1]//th | .//tbody//tr[position()>0]//td[1]//span//a"  #
                    )
                ]
            ):
                req_table = table
            else:
                continue

            # Extract the table data
            table_data_0 = self._extract_tabular_data(req_table)
            if table_data_0:
                self.tables_data.append(table_data_0)

            if self.purpose == "detail" and len(self.tables_data) < 3:
                table_data_1 = self._get_tentative_date(req_table, element)
                self.tables_data.append(table_data_1)

        return self.tables_data

    def _get_tentative_date(self, table, element):
        headers = []
        rows = []
        # Extract Tentative listing date
        row = []
        for line in element.xpath(
            "//html//body//div[8]//div[5]//div[1]//div[2]//div//div//ul//li"
        ):
            for span in line.xpath(".//span"):
                text = span.text_content().strip()
                row.append(text)

        if row:
            headers.extend([x for i, x in enumerate(row) if i % 2 == 0])
            rows.extend([x for i, x in enumerate(row) if i % 2 != 0])

        return {
            "headers": headers,
            "rows": rows,
            "data": self._return_data(headers, rows),
            "caption": self._get_caption(),
            "summary": table.get("summary", ""),
            "metadata": {
                "type": "ipo_data",
                "row_count": len(rows),
                "column_count": len(headers) or len(rows[0]) if rows else 0,
            },
        }

    def _extract_tabular_data(self, table):
        """Extract and clean data from table."""
        headers = []
        rows = []
        # Extract headers
        for th in table.xpath(".//thead//th | .//tr[1]//th"):
            headers.append(th.text_content().strip())

        # Extract and clean rows
        for tr in table.xpath(".//tbody//tr | .//tr[position()>1]"):
            row = []
            for td in tr.xpath(".//td"):
                text = td.text_content().strip()
                text = re.sub(r"[$€£¥,]", "", text)
                row.append(text)
            if row:
                if self.purpose == "detail":
                    headers.append(row[0])
                    rows.append(row[1])
                else:
                    rows.append(row)

        return {
            "headers": headers,
            "rows": rows,
            "data": self._return_data(headers, rows),
            "caption": self._get_caption(),
            "summary": table.get("summary", ""),
            "metadata": {
                "type": "ipo_data",
                "row_count": len(rows),
                "column_count": len(headers) or len(rows[0]) if rows else 0,
            },
        }

    def _get_caption(self):
        caption = {
            "detail": "ipo_details",
            "review": "ipo_reviews",
            "sbscrp": "ipo_subscriptions",
            "grmkpt": "ipo_gmp",
        }.get(self.purpose, "")
        return caption

    def _return_data(self, headers, rows):
        return_data = {}
        if self.purpose == "detail":
            return_data = {
                key: val if isinstance(val, list) else [val]
                for key, val in zip(headers, rows)
            }

        elif self.purpose == "review":
            rows[0][0] = "Broker" if len(self.tables_data) == 0 else "Member"
            return_data = {
                key: val if isinstance(val, list) else [val]
                for key, val in zip(headers, rows[0])
            }

        elif self.purpose == "sbscrp":
            return_data = {
                header: [row[i] for row in rows] for i, header in enumerate(headers)
            }

        elif self.purpose == "grmkpt":
            return_data = {
                header: [row[i] for row in rows] for i, header in enumerate(headers)
            }
        return return_data


class ScreenerCSSCrawlerConfig:
    def __init__(self, max_parallel: int = 5, len_list: int = 0):
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
        )
        company_schema = {
            "name": "Company_Details",
            "baseSelector": "body",  # /html/body/
            "fields": [
                {
                    "name": "company_long_name",  # /html/body/div/div[1]/h1
                    "selector": "body > div > div:nth-child(1) > h1",
                    "type": "text",
                },
                {
                    "name": "link_1",  # /html/body/main/div[3]/div[3]/div[1]/div[2]/a[1]
                    "selector": "main > div:nth-child(3) > div:nth-child(3) > div:nth-child(1) > div:nth-child(2) > a",
                    "type": "attribute",
                    "attribute": "href",
                },
                {
                    "name": "link_2",  # /html/body/main/div[3]/div[3]/div[1]/div[2]/a[2]
                    "selector": "main > div:nth-child(3) > div:nth-child(3) > div:nth-child(1) > div:nth-child(2) > a:nth-child(2)",
                    "type": "attribute",
                    "attribute": "href",
                },
                {
                    "name": "link_3",  # /html/body/main/div[3]/div[3]/div[1]/div[2]/a[3]
                    "selector": "main > div:nth-child(3) > div:nth-child(3) > div:nth-child(1) > div:nth-child(2) > a:nth-child(3)",
                    "type": "attribute",
                    "attribute": "href",
                },
            ],
        }
        extraction_company_strategy = JsonCssExtractionStrategy(
            company_schema, verbose=False
        )
        # dont need table_strategy keeping it to delay page loading
        table_strategy = DefaultTableExtraction(
            table_score_threshold=5, min_rows=2, min_cols=2
        )
        self.run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_company_strategy,
            wait_for="xpath:/html/body/main/div[3]/div[3]/div[1]/div[2]/a",
            table_extraction=table_strategy,
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
            # stream=True,
        )

        # Rate limiting to avoid request throttling or blocking:
        self.rate_limiter = RateLimiter(
            base_delay=(5, 10),  # Random delay
            max_delay=40.0,  # Maximum delay on exponential backoff
            max_retries=3,  # Retry 3 times on 429/503 errors
            rate_limit_codes=[429, 503],
        )

        # Optional: live monitoring of crawling progress
        self.crawl_monitor = CrawlerMonitor(
            urls_total=len_list,
            refresh_rate=0.1,
            enable_ui=True,
            max_width=120,
        )

        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            max_session_permit=max_parallel,
            check_interval=1.0,
            rate_limiter=self.rate_limiter,
            monitor=self.crawl_monitor,
        )


class ScreenerHTMLCrawlerConfig:
    def __init__(self, max_parallel: int = 5, len_list: int = 0):
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
        )
        # dont need table_strategy keeping it to delay page loading
        table_strategy = DefaultTableExtraction(
            table_score_threshold=5, min_rows=2, min_cols=2
        )
        self.run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_for="xpath:/html/body/main/div[3]/div[3]/div[1]/div[2]/a",
            table_extraction=table_strategy,
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
            # stream=True,
        )

        # Rate limiting to avoid request throttling or blocking:
        self.rate_limiter = RateLimiter(
            base_delay=(5, 10),  # Random delay
            max_delay=40.0,  # Maximum delay on exponential backoff
            max_retries=3,  # Retry 3 times on 429/503 errors
            rate_limit_codes=[429, 503],
        )

        # Optional: live monitoring of crawling progress
        self.crawl_monitor = CrawlerMonitor(
            urls_total=len_list,
            refresh_rate=0.1,
            enable_ui=True,
            max_width=120,
        )

        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            max_session_permit=max_parallel,
            check_interval=1.0,
            rate_limiter=self.rate_limiter,
            monitor=self.crawl_monitor,
        )


class BSECrawlerConfig:
    def __init__(self, max_parallel: int = 10, len_list: int = 0):
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,  # False,  #
            verbose=False,
        )
        table_strategy = DefaultTableExtraction(
            table_score_threshold=5, min_rows=2, min_cols=2
        )
        self.run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_extraction=table_strategy,
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
        )

        # Rate limiting to avoid request throttling or blocking:
        self.rate_limiter = RateLimiter(
            base_delay=(5, 10),  # Random delay
            max_delay=40.0,  # Maximum delay on exponential backoff
            max_retries=3,  # Retry 3 times on 429/503 errors
            rate_limit_codes=[429, 503],
        )

        # Optional: live monitoring of crawling progress
        self.crawl_monitor = CrawlerMonitor(
            urls_total=len_list,
            refresh_rate=0.1,
            enable_ui=True,
            max_width=120,
        )

        self.dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0,
            max_session_permit=max_parallel,
            check_interval=1.0,
            rate_limiter=self.rate_limiter,
            monitor=self.crawl_monitor,
        )
