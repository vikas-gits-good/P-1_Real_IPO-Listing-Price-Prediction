import os
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
)

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


class LoadConfig:
    def __init__(self):
        pass


class CompanyCrawlConfig:
    def __init__(self):
        self.browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
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
            scraping_strategy=extract_strat,
            js_code=["""await new Promise(r=>setTimeout(r,5000));"""],
        )


class GMPCrawlerConfig:
    def __init__(self, max_parallel: int = 10, len_list: int = None):
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


class ScreenerCSSCrawlerConfig:
    def __init__(self, max_parallel: int = 5, len_list: int = None):
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
    def __init__(self, max_parallel: int = 5, len_list: int = None):
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
    def __init__(self, max_parallel: int = 10, len_list: int = None):
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
