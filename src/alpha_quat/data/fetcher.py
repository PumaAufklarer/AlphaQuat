"""Tushare API wrapper with retry logic."""

import time

import tushare


class FetchError(Exception):
    pass


class Fetcher:
    def __init__(
        self,
        token: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        tushare.set_token(token)
        self._api = tushare.pro_api(token)

    def query(self, api_name: str, **params):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._api.query(api_name, **params)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise FetchError(
            f"Failed after {self.max_retries} retries: {last_error}"
        )
