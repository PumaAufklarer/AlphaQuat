"""Tushare API wrapper with retry logic."""

import time

import tushare


class FetchError(Exception):
    pass


class Fetcher:
    def __init__(
        self,
        token: str,
        max_retries: int = 12,
        retry_delay: float = 5.0,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # set_token mutates global tushare state; kept for downstream
        # compatibility (e.g. clients that call tushare directly).
        tushare.set_token(token)
        self._api = tushare.pro_api(token)

    def query(self, api_name: str, **params):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._api.query(api_name, **params)
            except Exception as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise FetchError(f"Failed after {self.max_retries} retries: {last_error}")
