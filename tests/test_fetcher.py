"""Tests for fetcher module."""

import pandas as pd
import pytest

from alpha_quat.data.fetcher import Fetcher, FetchError


class FakeProApi:
    def __init__(self, results=None, fail_count=0):
        self.results = results or [pd.DataFrame({"col": [1]})]
        self.fail_count = fail_count
        self.call_count = 0
        self.last_api: str | None = None
        self.last_params: dict = {}

    def query(self, api_name, **params):
        self.call_count += 1
        self.last_api = api_name
        self.last_params = params
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("tushare error")
        return self.results[0]


def test_fetcher_query_returns_dataframe(monkeypatch):
    fake = FakeProApi()
    monkeypatch.setattr("alpha_quat.data.fetcher.tushare.pro_api", lambda token: fake)

    fetcher = Fetcher(token="dummy")
    result = fetcher.query("daily", trade_date="20240101")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert fake.last_api == "daily"
    assert fake.last_params["trade_date"] == "20240101"
    assert fake.call_count == 1


def test_fetcher_set_token_called_with_correct_token(monkeypatch):
    called_with = []

    def fake_set_token(t):
        called_with.append(t)

    monkeypatch.setattr("alpha_quat.data.fetcher.tushare.set_token", fake_set_token)
    monkeypatch.setattr(
        "alpha_quat.data.fetcher.tushare.pro_api", lambda token: FakeProApi()
    )

    Fetcher(token="my_token_456")
    assert called_with == ["my_token_456"]


def test_fetcher_retry_on_error(monkeypatch):
    fake = FakeProApi(fail_count=2)
    monkeypatch.setattr("alpha_quat.data.fetcher.tushare.pro_api", lambda token: fake)

    fetcher = Fetcher(token="dummy", max_retries=3, retry_delay=0.01)
    result = fetcher.query("stk_st", trade_date="20240101")

    assert len(result) == 1
    assert fake.call_count == 3  # 2 fails + 1 success


def test_fetcher_raises_after_max_retries(monkeypatch):
    fake = FakeProApi(fail_count=10)
    monkeypatch.setattr("alpha_quat.data.fetcher.tushare.pro_api", lambda token: fake)

    fetcher = Fetcher(token="dummy", max_retries=2, retry_delay=0.01)

    with pytest.raises(FetchError, match="Failed after 2 retries"):
        fetcher.query("daily", trade_date="20240101")

    assert fake.call_count == 2


def test_fetcher_stock_basic_query(monkeypatch):
    fake = FakeProApi(results=[pd.DataFrame({"ts_code": ["000001.SZ"]})])
    monkeypatch.setattr("alpha_quat.data.fetcher.tushare.pro_api", lambda token: fake)

    fetcher = Fetcher(token="dummy")
    result = fetcher.query("stock_basic", list_status="L", fields="ts_code,name")

    assert fake.last_api == "stock_basic"
    assert fake.last_params["list_status"] == "L"
    assert result.iloc[0]["ts_code"] == "000001.SZ"
