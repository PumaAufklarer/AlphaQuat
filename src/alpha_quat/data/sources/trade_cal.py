"""Tushare trade_cal API data source."""
from alpha_quat.data.source import DataSource

class TradeCalSource(DataSource):
    api_name = "trade_cal"
    partition_by = "none"
    fields = "exchange,cal_date,is_open,pretrade_date"

    def get_params(self, trade_date=None):
        return {"exchange": "SSE"}
