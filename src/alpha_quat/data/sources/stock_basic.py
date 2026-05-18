"""Tushare stock_basic API data source."""
from alpha_quat.data.source import DataSource

class StockBasicSource(DataSource):
    api_name = "stock_basic"
    partition_by = "none"
    fields = "ts_code,symbol,name,area,industry,market,list_status,list_date"

    def get_params(self, trade_date=None):
        return {"list_status": "L"}
