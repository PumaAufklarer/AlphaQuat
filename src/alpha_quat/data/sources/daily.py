"""Tushare daily API data source."""
from alpha_quat.data.source import DataSource

class DailySource(DataSource):
    api_name = "daily"
    partition_by = "date"
    fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"

    def get_params(self, trade_date=None):
        return {"trade_date": trade_date}
