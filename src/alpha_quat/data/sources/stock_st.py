"""Tushare stock_st API data source (ST stock list)."""

from alpha_quat.data.source import DataSource


class StockStSource(DataSource):
    api_name = "stock_st"
    partition_by = "date"
    fields = "ts_code,name,trade_date,type,type_name"
    start_date = "20160101"

    def get_params(self, trade_date=None):
        return {"trade_date": trade_date}
