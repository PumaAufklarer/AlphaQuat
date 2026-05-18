"""Tushare stk_st API data source (ST stock list)."""

from alpha_quat.data.source import DataSource


class StkStSource(DataSource):
    api_name = "stk_st"
    partition_by = "date"
    fields = "ts_code,name,type"

    def get_params(self, trade_date=None):
        return {"trade_date": trade_date}
