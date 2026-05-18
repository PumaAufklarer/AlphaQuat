"""Tushare daily_basic API data source."""

from alpha_quat.data.source import DataSource


class DailyBasicSource(DataSource):
    api_name = "daily_basic"
    partition_by = "date"
    fields = "ts_code,trade_date,total_mv,circ_mv,pe,pe_ttm,pb,turnover_rate,turnover_rate_f,volume_ratio"

    def get_params(self, trade_date=None):
        return {"trade_date": trade_date}
