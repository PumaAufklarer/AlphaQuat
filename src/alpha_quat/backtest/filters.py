from pathlib import Path

import pandas as pd


def _date_to_path(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}_{yyyymmdd[4:6]}_{yyyymmdd[6:8]}"


def build_universe(trade_date: str, data_dir: Path) -> set[str]:
    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    main_board = set(sb.loc[sb["market"] == "主板", "ts_code"])
    st_path = data_dir / "stock_st" / f"{_date_to_path(trade_date)}.parquet"
    if st_path.exists():
        st = pd.read_parquet(st_path)
        return main_board - set(st["ts_code"])
    return main_board
