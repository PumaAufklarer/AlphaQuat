from pathlib import Path

import pandas as pd

from alpha_quat.model.constants import date_to_path


def _get_industry_map(data_dir: Path) -> dict[str, str]:
    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    return dict(zip(sb["ts_code"], sb["industry"]))


_INDUSTRY_MAP: dict[str, str] | None = None


def build_universe(
    trade_date: str,
    data_dir: Path,
    quality_filter: bool = False,
    min_price: float = 0.0,
) -> set[str]:
    global _INDUSTRY_MAP
    if _INDUSTRY_MAP is None:
        _INDUSTRY_MAP = _get_industry_map(data_dir)

    sb = pd.read_parquet(data_dir / "stock_basic.parquet")
    main_board = set(sb.loc[sb["market"] == "主板", "ts_code"])

    # Remove ST stocks
    st_path = data_dir / "stock_st" / f"{date_to_path(trade_date)}.parquet"
    if st_path.exists():
        st = pd.read_parquet(st_path)
        main_board -= set(st["ts_code"])

    universe = main_board

    if min_price > 0:
        daily_path = data_dir / "daily" / f"{date_to_path(trade_date)}.parquet"
        if daily_path.exists():
            daily = pd.read_parquet(daily_path)
            high_price_stocks = set(daily.loc[daily["high"] >= min_price, "ts_code"])
            universe &= high_price_stocks

    if not quality_filter:
        return universe

    # Quality filter: industry-relative PE/PB + market cap + liquidity
    db_path = data_dir / "daily_basic" / f"{date_to_path(trade_date)}.parquet"
    if not db_path.exists():
        return main_board

    db = pd.read_parquet(db_path)
    db["industry"] = db["ts_code"].map(_INDUSTRY_MAP)
    db = db.dropna(subset=["industry", "pe_ttm", "pb", "circ_mv", "turnover_rate"])
    if db.empty:
        return main_board

    # Industry medians
    ind_pe = db.groupby("industry")["pe_ttm"].transform("median").replace(0, 1e-6)
    ind_pb = db.groupby("industry")["pb"].transform("median").replace(0, 1e-6)

    mask = (
        (db["circ_mv"] >= 500000)
        & (db["turnover_rate"] >= 0.1)
        & (db["pe_ttm"] > 0)
        & (db["pe_ttm"] < ind_pe * 1.5)
        & (db["pe_ttm"] >= ind_pe * 0.1)
        & (db["pb"] > 0)
        & (db["pb"] < ind_pb * 1.5)
        & (db["pb"] >= ind_pb * 0.1)
    )
    quality_set = set(db.loc[mask, "ts_code"])
    universe &= quality_set
    return universe
