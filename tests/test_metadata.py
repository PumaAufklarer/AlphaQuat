"""Tests for metadata manager module."""

from alpha_quat.data.metadata import MetadataManager


def test_init_creates_table(tmp_path):
    db_path = str(tmp_path / "test_registry.db")
    mgr = MetadataManager(db_path)

    result = mgr.conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name='data_registry'"
    ).fetchall()
    assert len(result) == 1
    assert result[0][0] == "data_registry"


def test_insert_single_record(tmp_path):
    db_path = str(tmp_path / "test_registry.db")
    mgr = MetadataManager(db_path)

    mgr.insert("daily", "2026-05-18", "data/daily/2026_05_18.parquet", 100)

    result = mgr.conn.execute(
        "SELECT api_name, trade_date, file_path, row_count FROM data_registry"
    ).fetchall()
    assert len(result) == 1
    assert result[0] == ("daily", "2026-05-18", "data/daily/2026_05_18.parquet", 100)


def test_upsert_overwrites_existing(tmp_path):
    db_path = str(tmp_path / "test_registry.db")
    mgr = MetadataManager(db_path)

    mgr.insert("stock_basic", None, "data/stock_basic.parquet", 200)
    mgr.insert("stock_basic", None, "data/stock_basic.parquet", 300)

    result = mgr.conn.execute(
        "SELECT row_count FROM data_registry WHERE api_name='stock_basic'"
    ).fetchall()
    assert len(result) == 1
    assert result[0][0] == 300


def test_get_last_date_returns_none_when_empty(tmp_path):
    db_path = str(tmp_path / "test_registry.db")
    mgr = MetadataManager(db_path)

    assert mgr.get_last_date("daily") is None


def test_get_last_date_returns_max_date(tmp_path):
    db_path = str(tmp_path / "test_registry.db")
    mgr = MetadataManager(db_path)

    mgr.insert("daily", "2026-05-10", "data/daily/2026_05_10.parquet", 50)
    mgr.insert("daily", "2026-05-15", "data/daily/2026_05_15.parquet", 60)
    mgr.insert("daily", "2026-05-12", "data/daily/2026_05_12.parquet", 55)

    assert mgr.get_last_date("daily") == "2026-05-15"


def test_summary_returns_grouped_counts(tmp_path):
    db_path = str(tmp_path / "test_registry.db")
    mgr = MetadataManager(db_path)

    mgr.insert("daily", "2026-05-10", "data/daily/2026_05_10.parquet", 50)
    mgr.insert("daily", "2026-05-11", "data/daily/2026_05_11.parquet", 55)
    mgr.insert("stk_st", "2026-05-10", "data/stk_st/2026_05_10.parquet", 10)

    result = mgr.summary()

    rows = {r[0]: (r[1], r[2]) for r in result}
    assert rows["daily"] == (2, "2026-05-11")
    assert rows["stk_st"] == (1, "2026-05-10")
