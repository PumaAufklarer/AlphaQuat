# Data Fetcher Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified data fetching framework that pulls stock data from tushare, stores it as Parquet files, and manages metadata in duckdb.

**Architecture:** Six components — Config reads config.yaml; DataSource (abstract base) defines per-API behavior; Fetcher wraps tushare with retry; ParquetWriter handles partitioned file output; MetadataManager tracks pulls in duckdb; Pipeline orchestrates everything.

**Tech Stack:** Python 3.11, tushare, duckdb, pandas/pyarrow, PyYAML

---

### Task 0: Dependencies and .gitignore

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 0.1: Add duckdb and PyYAML dependencies**

```bash
uv add duckdb pyyaml
```

Expected: `duckdb` and `pyyaml` appear in pyproject.toml dependencies and uv.lock is updated.

- [ ] **Step 0.2: Add config.yaml and data/ to .gitignore**

Read `.gitignore` first, then add:
```
# Project config (contains secrets)
config.yaml

# Data files
data/
```

Expected: `config.yaml` and `data/` are gitignored.

- [ ] **Step 0.3: Install and verify**

```bash
uv run python -c "import duckdb; import yaml; print('ok')"
```

Expected: `ok`

- [ ] **Step 0.4: Commit**

```bash
git add pyproject.toml uv.lock .gitignore
git commit -m "chore: add duckdb, pyyaml deps; gitignore config and data"
```

---

### Task 1: Config Module

**Files:**
- Create: `src/alpha_quat/__init__.py`
- Create: `src/alpha_quat/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1.1: Write the failing test**

```bash
mkdir -p src/alpha_quat tests
```

Create `src/alpha_quat/__init__.py`:
```python
"""alpha-quat — quantitative stock selection framework."""
```

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:
```python
"""Shared test fixtures."""
```

Create `tests/test_config.py`:
```python
"""Tests for config module."""

from pathlib import Path

from alpha_quat.config import Config


def test_config_reads_token_and_data_dir(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("""
tushare:
  token: "test_token_123"
data:
  dir: "/tmp/test_data"
""")
    config = Config.from_yaml(str(yaml_path))

    assert config.token == "test_token_123"
    assert config.data_dir == Path("/tmp/test_data")


def test_config_default_data_dir(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("""
tushare:
  token: "abc"
""")
    config = Config.from_yaml(str(yaml_path))

    assert config.token == "abc"
    assert config.data_dir == Path("data")


def test_config_missing_token_raises(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("""
data:
  dir: "./data"
""")
    try:
        Config.from_yaml(str(yaml_path))
        assert False, "Expected KeyError"
    except KeyError:
        pass
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
uv run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha_quat.config'`

- [ ] **Step 1.3: Write minimal implementation**

Create `src/alpha_quat/config.py`:
```python
"""Configuration from YAML file."""

from pathlib import Path

import yaml


class Config:
    token: str
    data_dir: Path

    def __init__(self, token: str, data_dir: Path | str = "data"):
        self.token = token
        self.data_dir = Path(data_dir)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            token=raw["tushare"]["token"],
            data_dir=raw.get("data", {}).get("dir", "data"),
        )
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 3 PASS

- [ ] **Step 1.5: Commit**

```bash
git add src/alpha_quat/__init__.py src/alpha_quat/config.py tests/ tests/conftest.py
git commit -m "feat: add Config module with YAML loader"
```

---

### Task 2: Fetcher Module

**Files:**
- Create: `src/alpha_quat/data/__init__.py`
- Create: `src/alpha_quat/data/fetcher.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 2.1: Write the failing test**

```bash
mkdir -p src/alpha_quat/data
```

Create `src/alpha_quat/data/__init__.py` (empty file).

Create `tests/test_fetcher.py`:
```python
"""Tests for fetcher module."""

import time

import pandas as pd
import pytest

from alpha_quat.data.fetcher import Fetcher, FetchError


class FakeProApi:
    def __init__(self, results=None, fail_count=0):
        self.results = results or [pd.DataFrame({"col": [1]})]
        self.fail_count = fail_count
        self.call_count = 0
        self.last_api = None
        self.last_params = None

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
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
uv run pytest tests/test_fetcher.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha_quat.data.fetcher'`

- [ ] **Step 2.3: Write minimal implementation**

Create `src/alpha_quat/data/fetcher.py`:
```python
"""Tushare API wrapper with retry logic."""

import time

import tushare as ts


class FetchError(Exception):
    pass


class Fetcher:
    def __init__(
        self,
        token: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        ts.set_token(token)
        self._api = ts.pro_api(token)

    def query(self, api_name: str, **params):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._api.query(api_name, **params)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise FetchError(
            f"Failed after {self.max_retries} retries: {last_error}"
        )
```

- [ ] **Step 2.4: Run test to verify it passes**

```bash
uv run pytest tests/test_fetcher.py -v
```

Expected: 5 PASS

- [ ] **Step 2.5: Commit**

```bash
git add src/alpha_quat/data/ tests/test_fetcher.py
git commit -m "feat: add Fetcher with tushare query and retry logic"
```

---

### Task 3: MetadataManager Module

**Files:**
- Create: `src/alpha_quat/data/metadata.py`
- Create: `tests/test_metadata.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_metadata.py`:
```python
"""Tests for metadata manager module."""

import time

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
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
uv run pytest tests/test_metadata.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha_quat.data.metadata'`

- [ ] **Step 3.3: Write minimal implementation**

Create `src/alpha_quat/data/metadata.py`:
```python
"""DuckDB metadata manager for data registry."""

import duckdb


class MetadataManager:
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)
        self._init_table()

    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_registry (
                id          INTEGER PRIMARY KEY,
                api_name    VARCHAR NOT NULL,
                trade_date  DATE,
                file_path   VARCHAR NOT NULL,
                row_count   INTEGER NOT NULL,
                pull_time   TIMESTAMP DEFAULT now(),
                UNIQUE(api_name, trade_date)
            )
        """)

    def insert(
        self,
        api_name: str,
        trade_date: str | None,
        file_path: str,
        row_count: int,
    ):
        date_val = f"'{trade_date}'" if trade_date else "NULL"
        self.conn.execute(f"""
            INSERT OR REPLACE INTO data_registry
                (api_name, trade_date, file_path, row_count, pull_time)
            VALUES ('{api_name}', {date_val}, '{file_path}', {row_count}, now())
        """)

    def get_last_date(self, api_name: str) -> str | None:
        result = self.conn.execute(
            "SELECT MAX(trade_date) FROM data_registry WHERE api_name = ?",
            [api_name],
        ).fetchone()
        if result and result[0]:
            return str(result[0])
        return None

    def summary(self):
        return self.conn.execute(
            "SELECT api_name, COUNT(*), MAX(trade_date) "
            "FROM data_registry GROUP BY api_name ORDER BY api_name"
        ).fetchall()
```

- [ ] **Step 3.4: Run test to verify it passes**

```bash
uv run pytest tests/test_metadata.py -v
```

Expected: 6 PASS

- [ ] **Step 3.5: Commit**

```bash
git add src/alpha_quat/data/metadata.py tests/test_metadata.py
git commit -m "feat: add MetadataManager with duckdb registry"
```

---

### Task 4: ParquetWriter Module

**Files:**
- Create: `src/alpha_quat/data/writer.py`
- Create: `tests/test_writer.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_writer.py`:
```python
"""Tests for Parquet writer module."""

import pandas as pd
import pytest

from alpha_quat.data.writer import ParquetWriter


def test_overwrite_writes_parquet_file(tmp_path):
    writer = ParquetWriter()
    df = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "name": ["平安银行", "万科A"]})
    path = tmp_path / "stock_basic.parquet"

    writer.overwrite(df, path)

    assert path.exists()
    result = pd.read_parquet(path)
    assert len(result) == 2
    assert list(result["ts_code"]) == ["000001.SZ", "000002.SZ"]


def test_overwrite_replaces_existing_file(tmp_path):
    writer = ParquetWriter()
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2, 3]})
    path = tmp_path / "test.parquet"

    writer.overwrite(df1, path)
    writer.overwrite(df2, path)

    result = pd.read_parquet(path)
    assert len(result) == 2


def test_write_creates_date_partitioned_file(tmp_path):
    writer = ParquetWriter()
    df = pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]})
    base_dir = tmp_path / "daily"

    writer.write(df, base_dir, trade_date="2026_05_18")

    expected = base_dir / "2026_05_18.parquet"
    assert expected.exists()
    result = pd.read_parquet(expected)
    assert len(result) == 1
    assert result.iloc[0]["close"] == 10.5


def test_write_creates_parent_directory(tmp_path):
    writer = ParquetWriter()
    df = pd.DataFrame({"a": [1]})
    base_dir = tmp_path / "nested" / "deep" / "daily"

    writer.write(df, base_dir, trade_date="2026_05_18")

    expected = base_dir / "2026_05_18.parquet"
    assert expected.exists()
```

- [ ] **Step 4.2: Run test to verify it fails**

```bash
uv run pytest tests/test_writer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha_quat.data.writer'`

- [ ] **Step 4.3: Write minimal implementation**

Create `src/alpha_quat/data/writer.py`:
```python
"""Parquet file writer with partitioning support."""

from pathlib import Path

import pandas as pd


class ParquetWriter:
    def overwrite(self, df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def write(self, df: pd.DataFrame, base_dir: Path, trade_date: str):
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / f"{trade_date}.parquet"
        df.to_parquet(file_path, index=False)
```

- [ ] **Step 4.4: Run test to verify it passes**

```bash
uv run pytest tests/test_writer.py -v
```

Expected: 4 PASS

- [ ] **Step 4.5: Commit**

```bash
git add src/alpha_quat/data/writer.py tests/test_writer.py
git commit -m "feat: add ParquetWriter with overwrite and date-partitioned write"
```

---

### Task 5: DataSource Base Class

**Files:**
- Create: `src/alpha_quat/data/source.py`
- Create: `tests/test_source.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_source.py`:
```python
"""Tests for DataSource base class."""

from abc import ABC
from pathlib import Path

import pytest

from alpha_quat.data.source import DataSource


class FakeFullSource(DataSource):
    api_name = "full_test"
    partition_by = "none"
    fields = "ts_code,name"

    def get_params(self, trade_date=None):
        return {"list_status": "L"}


class FakeDateSource(DataSource):
    api_name = "date_test"
    partition_by = "date"
    fields = "ts_code,close"

    def get_params(self, trade_date):
        return {"trade_date": trade_date}


def test_full_source_path_for():
    src = FakeFullSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/full_test.parquet")


def test_date_source_path_for():
    src = FakeDateSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/date_test/2026_05_18.parquet")


def test_full_source_path_for_ignores_trade_date():
    src = FakeFullSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/full_test.parquet")


def test_date_source_path_for_no_date_returns_dir():
    src = FakeDateSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/date_test")


def test_datasource_is_abstract():
    with pytest.raises(TypeError):
        DataSource()  # type: ignore
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
uv run pytest tests/test_source.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha_quat.data.source'`

- [ ] **Step 5.3: Write minimal implementation**

Create `src/alpha_quat/data/source.py`:
```python
"""Abstract base class for tushare data sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal


class DataSource(ABC):
    api_name: str
    partition_by: Literal["none", "date"]
    fields: str

    @abstractmethod
    def get_params(self, trade_date: str | None = None) -> dict:
        ...

    def path_for(
        self, data_dir: Path, trade_date: str | None = None
    ) -> Path:
        if self.partition_by == "none":
            return data_dir / f"{self.api_name}.parquet"
        if trade_date is None:
            return data_dir / self.api_name
        return data_dir / self.api_name / f"{trade_date}.parquet"
```

- [ ] **Step 5.4: Run test to verify it passes**

```bash
uv run pytest tests/test_source.py -v
```

Expected: 5 PASS

- [ ] **Step 5.5: Commit**

```bash
git add src/alpha_quat/data/source.py tests/test_source.py
git commit -m "feat: add DataSource abstract base class"
```

---

### Task 6: Full Coverage Data Sources (stock_basic, trade_cal)

**Files:**
- Create: `src/alpha_quat/data/sources/__init__.py`
- Create: `src/alpha_quat/data/sources/stock_basic.py`
- Create: `src/alpha_quat/data/sources/trade_cal.py`
- Create: `tests/test_sources/__init__.py`
- Create: `tests/test_sources/test_stock_basic.py`
- Create: `tests/test_sources/test_trade_cal.py`

- [ ] **Step 6.1: Write the failing tests**

```bash
mkdir -p src/alpha_quat/data/sources tests/test_sources
```

Create `src/alpha_quat/data/sources/__init__.py` (empty file).

Create `tests/test_sources/__init__.py` (empty file).

Create `tests/test_sources/test_stock_basic.py`:
```python
"""Tests for StockBasicSource."""

from pathlib import Path

from alpha_quat.data.sources.stock_basic import StockBasicSource


def test_stock_basic_api_name():
    src = StockBasicSource()
    assert src.api_name == "stock_basic"


def test_stock_basic_partition_by():
    src = StockBasicSource()
    assert src.partition_by == "none"


def test_stock_basic_get_params():
    src = StockBasicSource()
    params = src.get_params()
    assert params == {"list_status": "L"}


def test_stock_basic_path_for():
    src = StockBasicSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/stock_basic.parquet")
```

Create `tests/test_sources/test_trade_cal.py`:
```python
"""Tests for TradeCalSource."""

from pathlib import Path

from alpha_quat.data.sources.trade_cal import TradeCalSource


def test_trade_cal_api_name():
    src = TradeCalSource()
    assert src.api_name == "trade_cal"


def test_trade_cal_partition_by():
    src = TradeCalSource()
    assert src.partition_by == "none"


def test_trade_cal_get_params():
    src = TradeCalSource()
    params = src.get_params()
    assert params == {"exchange": "SSE"}


def test_trade_cal_path_for():
    src = TradeCalSource()
    path = src.path_for(data_dir=Path("/data"))
    assert path == Path("/data/trade_cal.parquet")
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sources/test_stock_basic.py tests/test_sources/test_trade_cal.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 6.3: Write minimal implementation**

Create `src/alpha_quat/data/sources/stock_basic.py`:
```python
"""Tushare stock_basic API data source."""

from alpha_quat.data.source import DataSource


class StockBasicSource(DataSource):
    api_name = "stock_basic"
    partition_by = "none"
    fields = "ts_code,symbol,name,area,industry,market,list_status,list_date"

    def get_params(self, trade_date=None):
        return {"list_status": "L"}
```

Create `src/alpha_quat/data/sources/trade_cal.py`:
```python
"""Tushare trade_cal API data source."""

from alpha_quat.data.source import DataSource


class TradeCalSource(DataSource):
    api_name = "trade_cal"
    partition_by = "none"
    fields = "exchange,cal_date,is_open,pretrade_date"

    def get_params(self, trade_date=None):
        return {"exchange": "SSE"}
```

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sources/test_stock_basic.py tests/test_sources/test_trade_cal.py -v
```

Expected: 8 PASS

- [ ] **Step 6.5: Commit**

```bash
git add src/alpha_quat/data/sources/ tests/test_sources/
git commit -m "feat: add StockBasicSource and TradeCalSource"
```

---

### Task 7: Incremental Data Sources (stk_st, daily, daily_basic)

**Files:**
- Create: `src/alpha_quat/data/sources/stk_st.py`
- Create: `src/alpha_quat/data/sources/daily.py`
- Create: `src/alpha_quat/data/sources/daily_basic.py`
- Create: `tests/test_sources/test_stk_st.py`
- Create: `tests/test_sources/test_daily.py`
- Create: `tests/test_sources/test_daily_basic.py`

- [ ] **Step 7.1: Write the failing tests**

Create `tests/test_sources/test_stk_st.py`:
```python
"""Tests for StkStSource."""

from pathlib import Path

from alpha_quat.data.sources.stk_st import StkStSource


def test_stk_st_api_name():
    src = StkStSource()
    assert src.api_name == "stk_st"


def test_stk_st_partition_by():
    src = StkStSource()
    assert src.partition_by == "date"


def test_stk_st_get_params():
    src = StkStSource()
    params = src.get_params(trade_date="20260115")
    assert params == {"trade_date": "20260115"}


def test_stk_st_path_for():
    src = StkStSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/stk_st/2026_05_18.parquet")
```

Create `tests/test_sources/test_daily.py`:
```python
"""Tests for DailySource."""

from pathlib import Path

from alpha_quat.data.sources.daily import DailySource


def test_daily_api_name():
    src = DailySource()
    assert src.api_name == "daily"


def test_daily_partition_by():
    src = DailySource()
    assert src.partition_by == "date"


def test_daily_get_params():
    src = DailySource()
    params = src.get_params(trade_date="20260115")
    assert params == {"trade_date": "20260115"}


def test_daily_path_for():
    src = DailySource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/daily/2026_05_18.parquet")
```

Create `tests/test_sources/test_daily_basic.py`:
```python
"""Tests for DailyBasicSource."""

from pathlib import Path

from alpha_quat.data.sources.daily_basic import DailyBasicSource


def test_daily_basic_api_name():
    src = DailyBasicSource()
    assert src.api_name == "daily_basic"


def test_daily_basic_partition_by():
    src = DailyBasicSource()
    assert src.partition_by == "date"


def test_daily_basic_get_params():
    src = DailyBasicSource()
    params = src.get_params(trade_date="20260115")
    assert params == {"trade_date": "20260115"}


def test_daily_basic_path_for():
    src = DailyBasicSource()
    path = src.path_for(data_dir=Path("/data"), trade_date="2026_05_18")
    assert path == Path("/data/daily_basic/2026_05_18.parquet")
```

- [ ] **Step 7.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sources/test_stk_st.py tests/test_sources/test_daily.py tests/test_sources/test_daily_basic.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 7.3: Write minimal implementation**

Create `src/alpha_quat/data/sources/stk_st.py`:
```python
"""Tushare stk_st API data source (ST stock list)."""

from alpha_quat.data.source import DataSource


class StkStSource(DataSource):
    api_name = "stk_st"
    partition_by = "date"
    fields = "ts_code,name,type"

    def get_params(self, trade_date=None):
        return {"trade_date": trade_date}
```

Create `src/alpha_quat/data/sources/daily.py`:
```python
"""Tushare daily API data source."""

from alpha_quat.data.source import DataSource


class DailySource(DataSource):
    api_name = "daily"
    partition_by = "date"
    fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"

    def get_params(self, trade_date=None):
        return {"trade_date": trade_date}
```

Create `src/alpha_quat/data/sources/daily_basic.py`:
```python
"""Tushare daily_basic API data source."""

from alpha_quat.data.source import DataSource


class DailyBasicSource(DataSource):
    api_name = "daily_basic"
    partition_by = "date"
    fields = "ts_code,trade_date,total_mv,circ_mv,pe,pe_ttm,pb,turnover_rate,turnover_rate_f,volume_ratio"

    def get_params(self, trade_date=None):
        return {"trade_date": trade_date}
```

- [ ] **Step 7.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sources/test_stk_st.py tests/test_sources/test_daily.py tests/test_sources/test_daily_basic.py -v
```

Expected: 12 PASS

- [ ] **Step 7.5: Commit**

```bash
git add src/alpha_quat/data/sources/ tests/test_sources/
git commit -m "feat: add StkStSource, DailySource, DailyBasicSource"
```

---

### Task 8: Pipeline Module

**Files:**
- Create: `src/alpha_quat/data/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 8.1: Write the failing test**

Create `tests/test_pipeline.py`:
```python
"""Tests for Pipeline module."""

from pathlib import Path

import pandas as pd
import pytest

from alpha_quat.data.pipeline import Pipeline
from alpha_quat.data.fetcher import Fetcher
from alpha_quat.data.metadata import MetadataManager
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.sources.stock_basic import StockBasicSource
from alpha_quat.data.sources.daily import DailySource


class FakeFetcher:
    def __init__(self, calls=None):
        self.calls = calls or []
        self._idx = 0
        self.query_log = []

    def query(self, api_name, **params):
        if self._idx < len(self.calls):
            df = self.calls[self._idx]
        else:
            df = pd.DataFrame()
        self.query_log.append((api_name, params))
        self._idx += 1
        return df


def test_pipeline_runs_full_source(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    fetcher = FakeFetcher(calls=[
        pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["平安银行"]}),
    ])
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer)

    pipeline.run_full_source(StockBasicSource())

    parquet_path = data_dir / "stock_basic.parquet"
    assert parquet_path.exists()
    result = pd.read_parquet(parquet_path)
    assert len(result) == 1
    assert result.iloc[0]["ts_code"] == "000001.SZ"


def test_pipeline_run_incremental_source_no_prior_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    # Prepare trade_cal data
    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "cal_date": ["20260112", "20260113", "20260114", "20260115"],
        "is_open": [1, 1, 1, 0],
    }).to_parquet(trade_cal_path)

    fetcher = FakeFetcher(calls=[
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]}),
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]}),
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [11.0]}),
    ])
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer)

    results = pipeline.run_incremental_source(DailySource())

    assert results["success"] == 3
    assert results["failed"] == 0
    assert (data_dir / "daily" / "2026_01_12.parquet").exists()
    assert (data_dir / "daily" / "2026_01_13.parquet").exists()
    assert (data_dir / "daily" / "2026_01_14.parquet").exists()
    # 20260115 is not open, should be skipped
    assert not (data_dir / "daily" / "2026_01_15.parquet").exists()


def test_pipeline_run_incremental_source_with_prior_data(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "cal_date": ["20260112", "20260113", "20260114"],
        "is_open": [1, 1, 1],
    }).to_parquet(trade_cal_path)

    metadata = MetadataManager(db_path)
    metadata.insert("daily", "2026-01-12", "data/daily/2026_01_12.parquet", 50)

    fetcher = FakeFetcher(calls=[
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]}),
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [11.0]}),
    ])
    writer = ParquetWriter()
    pipeline = Pipeline(data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer)

    results = pipeline.run_incremental_source(DailySource())

    # Should skip 20260112 (already pulled), only pull 20260113-14
    assert results["success"] == 2
    assert not (data_dir / "daily" / "2026_01_12.parquet").exists()
    assert (data_dir / "daily" / "2026_01_13.parquet").exists()
    assert (data_dir / "daily" / "2026_01_14.parquet").exists()
    assert len(fetcher.query_log) == 2


def test_pipeline_incremental_skips_when_trade_cal_missing(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    fetcher = FakeFetcher()
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer)

    results = pipeline.run_incremental_source(DailySource())

    assert results["success"] == 0
    assert results["failed"] == 0
    assert results["message"] == "trade_cal.parquet not found"


def test_pipeline_run_all(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    # trade_cal parquet
    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "cal_date": ["20260112", "20260113"],
        "is_open": [1, 1],
    }).to_parquet(trade_cal_path)

    calls = [
        # stock_basic full
        pd.DataFrame({"ts_code": ["000001.SZ"]}),
        # trade_cal full (both dates so daily can pull them)
        pd.DataFrame({"cal_date": ["20260112", "20260113"], "is_open": [1, 1]}),
        # daily incremental - 2 days
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]}),
        pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.5]}),
    ]
    fetcher = FakeFetcher(calls=calls)
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer)

    pipeline.run([StockBasicSource(), TradeCalSource(), DailySource()])

    assert (data_dir / "stock_basic.parquet").exists()
    assert (data_dir / "daily" / "2026_01_12.parquet").exists()
    assert (data_dir / "daily" / "2026_01_13.parquet").exists()


def test_pipeline_error_handling_continues_after_failure(tmp_path):
    data_dir = tmp_path / "data"
    db_path = str(tmp_path / "registry.db")

    trade_cal_path = data_dir / "trade_cal.parquet"
    trade_cal_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "cal_date": ["20260112", "20260113", "20260114"],
        "is_open": [1, 1, 1],
    }).to_parquet(trade_cal_path)

    class FailingFetcher:
        def __init__(self):
            self.call_count = 0
            self.query_log = []

        def query(self, api_name, **params):
            self.call_count += 1
            self.query_log.append((api_name, params))
            if self.call_count == 2:
                raise RuntimeError("network failure")
            return pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})

    fetcher = FailingFetcher()
    metadata = MetadataManager(db_path)
    writer = ParquetWriter()
    pipeline = Pipeline(data_dir=data_dir, fetcher=fetcher, metadata=metadata, writer=writer)

    results = pipeline.run_incremental_source(DailySource())

    assert results["success"] == 2
    assert results["failed"] == 1
    assert len(results["errors"]) == 1
    assert "20260113" in results["errors"][0]
```

- [ ] **Step 8.2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'alpha_quat.data.pipeline'`

- [ ] **Step 8.3: Write minimal implementation**

Create `src/alpha_quat/data/pipeline.py`:
```python
"""Pipeline that orchestrates data fetching, writing, and metadata tracking."""

import logging
from pathlib import Path

import pandas as pd

from alpha_quat.data.source import DataSource

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        data_dir: Path,
        fetcher,
        metadata,
        writer,
    ):
        self.data_dir = data_dir
        self.fetcher = fetcher
        self.metadata = metadata
        self.writer = writer

    def run(self, sources: list[DataSource]):
        for source in sources:
            if source.partition_by == "none":
                self.run_full_source(source)
            else:
                self.run_incremental_source(source)

    def run_full_source(self, source: DataSource):
        params = source.get_params()
        df = self.fetcher.query(source.api_name, fields=source.fields, **params)
        path = source.path_for(self.data_dir)
        self.writer.overwrite(df, path)
        self.metadata.insert(
            api_name=source.api_name,
            trade_date=None,
            file_path=str(path),
            row_count=len(df),
        )
        logger.info(f"[{source.api_name}] pulled {len(df)} rows → {path}")

    def run_incremental_source(self, source: DataSource) -> dict:
        trade_cal_path = self.data_dir / "trade_cal.parquet"
        if not trade_cal_path.exists():
            return {"success": 0, "failed": 0, "message": "trade_cal.parquet not found", "errors": []}

        cal_df = pd.read_parquet(trade_cal_path)
        open_dates = sorted(
            cal_df[cal_df["is_open"] == 1]["cal_date"].astype(str).tolist()
        )

        last_date = self.metadata.get_last_date(source.api_name)

        if last_date:
            # Parse date formats (tushare uses 2026YMMDD or 2026-MM-DD)
            last_date_clean = last_date.replace("-", "")
            pending = [d for d in open_dates if d.replace("-", "") > last_date_clean]
        else:
            pending = open_dates

        success, failed = 0, 0
        errors = []

        for trade_date in pending:
            try:
                params = source.get_params(trade_date=trade_date)
                df = self.fetcher.query(
                    source.api_name, fields=source.fields, **params
                )
                date_file = trade_date.replace("-", "_")  # 2026YMMDD → 2026_MM_DD
                base_dir = source.path_for(self.data_dir, trade_date=None)
                self.writer.write(df, base_dir, trade_date=date_file)
                self.metadata.insert(
                    api_name=source.api_name,
                    trade_date=trade_date,
                    file_path=str(base_dir / f"{date_file}.parquet"),
                    row_count=len(df),
                )
                success += 1
                logger.info(f"[{source.api_name}] {trade_date}: {len(df)} rows")
            except Exception as e:
                failed += 1
                errors.append(f"[{source.api_name}] {trade_date}: {e}")
                logger.error(f"[{source.api_name}] {trade_date}: {e}")

        return {"success": success, "failed": failed, "errors": errors}
```

- [ ] **Step 8.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: 7 PASS

- [ ] **Step 8.5: Commit**

```bash
git add src/alpha_quat/data/pipeline.py tests/test_pipeline.py
git commit -m "feat: add Pipeline orchestrating full and incremental data pulls"
```

---

### Task 9: CLI Entrypoint

**Files:**
- Create: `src/alpha_quat/cli.py`
- Modify: `pyproject.toml` (add console_scripts entry point)

- [ ] **Step 9.1: Add entry point to pyproject.toml**

Read `pyproject.toml` and add the `[project.scripts]` section:

```toml
[project.scripts]
alpha-quat = "alpha_quat.cli:main"
```

- [ ] **Step 9.2: Write the CLI module**

Create `src/alpha_quat/cli.py`:
```python
"""CLI entry point for alpha-quat data fetching."""

import argparse
import logging

from alpha_quat.config import Config
from alpha_quat.data.fetcher import Fetcher
from alpha_quat.data.metadata import MetadataManager
from alpha_quat.data.pipeline import Pipeline
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.sources.stock_basic import StockBasicSource
from alpha_quat.data.sources.trade_cal import TradeCalSource
from alpha_quat.data.sources.stk_st import StkStSource
from alpha_quat.data.sources.daily import DailySource
from alpha_quat.data.sources.daily_basic import DailyBasicSource

ALL_SOURCES = {
    "stock_basic": StockBasicSource,
    "trade_cal": TradeCalSource,
    "stk_st": StkStSource,
    "daily": DailySource,
    "daily_basic": DailyBasicSource,
}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch stock data from tushare and store as Parquet"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        choices=list(ALL_SOURCES.keys()) + ["all"],
        default=["all"],
        help="Data sources to pull (default: all)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show data registry summary and exit"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = Config.from_yaml(args.config)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    db_path = config.data_dir / "registry.db"
    metadata = MetadataManager(str(db_path))

    if args.summary:
        rows = metadata.summary()
        if rows:
            print(f"{'api_name':<15} {'count':<8} {'max_date'}")
            print("-" * 40)
            for row in rows:
                print(f"{row[0]:<15} {row[1]:<8} {row[2] or 'N/A'}")
        else:
            print("No data in registry. Run a pull first.")
        return

    fetcher = Fetcher(token=config.token)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=config.data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )

    names = (
        list(ALL_SOURCES.keys()) if "all" in args.sources else args.sources
    )
    sources = [ALL_SOURCES[name]() for name in names]
    pipeline.run(sources)

    # Print summary after run
    print()
    summary_rows = metadata.summary()
    if summary_rows:
        print(f"{'api_name':<15} {'files':<8} {'max_date'}")
        print("-" * 40)
        for row in summary_rows:
            print(f"{row[0]:<15} {row[1]:<8} {row[2] or 'N/A'}")
```

- [ ] **Step 9.3: Test CLI help output**

```bash
uv run alpha-quat --help
```

Expected: Shows help text with --config, --sources, --summary, --verbose options.

- [ ] **Step 9.5: Commit**

```bash
git add src/alpha_quat/cli.py pyproject.toml
git commit -m "feat: add CLI entrypoint with source selection and summary"
```

---

### Task 10: Final Verification

**Files:** None (verification only)

- [ ] **Step 10.1: Lint and format**

```bash
uv run ruff format .
```

Expected: All files formatted.

- [ ] **Step 10.2: Fix and check lint**

```bash
ruff check --fix .
```

Expected: 0 errors.

- [ ] **Step 10.3: Typecheck**

```bash
uv run pyright
```

Expected: 0 errors (or minimal, fix any issues).

- [ ] **Step 10.4: Run all tests**

```bash
uv run pytest --cov=src -v
```

Expected: All tests PASS.

- [ ] **Step 10.5: Review and commit any fixes**

```bash
git add -u && git commit -m "chore: fix lint/typecheck issues after implementation"
```
