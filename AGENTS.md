# AGENTS.md — alpha-quat

## Toolchain

Always use `uv` — pip/poetry won't work here.

```
uv run pytest --cov=src          # test
uv run ruff format $FILE         # format
uv run ruff check --fix $FILE    # lint
uv run pyright                   # typecheck
```

## Verification order

```
uv run ruff format . && uv run ruff check --fix . && uv run pyright && uv run pytest --cov=src
```

## Project layout

- Package: `alpha-quat` (PyPI) → import `alpha_quat`
- Source: `src/alpha_quat/`, tests: `tests/`
- `config.yaml` (gitignored) required at runtime:
  ```yaml
  tushare:
    token: "xxx"
  data:
    dir: "./data"
  ```

## Architecture: data fetching framework

Core pipeline (`src/alpha_quat/data/`):

| Module | Purpose |
|--------|---------|
| `source.py` | `DataSource` ABC — declare api_name, partition_by ("none"|"date"), fields, optional start_date |
| `fetcher.py` | Wraps tushare with retry (12×5s by default) |
| `writer.py` | `ParquetWriter` — `overwrite()` for full sources, `write()` for date-partitioned |
| `metadata.py` | `MetadataManager` — duckdb `data/registry.db` tracks what's been pulled per API+date |
| `pipeline.py` | `Pipeline` — dispatches full/incremental sources, reads trade_cal for date lists |
| `sources/` | Concrete DataSource subclasses (stock_basic, trade_cal, daily, daily_basic, stock_st) |

```
config.yaml → Config → Pipeline(DataSource, Fetcher, ParquetWriter, MetadataManager)
```

Data output:
```
data/
├── stock_basic.parquet        # overwritten each run
├── trade_cal.parquet          # overwritten each run
├── daily/YYYY_MM_DD.parquet   # incremental, one per trade date
├── daily_basic/YYYY_MM_DD.parquet
└── stock_st/YYYY_MM_DD.parquet
```

CLI: `uv run alpha-quat [-s source1 source2] [--summary]`

## Gotchas

- **Stale `__pycache__`** — if behavior seems wrong after a code change, `find . -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +`
- **`start_date` on DataSource** — for APIs with limited history (e.g. `stock_st` only has data from 2016), set `start_date = "20160101"` on the subclass to skip useless queries
- **Future dates excluded** — Pipeline auto-filters trade_cal dates > today
- **Tushare rate limits** — minute-level, Fetcher defaults to 12 retries at 5s intervals (60s total)
- **`stock_st` is the API name** — tushare pro uses `stock_st` (not `stk_st`)

## Adding a new data source

1. Create `src/alpha_quat/data/sources/<name>.py` as a DataSource subclass with api_name, partition_by, fields
2. Create `tests/test_sources/test_<name>.py` (test api_name, partition_by, get_params, path_for)
3. Register in `cli.py` `ALL_SOURCES` dict
4. If data doesn't go back to 1990, set `start_date` on the class
