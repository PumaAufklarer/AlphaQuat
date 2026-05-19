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

## Architecture: data fetching (`src/alpha_quat/data/`)

| Module | Purpose |
|--------|---------|
| `source.py` | `DataSource` ABC — declare api_name, partition_by ("none"\|"date"), fields, optional start_date |
| `fetcher.py` | Wraps tushare with retry (12×5s by default) |
| `writer.py` | `ParquetWriter` — `overwrite()` for full sources, `write()` for date-partitioned, `merge()` for multi-factor column joining |
| `metadata.py` | `MetadataManager` — duckdb `data/registry.db` tracks what's been pulled per API+date. Also `delete_since()` for rebuild |
| `pipeline.py` | `Pipeline` — dispatches full/incremental sources, reads trade_cal for date lists |
| `sources/` | Concrete DataSource subclasses (stock_basic, trade_cal, daily, daily_basic, stock_st) |

```
config.yaml → Config → Pipeline(DataSource, Fetcher, ParquetWriter, MetadataManager)
```

## Architecture: feature engineering (`src/alpha_quat/features/`)

| Module | Purpose |
|--------|---------|
| `factor.py` | `Factor` dataclass (name, expression, category, depends_on). `compile()` translates DSL to DuckDB SQL via regex |
| `registry.py` | `FactorRegistry` — topo sort (Kahn's BFS), cycle detection, `min_lookback()` |
| `engine.py` | `FeatureEngine` — builds 2-CTE DuckDB query (ts + rank), computes per batch of dates |
| `pipeline.py` | `FeaturePipeline` — date scheduling with incremental/rebuild/--since, batch of 20 dates |
| `alphasets/` | Pre-built factor sets. `alpha158.py` → `build_alpha158()` returns a `FactorRegistry` |

**Computation model:**

```
raw (base CTE: read daily/*.parquet with vwap column)
  │
  ▼
_ts CTE (all time-series + RANK/QUANTILE inner exprs, WINDOW w_time)
  │
  ▼
_rank CTE (all RANK()/NTILE() OVER expressions, no WINDOW)
  │
  ▼
SELECT ts_code, trade_date, <all factors> WHERE trade_date BETWEEN min AND max
```

Only **2 CTEs** regardless of factor count — never create one CTE per factor.

## CLI

```
uv run alpha-quat                                     # same as "fetch" (backward compat)
uv run alpha-quat fetch -s daily,trade_cal            # pull raw data
uv run alpha-quat feature                             # compute alpha158 (incremental)
uv run alpha-quat feature --rebuild                   # recompute from scratch
uv run alpha-quat feature --since 20260101            # recompute from date onward
uv run alpha-quat --summary                           # show registry status (data + features)
```

## Data output

```
data/
├── registry.db                  # shared metadata: api_name="daily"/"alpha158"/...
├── stock_basic.parquet          # overwritten each run
├── trade_cal.parquet            # overwritten each run
├── daily/YYYY_MM_DD.parquet     # raw input for features
├── daily_basic/YYYY_MM_DD.parquet
├── stock_st/YYYY_MM_DD.parquet
└── features/YYYY_MM_DD.parquet  # wide table: ts_code, trade_date, +158+ factor columns
```

## Gotchas

- **Stale `__pycache__`** — if behavior seems wrong after a code change, `find . -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +`
- **`start_date` on DataSource** — for APIs with limited history (e.g. `stock_st` only has data from 2016), set `start_date = "20160101"` on the subclass to skip useless queries
- **Future dates excluded** — Pipeline auto-filters trade_cal dates > today
- **Tushare rate limits** — minute-level, Fetcher defaults to 12 retries at 5s intervals (60s total)
- **`stock_st` is the API name** — tushare pro uses `stock_st` (not `stk_st`)

### Feature engineering gotchas

- **`$vwap` is a column, not inline SQL** — vwap is computed in the engine's base CTE (`amount / NULLIF(vol, 0) AS vwap`). `compile()` replaces `$vwap` → `vwap` (plain column ref). Never expand `$vwap` to inline SQL — it breaks downstream `\w+` regex patterns.
- **RANK/QUANTILE must be two CTEs** — wrapping a time-series expression inside RANK (e.g. `RANK(REF(...))`) requires an inner CTE (with `w_time` window) and an outer CTE (pure ranking, no `w_time`). DuckDB rejects nested window functions. The engine handles this automatically — do not attempt single-CTE for RANK/QUANTILE factors.
- **Batch processing** — pipeline processes dates in batches of 20. Each batch builds the CTE chain once and filters by date range (`WHERE trade_date BETWEEN`). Do not change to per-date processing without re-benchmarking (single date: 0.86s, batch of 20: ~3s).
- **Factor dependencies** — alpha158 factors only depend on `$raw` fields, not other factors. The topo sort exists for future factor sets that build chains of derived factors. Adding a cycle between factors raises `ValueError`.

## Adding a new data source

1. Create `src/alpha_quat/data/sources/<name>.py` as a DataSource subclass with api_name, partition_by, fields
2. Create `tests/test_sources/test_<name>.py` (test api_name, partition_by, get_params, path_for)
3. Register in `cli.py` `ALL_SOURCES` dict
4. If data doesn't go back to 1990, set `start_date` on the class

## Adding a new factor set

1. Create `src/alpha_quat/features/alphasets/<name>.py` with a `build_<name>() -> FactorRegistry` function
2. Add to `cli.py` `ALL_FEATURE_SETS` dict as `"name": "alpha_quat.features.alphasets.<name>:build_<name>"`
3. Define `Factor` instances using the DSL: `$close`, `$volume`, `$amount`, `$open`, `$high`, `$low`, `$vwap` for raw fields; `REF(f, N)`, `MEAN(f, N)`, `STD(f, N)`, `SUM(f, N)`, `MAX(f, N)`, `MIN(f, N)`, `CORR(f1, f2, N)`, `DELTA(f, N)`, `RANK(f)`, `QUANTILE(f, N)` for operators
4. Create `tests/test_features/test_<name>.py` verifying: all compile, no cycles, valid deps, lookback consistent
