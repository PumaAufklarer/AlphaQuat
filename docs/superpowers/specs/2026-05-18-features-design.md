# Feature Engineering Module — Design Spec

## Overview

新增 `src/alpha_quat/features/` 模块，实现特征工程管线：

- 从已采集的原始 daily/daily_basic 数据中计算派生因子
- 首版严格复刻 Qlib alpha158（158 个因子），后续可扩展 alpha360 等
- 输出到 `data/features/YYYY_MM_DD.parquet`（逐日分片，全市场宽表）
- 独立 FeaturePipeline（与现有的数据采集 Pipeline 平行），复用 ParquetWriter 和 MetadataManager

## Design Decisions

| 决策 | 选择 |
|------|------|
| 架构关系 | 独立 FeaturePipeline，不复用 DataSource/Fetcher，仅复用 Writer/Metadata |
| 计算引擎 | DuckDB SQL，窗口函数原生支持 |
| 因子定义 | SQL 表达式 + 正则编译 DSL |
| 增量策略 | 默认增量 + `--rebuild` / `--since` 回填 |
| 输出格式 | 逐日单张宽表（ts_code + trade_date + 158 因子） |
| 多因子集 | 同目录合并写入，通过 `merge()` 方法追加列 |

## Architecture

```
CLI: alpha-quat feature [-f alpha158] [--rebuild] [--since DATE]

     FeaturePipeline
     ├── FactorRegistry  → Factor(name, expression, category, depends_on)
     ├── FeatureEngine    → DuckDB CTE compiler + executor
     ├── ParquetWriter    → write() / merge() (复用)
     └── MetadataManager  → insert() / get_last_date() (复用)

     Raw Data (read):      daily/*.parquet + daily_basic/*.parquet
     Output (write):       data/features/YYYY_MM_DD.parquet
```

**不变更**：`source.py`, `fetcher.py`, `pipeline.py`（数据采集），所有 `sources/*`

## Module Layout

```
src/alpha_quat/features/
├── __init__.py
├── factor.py              # Factor dataclass + compile()
├── registry.py            # FactorRegistry (topo_sort, min_lookback, cycle_detect)
├── engine.py              # FeatureEngine (DuckDB CTE builder)
├── pipeline.py            # FeaturePipeline (schedule + incremental)
└── alphasets/
    ├── __init__.py
    └── alpha158.py        # 158 Factor definitions

tests/test_features/
├── __init__.py
├── test_factor.py
├── test_registry.py
├── test_engine.py
├── test_pipeline.py
└── test_alpha158.py       # Factor definition integrity
```

## Factor System

### Factor Dataclass

```python
@dataclass
class Factor:
    name: str           # Qlib original name, e.g. "KMID", "KLEN"
    expression: str     # mini DSL expression
    category: str       # momentum / volatility / volume / price / ...
    depends_on: list[str]  # auto-parsed from expression
```

### Expression DSL → DuckDB SQL

Factors are defined in a mini expression language that compiles to DuckDB window functions via regex substitution.

| DSL Syntax | Meaning | DuckDB SQL |
|-----------|---------|------------|
| `$close` | Raw close price | `close` |
| `$open`, `$high`, `$low` | OHLC prices | Column names from daily |
| `$volume`, `$amount` | Volume/amount | Column names from daily |
| `$vwap` | VWAP | `$amount / NULLIF($volume, 0)` |
| `REF(f, N)` | Value N days ago | `LAG(f, N) OVER w_time` |
| `MEAN(f, N)` | N-day rolling mean | `AVG(f) OVER (w_time ROWS N-1 PRECEDING)` |
| `STD(f, N)` | N-day rolling std | `STDDEV_SAMP(f) OVER (w_time ROWS N-1 PRECEDING)` |
| `SUM(f, N)` | N-day rolling sum | `SUM(f) OVER (w_time ROWS N-1 PRECEDING)` |
| `MAX(f, N)` | N-day rolling max | `MAX(f) OVER (w_time ROWS N-1 PRECEDING)` |
| `MIN(f, N)` | N-day rolling min | `MIN(f) OVER (w_time ROWS N-1 PRECEDING)` |
| `CORR(f1, f2, N)` | N-day correlation | `CORR(f1, f2) OVER (w_time ROWS N-1 PRECEDING)` |
| `DELTA(f, N)` | Change over N days | `f - LAG(f, N) OVER w_time` |
| `RANK(f)` | Cross-sectional rank | `RANK() OVER (PARTITION BY trade_date ORDER BY f)` |
| `QUANTILE(f, q)` | Cross-sectional quantile | `NTILE(q) OVER (PARTITION BY trade_date ORDER BY f)` |

Arithmetic operators (`+`, `-`, `*`, `/`) pass through unchanged.

Two window types:
- **`w_time`** (shared): `PARTITION BY ts_code ORDER BY trade_date` — used by REF/MEAN/STD/SUM/MAX/MIN/CORR/DELTA
- **Dynamic cross-section** (per-factor): `PARTITION BY trade_date ORDER BY <factor>` — used by RANK/QUANTILE

### Dependency DAG

Factors can depend on raw fields (`$close`, etc.) or other factors (`f_001`). Dependencies are auto-parsed from the expression. FactorRegistry performs topological sort; cyclic dependencies cause a startup error.

### FactorRegistry

```python
class FactorRegistry:
    name: str              # "alpha158"
    factors: dict[str, Factor]

    def register(self, factor: Factor): ...
    def topological_order(self) -> list[Factor]: ...  # raises on cycle
    def min_lookback(self) -> int: ...  # max N across all REF/MEAN/STD/...
```

## FeatureEngine

### DuckDB CTE Builder

For each target `trade_date`, FeatureEngine generates a SQL query:

1. **Base CTE** (`raw`): reads `daily/*.parquet` and `daily_basic/*.parquet`, JOINs on `(ts_code, trade_date)`, filters to `[target - max_lookback - 5, target]`
2. **Factor CTEs**: one CTE per factor in topological order, each `SELECT *, compiled_expression AS factor_name FROM prev_cte`
3. **Final SELECT**: picks target date, outputs only `ts_code, trade_date, f_001 ... f_158`

```python
class FeatureEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.conn = duckdb.connect()  # in-memory

    def compute(self, registry: FactorRegistry, trade_date: str) -> pd.DataFrame:
        # 1. build base CTE
        # 2. append factor CTEs in topo order
        # 3. execute, return df
```

### compile() Function

```python
def compile(expression: str) -> str:
    """Compile DSL expression to DuckDB SQL via regex substitution."""
    # REF, MEAN, STD, SUM, MAX, MIN, CORR, DELTA, RANK, QUANTILE
    # $vars → column names
    # Returns valid DuckDB SQL expression
```

## FeaturePipeline

### Incremental Mode (default)

```
1. Read trade_cal.parquet → open_dates
2. metadata.get_last_date(registry.name) → last_date
3. pending = dates > last_date
4. Exclude future dates (> today)
5. Filter: exclude first ~60 days (lookback window insufficient)
6. For each date in pending:
   a. engine.compute(registry, date) → df
   b. writer.merge(df, path) → merge with existing columns
   c. metadata.insert(registry.name, date, path, len(df))
```

### Rebuild Modes

| Flag | Behavior |
|------|----------|
| `--rebuild` | Delete ALL metadata for this factor set, recompute from earliest date |
| `--since DATE` | Delete metadata records >= DATE, recompute from DATE |

### Error Tolerance

- Single-date failures do NOT halt the pipeline
- Errors are accumulated and reported in the final summary
- Pipeline returns `{"success": N, "failed": N, "errors": [...]}`

## Multi-Factor Merge

All factor sets output to `data/features/YYYY_MM_DD.parquet`.

When a second factor set (e.g., alpha360) is computed for a date that alpha158 already computed:
1. Read existing parquet file
2. JOIN new factor columns on `ts_code`
3. Overwrite with merged DataFrame

### ParquetWriter.merge() (new method)

```python
def merge(self, df: pd.DataFrame, path: Path):
    """If path exists, JOIN new columns; otherwise write fresh."""
    if path.exists():
        existing = pd.read_parquet(path)
        merged = existing.merge(df, on="ts_code", how="outer")
        self.overwrite(merged, path)
    else:
        self.overwrite(df, path)
```

Metadata tracks each factor set independently via `api_name` — different factor sets have separate progress tracking, but share the same parquet file on disk.

## CLI

### New Subcommand: `alpha-quat feature`

```
uv run alpha-quat feature                 # Incremental alpha158
uv run alpha-quat feature -f alpha158     # Explicit factor set
uv run alpha-quat feature --rebuild       # Full recompute
uv run alpha-quat feature --since 2024-01-01   # Partial rebuild
uv run alpha-quat feature --summary       # Show compute progress
```

Future: `-f alpha158,alpha360` for multiple sets.

Existing `alpha-quat` (data pull) command is unchanged.

## Data Output Layout

```
data/
├── registry.db                     # Shared: api_name = "alpha158" / "alpha360"
├── daily/                          # Raw input
├── daily_basic/                    # Raw input
├── trade_cal.parquet              # Date reference
└── features/                       # ★ Feature output
    ├── 2024_01_02.parquet          # ts_code, trade_date, f_001..f_158
    ├── 2024_01_03.parquet
    └── ...
```

## File Change Summary

**New files (8):**

| File | Purpose |
|------|---------|
| `src/alpha_quat/features/__init__.py` | Module init |
| `src/alpha_quat/features/factor.py` | Factor dataclass + compile() |
| `src/alpha_quat/features/registry.py` | FactorRegistry |
| `src/alpha_quat/features/engine.py` | FeatureEngine (DuckDB) |
| `src/alpha_quat/features/pipeline.py` | FeaturePipeline |
| `src/alpha_quat/features/alphasets/__init__.py` | Alphasets init |
| `src/alpha_quat/features/alphasets/alpha158.py` | 158 factor definitions |
| `docs/superpowers/specs/2026-05-18-features-design.md` | This spec |

**Modified files (3):**

| File | Change |
|------|--------|
| `src/alpha_quat/cli.py` | Add `feature` subcommand and argument parsing |
| `src/alpha_quat/data/writer.py` | Add `merge()` method |
| `src/alpha_quat/data/metadata.py` | Add `delete_since()` method |

**Unchanged:** All other files in `src/alpha_quat/data/`

**New tests (5):**

| File | Coverage |
|------|----------|
| `tests/test_features/test_factor.py` | compile() correctness, depends_on parsing |
| `tests/test_features/test_registry.py` | Topo sort, cycle detection, lookback calc |
| `tests/test_features/test_engine.py` | CTE SQL generation with tmp_path parquet |
| `tests/test_features/test_pipeline.py` | Incremental/rebuild/skip/error tolerance |
| `tests/test_features/test_alpha158.py` | All 158 compile, no cycles, valid dependencies |

**Modified tests (1):**

| File | Change |
|------|--------|
| `tests/test_writer.py` | Add `test_merge()` for new method |

## Testing Strategy

- **Unit tests**: compile() DSL mapping, registry topo sort, writer.merge()
- **Integration tests**: FeatureEngine with synthetic parquet data in tmp_path
- **Pipeline tests**: Mock MetadataManager/FactorRegistry, verify date selection logic
- **Integrity tests**: alpha158 factor set compiles without errors, no cyclic dependencies
- **All tests use tmp_path** for isolation (consistent with existing test pattern)

## Verification

```bash
uv run ruff format . && uv run ruff check --fix . && uv run pyright && uv run pytest --cov=src
```
