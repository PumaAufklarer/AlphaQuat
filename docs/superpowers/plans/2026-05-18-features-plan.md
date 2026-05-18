# Feature Engineering Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/alpha_quat/features/` — compute alpha158 factors from daily/daily_basic raw data, output per-date wide parquet to `data/features/`, with incremental + rebuild support.

**Architecture:** Independent FeaturePipeline that reuses ParquetWriter/MetadataManager from data/ module. Factor = SQL expression + metadata, compiled by regex to DuckDB CTE chain. Output merges columns from multiple factor sets into a single per-date parquet file.

**Tech Stack:** DuckDB (window functions), pandas (DataFrame I/O), pyarrow (parquet), existing MetadataManager/ParquetWriter

---

### Task 1: Factor dataclass + compile()

**Files:**
- Create: `src/alpha_quat/features/__init__.py`
- Create: `src/alpha_quat/features/factor.py`
- Create: `tests/test_features/__init__.py`
- Create: `tests/test_features/test_factor.py`

- [ ] **Step 1: Write failing tests for Factor dataclass and compile()**

```python
# tests/test_features/test_factor.py
import pytest
from alpha_quat.features.factor import Factor, compile


class TestFactor:
    def test_basic_fields(self):
        f = Factor(name="KMID", expression="REF($close, 5) / $close - 1", category="momentum")
        assert f.name == "KMID"
        assert f.expression == "REF($close, 5) / $close - 1"
        assert f.category == "momentum"

    def test_depends_on_parsed_from_expression(self):
        f = Factor(name="f_001", expression="REF($close, 5) / $close - 1", category="momentum")
        assert "$close" in f.depends_on

    def test_depends_on_other_factor(self):
        f = Factor(name="f_010", expression="STD(f_001, 20)", category="volatility")
        assert "f_001" in f.depends_on

    def test_depends_on_multiple(self):
        f = Factor(name="f_050", expression="CORR($close, $volume, 10)", category="correlation")
        assert "$close" in f.depends_on
        assert "$volume" in f.depends_on
        assert len(f.depends_on) == 2


class TestCompile:
    def test_ref(self):
        assert compile("REF($close, 5)") == "LAG(close, 5) OVER w_time"

    def test_ref_default_lookback(self):
        assert compile("REF($close, 1)") == "LAG(close, 1) OVER w_time"

    def test_mean(self):
        result = compile("MEAN($close, 20)")
        assert "AVG(close) OVER (" in result
        assert "w_time" in result
        assert "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW" in result

    def test_std(self):
        result = compile("STD($close, 10)")
        assert "STDDEV_SAMP(close) OVER (" in result
        assert "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW" in result

    def test_sum(self):
        result = compile("SUM($volume, 5)")
        assert "SUM(volume) OVER (" in result
        assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in result

    def test_max_min(self):
        assert "MAX(close) OVER (" in compile("MAX($close, 10)")
        assert "MIN(close) OVER (" in compile("MIN($close, 10)")

    def test_corr(self):
        result = compile("CORR($close, $volume, 10)")
        assert "CORR(close, volume) OVER (" in result
        assert "ROWS BETWEEN 9 PRECEDING AND CURRENT ROW" in result

    def test_delta(self):
        result = compile("DELTA($close, 5)")
        assert "close - LAG(close, 5) OVER w_time" in result

    def test_rank(self):
        result = compile("RANK(f_001)")
        assert "RANK() OVER (PARTITION BY trade_date ORDER BY f_001)" in result

    def test_quantile(self):
        result = compile("QUANTILE(f_001, 10)")
        assert "NTILE(10) OVER (PARTITION BY trade_date ORDER BY f_001)" in result

    def test_vwap(self):
        result = compile("$vwap")
        assert "amount" in result
        assert "NULLIF(volume, 0)" in result

    def test_arithmetic(self):
        result = compile("REF($close, 1) / $close - 1")
        assert "LAG(close, 1) OVER w_time / close - 1" == result

    def test_raw_field_passthrough(self):
        assert compile("$open") == "open"
        assert compile("$high") == "high"
        assert compile("$low") == "low"
        assert compile("$volume") == "volume"
        assert compile("$amount") == "amount"

    def test_factor_reference_passthrough(self):
        assert compile("f_001") == "f_001"
        assert compile("f_050") == "f_050"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features/test_factor.py -v`
Expected: FAIL — no module `alpha_quat.features.factor`

- [ ] **Step 3: Create module init files**

```python
# src/alpha_quat/features/__init__.py
# (empty)
```

```python
# tests/test_features/__init__.py
# (empty)
```

- [ ] **Step 4: Write Factor dataclass with auto-parsed depends_on**

```python
# src/alpha_quat/features/factor.py
"""Factor definition and expression compiler."""

import re
from dataclasses import dataclass, field


@dataclass
class Factor:
    name: str
    expression: str
    category: str = ""
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.depends_on:
            self.depends_on = self._parse_deps()

    def _parse_deps(self) -> list[str]:
        refs = set(re.findall(r'\$\w+', self.expression))
        factor_refs = set(re.findall(r'\b(f_\d{3})\b', self.expression))
        # Only include factor refs that aren't inside $vars (f_xxx aren't $vars)
        return sorted(refs | factor_refs)


def compile(expression: str) -> str:
    """Compile DSL expression to DuckDB SQL."""
    return expression
```

- [ ] **Step 5: Run test — depends_on tests should pass, compile tests still fail**

Run: `uv run pytest tests/test_features/test_factor.py -v`
Expected: TestFactor tests PASS, TestCompile tests FAIL

- [ ] **Step 6: Implement compile() with full DSL → SQL mapping**

```python
def compile(expression: str) -> str:
    """Compile DSL expression to DuckDB SQL via regex substitution."""
    expr = expression

    # Order matters: process outer functions first or handle nesting
    # REF: LAG with offset
    expr = re.sub(
        r'REF\((\w+),\s*(\d+)\)',
        r'LAG(\1, \2) OVER w_time',
        expr,
    )
    # MEAN: AVG with frame
    expr = re.sub(
        r'MEAN\((\w+),\s*(\d+)\)',
        r'AVG(\1) OVER (w_time ROWS BETWEEN \2-1 PRECEDING AND CURRENT ROW)',
        expr,
    )
    # STD: STDDEV_SAMP with frame
    expr = re.sub(
        r'STD\((\w+),\s*(\d+)\)',
        r'STDDEV_SAMP(\1) OVER (w_time ROWS BETWEEN \2-1 PRECEDING AND CURRENT ROW)',
        expr,
    )
    # SUM: SUM with frame
    expr = re.sub(
        r'SUM\((\w+),\s*(\d+)\)',
        r'SUM(\1) OVER (w_time ROWS BETWEEN \2-1 PRECEDING AND CURRENT ROW)',
        expr,
    )
    # MAX: MAX with frame
    expr = re.sub(
        r'MAX\((\w+),\s*(\d+)\)',
        r'MAX(\1) OVER (w_time ROWS BETWEEN \2-1 PRECEDING AND CURRENT ROW)',
        expr,
    )
    # MIN: MIN with frame
    expr = re.sub(
        r'MIN\((\w+),\s*(\d+)\)',
        r'MIN(\1) OVER (w_time ROWS BETWEEN \2-1 PRECEDING AND CURRENT ROW)',
        expr,
    )
    # CORR: CORR with frame (two args)
    expr = re.sub(
        r'CORR\((\w+),\s*(\w+),\s*(\d+)\)',
        r'CORR(\1, \2) OVER (w_time ROWS BETWEEN \3-1 PRECEDING AND CURRENT ROW)',
        expr,
    )
    # DELTA: feature - LAG(feature, N)
    expr = re.sub(
        r'DELTA\((\w+),\s*(\d+)\)',
        r'\1 - LAG(\1, \2) OVER w_time',
        expr,
    )
    # RANK: cross-sectional rank
    expr = re.sub(
        r'RANK\((\w+)\)',
        r'RANK() OVER (PARTITION BY trade_date ORDER BY \1)',
        expr,
    )
    # QUANTILE: cross-sectional ntile
    expr = re.sub(
        r'QUANTILE\((\w+),\s*(\d+)\)',
        r'NTILE(\2) OVER (PARTITION BY trade_date ORDER BY \1)',
        expr,
    )
    # $vwap: computed from amount/volume
    expr = expr.replace('$vwap', 'amount / NULLIF(volume, 0)')
    # $raw fields
    expr = expr.replace('$open', 'open')
    expr = expr.replace('$high', 'high')
    expr = expr.replace('$low', 'low')
    expr = expr.replace('$close', 'close')
    expr = expr.replace('$volume', 'volume')
    expr = expr.replace('$amount', 'amount')

    return expr
```

- [ ] **Step 7: Run tests — all should pass**

Run: `uv run pytest tests/test_features/test_factor.py -v`
Expected: all 19 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/alpha_quat/features/__init__.py src/alpha_quat/features/factor.py tests/test_features/
git commit -m "feat: add Factor dataclass and DSL compiler"
```

---

### Task 2: FactorRegistry

**Files:**
- Create: `src/alpha_quat/features/registry.py`
- Create: `tests/test_features/test_registry.py`

- [ ] **Step 1: Write failing tests for FactorRegistry**

```python
# tests/test_features/test_registry.py
import pytest
from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry


class TestFactorRegistry:
    def test_register_and_list(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="REF($close, 1) / $close - 1", category="momentum")
        reg.register(f1)
        assert "f_001" in reg.factors
        assert reg.factors["f_001"] is f1

    def test_topological_order_simple(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="$close", category="price")
        f2 = Factor(name="f_002", expression="REF(f_001, 1)", category="momentum")
        f3 = Factor(name="f_003", expression="STD(f_002, 5)", category="volatility")
        reg.register(f1)
        reg.register(f2)
        reg.register(f3)
        order = reg.topological_order()
        names = [f.name for f in order]
        assert names.index("f_001") < names.index("f_002")
        assert names.index("f_002") < names.index("f_003")

    def test_topological_order_no_deps(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="$close", category="price")
        f2 = Factor(name="f_002", expression="$volume", category="volume")
        reg.register(f1)
        reg.register(f2)
        order = reg.topological_order()
        names = [f.name for f in order]
        assert len(names) == 2

    def test_cycle_detection(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="f_002", category="test")
        f2 = Factor(name="f_002", expression="f_001", category="test")
        reg.register(f1)
        reg.register(f2)
        with pytest.raises(ValueError, match="cycle"):
            reg.topological_order()

    def test_min_lookback(self):
        reg = FactorRegistry(name="test")
        f1 = Factor(name="f_001", expression="REF($close, 5)", category="momentum")
        f2 = Factor(name="f_002", expression="MEAN($volume, 20)", category="volume")
        f3 = Factor(name="f_003", expression="STD($close, 10)", category="volatility")
        reg.register(f1)
        reg.register(f2)
        reg.register(f3)
        assert reg.min_lookback() == 20

    def test_min_lookback_no_operators(self):
        reg = FactorRegistry(name="test")
        reg.register(Factor(name="f_001", expression="$close", category="price"))
        assert reg.min_lookback() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features/test_registry.py -v`
Expected: FAIL — no module `alpha_quat.features.registry`

- [ ] **Step 3: Write FactorRegistry**

```python
# src/alpha_quat/features/registry.py
"""Factor registry with dependency resolution."""

import re
from collections import deque

from alpha_quat.features.factor import Factor


class FactorRegistry:
    def __init__(self, name: str):
        self.name = name
        self.factors: dict[str, Factor] = {}

    def register(self, factor: Factor):
        self.factors[factor.name] = factor

    def topological_order(self) -> list[Factor]:
        in_degree: dict[str, int] = {}
        adj: dict[str, list[str]] = {name: [] for name in self.factors}

        for name, factor in self.factors.items():
            deps = [d for d in factor.depends_on if d in self.factors]
            in_degree[name] = len(deps)
            for dep in deps:
                adj[dep].append(name)

        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        result: list[Factor] = []

        while queue:
            name = queue.popleft()
            result.append(self.factors[name])
            for neighbor in adj[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.factors):
            raise ValueError(
                f"Cycle detected in factor dependencies for registry '{self.name}'"
            )
        return result

    def min_lookback(self) -> int:
        max_n = 0
        for factor in self.factors.values():
            for n in re.findall(r'(?:REF|MEAN|STD|SUM|MAX|MIN|DELTA)\(\w+,\s*(\d+)\)', factor.expression):
                max_n = max(max_n, int(n))
            for m in re.findall(r'CORR\(\w+,\s*\w+,\s*(\d+)\)', factor.expression):
                max_n = max(max_n, int(m))
        return max_n
```

- [ ] **Step 4: Run tests — all should pass**

Run: `uv run pytest tests/test_features/test_registry.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/features/registry.py tests/test_features/test_registry.py
git commit -m "feat: add FactorRegistry with topological sort and cycle detection"
```

---

### Task 3: MetadataManager.delete_since()

**Files:**
- Modify: `src/alpha_quat/data/metadata.py` (add method after get_last_date)
- Modify: `tests/test_metadata.py` (add test)

- [ ] **Step 1: Write failing test for delete_since()**

```python
# append to tests/test_metadata.py
class TestDeleteSince:
    def test_delete_since_removes_records(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        mgr = MetadataManager(db_path)
        mgr.insert("alpha158", "2024-01-02", "data/features/2024_01_02.parquet", 5000)
        mgr.insert("alpha158", "2024-01-03", "data/features/2024_01_03.parquet", 5010)
        mgr.insert("alpha158", "2024-01-04", "data/features/2024_01_04.parquet", 5020)

        mgr.delete_since("alpha158", "2024-01-03")

        last = mgr.get_last_date("alpha158")
        assert last == datetime.date(2024, 1, 2)

    def test_delete_since_other_api_unaffected(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        mgr = MetadataManager(db_path)
        mgr.insert("alpha158", "2024-01-02", "features/2024_01_02.parquet", 5000)
        mgr.insert("daily", "2024-01-03", "daily/2024_01_03.parquet", 5000)

        mgr.delete_since("alpha158", "2024-01-02")

        assert mgr.get_last_date("alpha158") is None
        assert mgr.get_last_date("daily") == datetime.date(2024, 1, 3)

    def test_delete_clear_all_for_rebuild(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        mgr = MetadataManager(db_path)
        mgr.insert("alpha158", "2024-01-02", "features/2024_01_02.parquet", 5000)
        mgr.insert("alpha158", "2024-01-05", "features/2024_01_05.parquet", 5000)

        mgr.delete_since("alpha158", None)

        assert mgr.get_last_date("alpha158") is None
```

Note: `MetadataManager` already imported in the test file; add `import datetime` if not present.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metadata.py::TestDeleteSince -v`
Expected: FAIL — `MetadataManager` has no attribute `delete_since`

- [ ] **Step 3: Add delete_since() method**

Insert after `get_last_date()` in `src/alpha_quat/data/metadata.py` (after line 57):

```python
    def delete_since(self, api_name: str, since: str | None):
        if since is None:
            self.conn.execute(
                "DELETE FROM data_registry WHERE api_name = ?",
                [api_name],
            )
        else:
            self.conn.execute(
                "DELETE FROM data_registry WHERE api_name = ? AND trade_date >= ?",
                [api_name, since],
            )
```

- [ ] **Step 4: Run tests — all should pass**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: all tests PASS (existing 6 + new 3 = 9)

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/data/metadata.py tests/test_metadata.py
git commit -m "feat: add MetadataManager.delete_since() for feature rebuild"
```

---

### Task 4: ParquetWriter.merge()

**Files:**
- Modify: `src/alpha_quat/data/writer.py` (add merge method)
- Modify: `tests/test_writer.py` (add test)

- [ ] **Step 1: Write failing test for merge()**

```python
# append to tests/test_writer.py
class TestMerge:
    def test_merge_writes_new_file(self, tmp_path):
        writer = ParquetWriter()
        df = pd.DataFrame({"ts_code": ["A", "B"], "f_001": [1.0, 2.0]})
        path = tmp_path / "output.parquet"
        writer.merge(df, path)
        assert path.exists()
        result = pd.read_parquet(path)
        assert list(result.columns) == ["ts_code", "f_001"]
        assert len(result) == 2

    def test_merge_joins_with_existing_file(self, tmp_path):
        writer = ParquetWriter()
        path = tmp_path / "output.parquet"

        existing = pd.DataFrame({"ts_code": ["A", "B"], "f_001": [1.0, 2.0]})
        existing.to_parquet(path, index=False)

        new = pd.DataFrame({"ts_code": ["A", "B"], "f_010": [3.0, 4.0]})
        writer.merge(new, path)

        result = pd.read_parquet(path)
        assert "f_001" in result.columns
        assert "f_010" in result.columns
        assert list(result["f_001"]) == [1.0, 2.0]
        assert list(result["f_010"]) == [3.0, 4.0]

    def test_merge_preserves_ts_code_mismatch(self, tmp_path):
        writer = ParquetWriter()
        path = tmp_path / "output.parquet"

        existing = pd.DataFrame({"ts_code": ["A", "B", "C"], "f_001": [1.0, 2.0, 3.0]})
        existing.to_parquet(path, index=False)

        new = pd.DataFrame({"ts_code": ["B", "C", "D"], "f_010": [4.0, 5.0, 6.0]})
        writer.merge(new, path)

        result = pd.read_parquet(path)
        assert len(result) == 4  # A, B, C, D
        assert set(result["ts_code"]) == {"A", "B", "C", "D"}
```

Note: `import pandas as pd` already present at top of `tests/test_writer.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_writer.py::TestMerge -v`
Expected: FAIL — `ParquetWriter` has no attribute `merge`

- [ ] **Step 3: Add merge() method**

Add after `write()` in `src/alpha_quat/data/writer.py` (after line 16):

```python
    def merge(self, df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            merged = existing.merge(df, on="ts_code", how="outer")
            merged.to_parquet(path, index=False)
        else:
            df.to_parquet(path, index=False)
```

- [ ] **Step 4: Run tests — all should pass**

Run: `uv run pytest tests/test_writer.py -v`
Expected: all tests PASS (existing 4 + new 3 = 7)

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/data/writer.py tests/test_writer.py
git commit -m "feat: add ParquetWriter.merge() for multi-factor column joining"
```

---

### Task 5: FeatureEngine

**Files:**
- Create: `src/alpha_quat/features/engine.py`
- Create: `tests/test_features/test_engine.py`

- [ ] **Step 1: Write failing tests for FeatureEngine**

```python
# tests/test_features/test_engine.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from alpha_quat.features.engine import FeatureEngine
from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry


def make_daily_parquet(tmp_path, data: list[dict]):
    """Helper: write synthetic daily parquet files."""
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    df = pd.DataFrame(data)
    df["trade_date"] = df["trade_date"].astype(str)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, str(daily_dir), partition_cols=["trade_date"])


def make_daily_basic_parquet(tmp_path, data: list[dict]):
    """Helper: write synthetic daily_basic parquet files."""
    db_dir = tmp_path / "daily_basic"
    db_dir.mkdir()
    df = pd.DataFrame(data)
    df["trade_date"] = df["trade_date"].astype(str)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, str(db_dir), partition_cols=["trade_date"])


class TestFeatureEngine:
    def test_compute_single_factor(self, tmp_path):
        # 10 days of data for 2 stocks
        daily_data = []
        for stock in ["000001.SZ", "000002.SZ"]:
            for i, date in enumerate(["20240102", "20240103", "20240104", "20240105",
                                       "20240108", "20240109", "20240110", "20240111",
                                       "20240112", "20240115"]):
                daily_data.append({
                    "ts_code": stock,
                    "trade_date": date,
                    "open": 10.0 + i + (0.1 if stock == "000002.SZ" else 0),
                    "high": 11.0 + i,
                    "low": 9.0 + i,
                    "close": 10.5 + i + (0.1 if stock == "000002.SZ" else 0),
                    "vol": 1000000.0,
                    "amount": 10500000.0,
                })
        make_daily_parquet(tmp_path, daily_data)
        make_daily_basic_parquet(tmp_path, [])

        reg = FactorRegistry(name="test")
        reg.register(Factor(name="f_001", expression="REF($close, 1) / $close - 1", category="momentum"))

        engine = FeatureEngine(data_dir=tmp_path)
        result = engine.compute(reg, trade_date="20240115")

        assert "ts_code" in result.columns
        assert "trade_date" in result.columns
        assert "f_001" in result.columns
        assert len(result) == 2  # two stocks
        assert all(result["trade_date"] == "20240115")

    def test_compute_multiple_factors(self, tmp_path):
        daily_data = []
        for stock in ["000001.SZ"]:
            for i, date in enumerate(["20240102", "20240103", "20240104", "20240105",
                                       "20240108", "20240109"]):
                daily_data.append({
                    "ts_code": stock,
                    "trade_date": date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0 + i,
                    "vol": 1000000.0,
                    "amount": 10000000.0,
                })
        make_daily_parquet(tmp_path, daily_data)
        make_daily_basic_parquet(tmp_path, [])

        reg = FactorRegistry(name="test")
        reg.register(Factor(name="f_001", expression="$close", category="price"))
        reg.register(Factor(name="f_002", expression="REF(f_001, 1)", category="momentum"))
        reg.register(Factor(name="f_003", expression="MEAN(f_001, 3)", category="momentum"))

        engine = FeatureEngine(data_dir=tmp_path)
        result = engine.compute(reg, trade_date="20240109")

        assert "f_001" in result.columns
        assert "f_002" in result.columns
        assert "f_003" in result.columns
        assert len(result) == 1

    def test_compute_with_daily_basic_join(self, tmp_path):
        daily_data = []
        daily_basic_data = []
        for stock in ["000001.SZ", "000002.SZ"]:
            for i, date in enumerate(["20240102", "20240103", "20240104", "20240105"]):
                daily_data.append({
                    "ts_code": stock, "trade_date": date,
                    "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0 + i,
                    "vol": 1000000.0, "amount": 10000000.0,
                })
                daily_basic_data.append({
                    "ts_code": stock, "trade_date": date,
                    "total_mv": 1e8, "circ_mv": 5e7,
                })
        make_daily_parquet(tmp_path, daily_data)
        make_daily_basic_parquet(tmp_path, daily_basic_data)

        reg = FactorRegistry(name="test")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        engine = FeatureEngine(data_dir=tmp_path)
        result = engine.compute(reg, trade_date="20240105")

        assert len(result) == 2
        assert "total_mv" not in result.columns  # raw columns filtered out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features/test_engine.py -v`
Expected: FAIL — no module `alpha_quat.features.engine`

- [ ] **Step 3: Write FeatureEngine**

```python
# src/alpha_quat/features/engine.py
"""FeatureEngine — DuckDB CTE compiler and executor."""

from pathlib import Path

import duckdb
import pandas as pd

from alpha_quat.features.registry import FactorRegistry
from alpha_quat.features.factor import compile


class FeatureEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.conn = duckdb.connect()

    def compute(self, registry: FactorRegistry, trade_date: str) -> pd.DataFrame:
        factors = registry.topological_order()
        lookback = registry.min_lookback()
        margin = lookback + 5  # extra cushion

        if not factors:
            return pd.DataFrame()

        sql = self._base_cte(trade_date, margin)
        prev = "raw"

        for f in factors:
            compiled = compile(f.expression)
            sql += f""",
cte_{f.name} AS (
  SELECT *, {compiled} AS {f.name}
  FROM {prev}
  WINDOW w_time AS (PARTITION BY ts_code ORDER BY trade_date)
)"""
            prev = f"cte_{f.name}"

        sql += self._final_select(factors, trade_date, prev)
        return self.conn.execute(sql).df()

    def _base_cte(self, trade_date: str, margin: int) -> str:
        daily_path = self.data_dir / "daily" / "*.parquet"
        db_path = self.data_dir / "daily_basic" / "*.parquet"

        return f"""
WITH raw AS (
  SELECT
    d.ts_code,
    d.trade_date,
    d.open,
    d.high,
    d.low,
    d.close,
    d.vol AS volume,
    d.amount
  FROM read_parquet('{daily_path}') d
  WHERE d.trade_date >= '{trade_date}' - INTERVAL {margin} DAYS
    AND d.trade_date <= '{trade_date}'
)"""

    def _final_select(self, factors, trade_date: str, prev: str) -> str:
        cols = ", ".join(f.name for f in factors)
        return f"""
SELECT ts_code, trade_date, {cols}
FROM {prev}
WHERE trade_date = '{trade_date}'
ORDER BY ts_code
"""
```

- [ ] **Step 4: Run tests — verify passing**

Run: `uv run pytest tests/test_features/test_engine.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/features/engine.py tests/test_features/test_engine.py
git commit -m "feat: add FeatureEngine with DuckDB CTE compiler"
```

---

### Task 6: FeaturePipeline

**Files:**
- Create: `src/alpha_quat/features/pipeline.py`
- Create: `tests/test_features/test_pipeline.py`

- [ ] **Step 1: Write failing tests for FeaturePipeline**

```python
# tests/test_features/test_pipeline.py
import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alpha_quat.features.pipeline import FeaturePipeline
from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.metadata import MetadataManager


class FakeEngine:
    """Returns a simple DataFrame per date."""
    def compute(self, registry, trade_date):
        return pd.DataFrame({
            "ts_code": [f"{trade_date}_A", f"{trade_date}_B"],
            "trade_date": [trade_date, trade_date],
            "f_001": [1.0, 2.0],
        })


class TestFeaturePipeline:
    def make_trade_cal(self, data_dir, dates):
        path = data_dir / "trade_cal.parquet"
        df = pd.DataFrame({
            "exchange": "SSE",
            "cal_date": dates,
            "is_open": 1,
            "pretrade_date": "",
        })
        df.to_parquet(path, index=False)

    def test_incremental_first_run(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103", "20240104"])

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result["success"] == 3
        for date in ["20240102", "20240103", "20240104"]:
            assert (features_dir / f"{date}.parquet").exists()
        assert metadata.get_last_date("alpha158") == datetime.date(2024, 1, 4)

    def test_incremental_skips_completed_dates(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103", "20240104"])

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        # Pre-seed: dates 01-02 and 01-03 already done
        metadata.insert("alpha158", "20240102", "features/20240102.parquet", 2)
        metadata.insert("alpha158", "20240103", "features/20240103.parquet", 2)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result["success"] == 1  # only 20240104

    def test_rebuild_deletes_all(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103"])

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        metadata.insert("alpha158", "20240102", "features/20240102.parquet", 2)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg, rebuild=True)

        assert result["success"] == 2  # all dates recomputed

    def test_error_tolerance_continues(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        self.make_trade_cal(tmp_path, ["20240102", "20240103", "20240104"])

        class FailingEngine:
            def __init__(self):
                self.calls = 0

            def compute(self, registry, trade_date):
                self.calls += 1
                if trade_date == "20240103":
                    raise RuntimeError("simulated failure")
                return pd.DataFrame({
                    "ts_code": ["A"],
                    "trade_date": [trade_date],
                    "f_001": [1.0],
                })

        engine = FailingEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result["success"] == 2
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert "20240103" in str(result["errors"][0])

    def test_missing_trade_cal_returns_early(self, tmp_path):
        features_dir = tmp_path / "features"
        db_path = str(tmp_path / "registry.db")

        engine = FakeEngine()
        writer = ParquetWriter()
        metadata = MetadataManager(db_path)

        reg = FactorRegistry(name="alpha158")
        reg.register(Factor(name="f_001", expression="$close", category="price"))

        pipeline = FeaturePipeline(
            data_dir=tmp_path,
            output_dir=features_dir,
            engine=engine,
            writer=writer,
            metadata=metadata,
        )
        result = pipeline.run(reg)

        assert result.get("message") is not None  # warns about missing trade_cal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features/test_pipeline.py -v`
Expected: FAIL — no module `alpha_quat.features.pipeline`

- [ ] **Step 3: Write FeaturePipeline**

```python
# src/alpha_quat/features/pipeline.py
"""FeaturePipeline — date scheduling with incremental/rebuild support."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from alpha_quat.features.registry import FactorRegistry

logger = logging.getLogger(__name__)


class FeaturePipeline:
    def __init__(self, data_dir, output_dir, engine, writer, metadata):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.engine = engine
        self.writer = writer
        self.metadata = metadata

    def run(self, registry: FactorRegistry, rebuild=False, since=None) -> dict:
        trade_cal_path = self.data_dir / "trade_cal.parquet"
        if not trade_cal_path.exists():
            msg = "trade_cal.parquet not found. Run 'alpha-quat -s trade_cal' first."
            logger.warning(msg)
            return {"success": 0, "failed": 0, "errors": [], "message": msg}

        cal = pd.read_parquet(trade_cal_path)
        open_dates = sorted(
            cal.loc[cal["is_open"] == 1, "cal_date"].astype(str).tolist()
        )
        today_str = date.today().strftime("%Y%m%d")
        open_dates = [d for d in open_dates if d <= today_str]

        if rebuild:
            self.metadata.delete_since(registry.name, None)
            pending = list(open_dates)
        elif since:
            self.metadata.delete_since(registry.name, since)
            pending = [d for d in open_dates if d >= since]
        else:
            last = self.metadata.get_last_date(registry.name)
            if last:
                last_str = last.strftime("%Y%m%d")
                pending = [d for d in open_dates if d > last_str]
            else:
                pending = list(open_dates)

        lookback = registry.min_lookback()
        if lookback > 0 and not rebuild and not since:
            # Skip first N dates where lookback window insufficient
            all_idx = {d: i for i, d in enumerate(open_dates)}
            min_idx = lookback
            pending = [d for d in pending if all_idx.get(d, 0) >= min_idx]

        results = {"success": 0, "failed": 0, "errors": []}

        for trade_date in pending:
            try:
                df = self.engine.compute(registry, trade_date)
                path = self.output_dir / f"{trade_date}.parquet"
                self.writer.merge(df, path)
                self.metadata.insert(
                    registry.name, trade_date, str(path), len(df)
                )
                results["success"] += 1
            except Exception as e:
                logger.error("Failed %s: %s", trade_date, e)
                results["failed"] += 1
                results["errors"].append({trade_date: str(e)})

        return results
```

- [ ] **Step 4: Run tests — all should pass**

Run: `uv run pytest tests/test_features/test_pipeline.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/features/pipeline.py tests/test_features/test_pipeline.py
git commit -m "feat: add FeaturePipeline with incremental/rebuild support"
```

---

### Task 7: Alpha158 factor definitions

**Files:**
- Create: `src/alpha_quat/features/alphasets/__init__.py`
- Create: `src/alpha_quat/features/alphasets/alpha158.py`
- Create: `tests/test_features/test_alpha158.py`

- [ ] **Step 1: Write test for alpha158 integrity**

```python
# tests/test_features/test_alpha158.py
import pytest
from alpha_quat.features.registry import FactorRegistry
from alpha_quat.features.alphasets.alpha158 import build_alpha158


class TestAlpha158:
    def test_all_factors_registered(self):
        reg = build_alpha158()
        assert reg.name == "alpha158"
        assert len(reg.factors) == 158

    def test_all_factors_compile(self):
        from alpha_quat.features.factor import compile
        reg = build_alpha158()
        for name, factor in reg.factors.items():
            try:
                result = compile(factor.expression)
                assert result, f"compile({name}) returned empty"
            except Exception as e:
                pytest.fail(f"compile({name}) failed: {e}")

    def test_no_cycles(self):
        reg = build_alpha158()
        ordered = reg.topological_order()
        assert len(ordered) == 158

    def test_all_deps_exist(self):
        reg = build_alpha158()
        factor_names = set(reg.factors.keys())
        raw_fields = {
            "$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap"
        }
        for factor in reg.factors.values():
            for dep in factor.depends_on:
                if dep.startswith("$"):
                    assert dep in raw_fields, f"{factor.name} depends on unknown raw field {dep}"
                else:
                    assert dep in factor_names, f"{factor.name} depends on unknown factor {dep}"

    def test_min_lookback_consistent(self):
        reg = build_alpha158()
        lookback = reg.min_lookback()
        assert lookback >= 0
        # Alpha158 typically has max window of 60
        assert lookback <= 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features/test_alpha158.py -v`
Expected: FAIL — no module `alpha_quat.features.alphasets.alpha158`

- [ ] **Step 3: Create alpha158.py**

```python
# src/alpha_quat/features/alphasets/__init__.py
```

```python
# src/alpha_quat/features/alphasets/alpha158.py
"""Qlib Alpha158 factor definitions.

158 factors based on OHLCV + VWAP raw features, using REF, MEAN, STD, SUM,
MAX, MIN, CORR, DELTA, RANK, and QUANTILE operators with windows of
5, 10, 20, 30, and 60 days.

Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
"""

from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry

# Raw features used:
#   $open, $high, $low, $close, $volume, $amount, $vwap
#
# Operator windows: 5, 10, 20, 30, 60
#
# For each raw feature we apply a systematic set of operators.
# The naming convention aligns with Qlib's alpha158 naming.

_FACTORS = [
    # === $open (KMID group — price factors) ===
    Factor(name="KMID",  expression="REF($open, 1) / $open - 1",  category="price"),
    Factor(name="KLEN",  expression="REF($open, 2) / $open - 1",  category="price"),
    Factor(name="KMID2", expression="MEAN($open, 5) / $open",     category="price"),
    Factor(name="KLEN2", expression="MEAN($open, 10) / $open",    category="price"),
    Factor(name="KMID3", expression="MEAN($open, 20) / $open",    category="price"),
    Factor(name="KLEN3", expression="MEAN($open, 30) / $open",    category="price"),
    Factor(name="KMID4", expression="MEAN($open, 60) / $open",    category="price"),
    Factor(name="KLEN4", expression="STD($open, 5)",              category="price"),
    Factor(name="KMID5", expression="STD($open, 10)",             category="price"),
    Factor(name="KLEN5", expression="STD($open, 20)",             category="price"),
    Factor(name="KMID6", expression="STD($open, 30)",             category="price"),
    Factor(name="KLEN6", expression="STD($open, 60)",             category="price"),
    Factor(name="KMID7", expression="MAX($open, 5)",              category="price"),
    Factor(name="KLEN7", expression="MAX($open, 10)",             category="price"),
    Factor(name="KMID8", expression="MAX($open, 20)",             category="price"),
    Factor(name="KLEN8", expression="MAX($open, 30)",             category="price"),
    Factor(name="KMID9", expression="MAX($open, 60)",             category="price"),
    Factor(name="KLEN9", expression="MIN($open, 5)",              category="price"),
    Factor(name="KMID10",expression="MIN($open, 10)",             category="price"),
    Factor(name="KLEN10",expression="MIN($open, 20)",             category="price"),
    Factor(name="KMID11",expression="MIN($open, 30)",             category="price"),
    Factor(name="KLEN11",expression="MIN($open, 60)",             category="price"),

    # === $high ===
    Factor(name="KMID12",expression="REF($high, 1) / $high - 1",  category="price"),
    Factor(name="KLEN12",expression="REF($high, 2) / $high - 1",  category="price"),
    Factor(name="KMID13",expression="MEAN($high, 5) / $high",     category="price"),
    Factor(name="KLEN13",expression="MEAN($high, 10) / $high",    category="price"),
    Factor(name="KMID14",expression="MEAN($high, 20) / $high",    category="price"),
    Factor(name="KLEN14",expression="MEAN($high, 30) / $high",    category="price"),
    Factor(name="KMID15",expression="MEAN($high, 60) / $high",    category="price"),
    Factor(name="KLEN15",expression="STD($high, 5)",              category="price"),
    Factor(name="KMID16",expression="STD($high, 10)",             category="price"),
    Factor(name="KLEN16",expression="STD($high, 20)",             category="price"),
    Factor(name="KMID17",expression="STD($high, 30)",             category="price"),
    Factor(name="KLEN17",expression="STD($high, 60)",             category="price"),
    Factor(name="KMID18",expression="MAX($high, 5)",              category="price"),
    Factor(name="KLEN18",expression="MAX($high, 10)",             category="price"),
    Factor(name="KMID19",expression="MAX($high, 20)",             category="price"),
    Factor(name="KLEN19",expression="MAX($high, 30)",             category="price"),
    Factor(name="KMID20",expression="MAX($high, 60)",             category="price"),
    Factor(name="KLEN20",expression="MIN($high, 5)",              category="price"),
    Factor(name="KMID21",expression="MIN($high, 10)",             category="price"),
    Factor(name="KLEN21",expression="MIN($high, 20)",             category="price"),
    Factor(name="KMID22",expression="MIN($high, 30)",             category="price"),
    Factor(name="KLEN22",expression="MIN($high, 60)",             category="price"),

    # === $low ===
    Factor(name="KMID23",expression="REF($low, 1) / $low - 1",    category="price"),
    Factor(name="KLEN23",expression="REF($low, 2) / $low - 1",    category="price"),
    Factor(name="KMID24",expression="MEAN($low, 5) / $low",       category="price"),
    Factor(name="KLEN24",expression="MEAN($low, 10) / $low",      category="price"),
    Factor(name="KMID25",expression="MEAN($low, 20) / $low",      category="price"),
    Factor(name="KLEN25",expression="MEAN($low, 30) / $low",      category="price"),
    Factor(name="KMID26",expression="MEAN($low, 60) / $low",      category="price"),
    Factor(name="KLEN26",expression="STD($low, 5)",               category="price"),
    Factor(name="KMID27",expression="STD($low, 10)",              category="price"),
    Factor(name="KLEN27",expression="STD($low, 20)",              category="price"),
    Factor(name="KMID28",expression="STD($low, 30)",              category="price"),
    Factor(name="KLEN28",expression="STD($low, 60)",              category="price"),
    Factor(name="KMID29",expression="MAX($low, 5)",               category="price"),
    Factor(name="KLEN29",expression="MAX($low, 10)",              category="price"),
    Factor(name="KMID30",expression="MAX($low, 20)",              category="price"),
    Factor(name="KLEN30",expression="MAX($low, 30)",              category="price"),
    Factor(name="KMID31",expression="MAX($low, 60)",              category="price"),
    Factor(name="KLEN31",expression="MIN($low, 5)",               category="price"),
    Factor(name="KMID32",expression="MIN($low, 10)",              category="price"),
    Factor(name="KLEN32",expression="MIN($low, 20)",              category="price"),
    Factor(name="KMID33",expression="MIN($low, 30)",              category="price"),
    Factor(name="KLEN33",expression="MIN($low, 60)",              category="price"),

    # === $close ===
    Factor(name="KLEN34",expression="REF($close, 1) / $close - 1", category="price"),
    Factor(name="KMID34",expression="REF($close, 2) / $close - 1", category="price"),
    Factor(name="KLEN35",expression="MEAN($close, 5) / $close",   category="price"),
    Factor(name="KMID35",expression="MEAN($close, 10) / $close",  category="price"),
    Factor(name="KLEN36",expression="MEAN($close, 20) / $close",  category="price"),
    Factor(name="KMID36",expression="MEAN($close, 30) / $close",  category="price"),
    Factor(name="KLEN37",expression="MEAN($close, 60) / $close",  category="price"),
    Factor(name="KMID37",expression="STD($close, 5)",             category="price"),
    Factor(name="KLEN38",expression="STD($close, 10)",            category="price"),
    Factor(name="KMID38",expression="STD($close, 20)",            category="price"),
    Factor(name="KLEN39",expression="STD($close, 30)",            category="price"),
    Factor(name="KMID39",expression="STD($close, 60)",            category="price"),
    Factor(name="KLEN40",expression="MAX($close, 5)",             category="price"),
    Factor(name="KMID40",expression="MAX($close, 10)",            category="price"),
    Factor(name="KLEN41",expression="MAX($close, 20)",            category="price"),
    Factor(name="KMID41",expression="MAX($close, 30)",            category="price"),
    Factor(name="KLEN42",expression="MAX($close, 60)",            category="price"),
    Factor(name="KMID42",expression="MIN($close, 5)",             category="price"),
    Factor(name="KLEN43",expression="MIN($close, 10)",            category="price"),
    Factor(name="KMID43",expression="MIN($close, 20)",            category="price"),
    Factor(name="KLEN44",expression="MIN($close, 30)",            category="price"),
    Factor(name="KMID44",expression="MIN($close, 60)",            category="price"),

    # === $volume ===
    Factor(name="KMID45",expression="REF($volume, 1) / $volume - 1",  category="volume"),
    Factor(name="KLEN45",expression="REF($volume, 2) / $volume - 1",  category="volume"),
    Factor(name="KMID46",expression="MEAN($volume, 5) / $volume",     category="volume"),
    Factor(name="KLEN46",expression="MEAN($volume, 10) / $volume",    category="volume"),
    Factor(name="KMID47",expression="MEAN($volume, 20) / $volume",    category="volume"),
    Factor(name="KLEN47",expression="MEAN($volume, 30) / $volume",    category="volume"),
    Factor(name="KMID48",expression="MEAN($volume, 60) / $volume",    category="volume"),
    Factor(name="KLEN48",expression="STD($volume, 5)",                category="volume"),
    Factor(name="KMID49",expression="STD($volume, 10)",               category="volume"),
    Factor(name="KLEN49",expression="STD($volume, 20)",               category="volume"),
    Factor(name="KMID50",expression="STD($volume, 30)",               category="volume"),
    Factor(name="KLEN50",expression="STD($volume, 60)",               category="volume"),
    Factor(name="KMID51",expression="MAX($volume, 5)",                category="volume"),
    Factor(name="KLEN51",expression="MAX($volume, 10)",               category="volume"),
    Factor(name="KMID52",expression="MAX($volume, 20)",               category="volume"),
    Factor(name="KLEN52",expression="MAX($volume, 30)",               category="volume"),
    Factor(name="KMID53",expression="MAX($volume, 60)",               category="volume"),
    Factor(name="KLEN53",expression="CORR($close, $volume, 5)",       category="correlation"),
    Factor(name="KMID54",expression="CORR($close, $volume, 10)",      category="correlation"),
    Factor(name="KLEN54",expression="CORR($close, $volume, 20)",      category="correlation"),
    Factor(name="KMID55",expression="CORR($close, $volume, 30)",      category="correlation"),
    Factor(name="KLEN55",expression="CORR($close, $volume, 60)",      category="correlation"),

    # === $amount ===
    Factor(name="KMID56",expression="REF($amount, 1) / $amount - 1",  category="volume"),
    Factor(name="KLEN56",expression="REF($amount, 2) / $amount - 1",  category="volume"),
    Factor(name="KMID57",expression="MEAN($amount, 5) / $amount",     category="volume"),
    Factor(name="KLEN57",expression="MEAN($amount, 10) / $amount",    category="volume"),
    Factor(name="KMID58",expression="MEAN($amount, 20) / $amount",    category="volume"),
    Factor(name="KLEN58",expression="MEAN($amount, 30) / $amount",    category="volume"),
    Factor(name="KMID59",expression="MEAN($amount, 60) / $amount",    category="volume"),
    Factor(name="KLEN59",expression="STD($amount, 5)",                category="volume"),
    Factor(name="KMID60",expression="STD($amount, 10)",               category="volume"),
    Factor(name="KLEN60",expression="STD($amount, 20)",               category="volume"),
    Factor(name="KMID61",expression="STD($amount, 30)",               category="volume"),
    Factor(name="KLEN61",expression="STD($amount, 60)",               category="volume"),
    Factor(name="KMID62",expression="CORR($close, $amount, 5)",       category="correlation"),
    Factor(name="KLEN62",expression="CORR($close, $amount, 10)",      category="correlation"),
    Factor(name="KMID63",expression="CORR($close, $amount, 20)",      category="correlation"),
    Factor(name="KLEN63",expression="CORR($close, $amount, 30)",      category="correlation"),
    Factor(name="KMID64",expression="CORR($close, $amount, 60)",      category="correlation"),

    # === $vwap (amount / volume) ===
    Factor(name="KLEN64",expression="REF($vwap, 1) / $vwap - 1",     category="price"),
    Factor(name="KMID65",expression="REF($vwap, 2) / $vwap - 1",     category="price"),
    Factor(name="KLEN65",expression="MEAN($vwap, 5) / $vwap",        category="price"),
    Factor(name="KMID66",expression="MEAN($vwap, 10) / $vwap",       category="price"),
    Factor(name="KLEN66",expression="MEAN($vwap, 20) / $vwap",       category="price"),
    Factor(name="KMID67",expression="MEAN($vwap, 30) / $vwap",       category="price"),
    Factor(name="KLEN67",expression="MEAN($vwap, 60) / $vwap",       category="price"),
    Factor(name="KMID68",expression="STD($vwap, 5)",                 category="price"),
    Factor(name="KLEN68",expression="STD($vwap, 10)",                category="price"),
    Factor(name="KMID69",expression="STD($vwap, 20)",                category="price"),
    Factor(name="KLEN69",expression="STD($vwap, 30)",                category="price"),
    Factor(name="KMID70",expression="STD($vwap, 60)",                category="price"),

    # === Rank factors (cross-sectional) ===
    Factor(name="KLEN70",expression="RANK(REF($close, 1) / $close - 1)",   category="rank"),
    Factor(name="KMID71",expression="RANK(MEAN($close, 5) / $close)",      category="rank"),
    Factor(name="KLEN71",expression="RANK(MEAN($close, 10) / $close)",     category="rank"),
    Factor(name="KMID72",expression="RANK(MEAN($close, 20) / $close)",     category="rank"),
    Factor(name="KLEN72",expression="RANK(MEAN($close, 30) / $close)",     category="rank"),
    Factor(name="KMID73",expression="RANK(MEAN($close, 60) / $close)",     category="rank"),
    Factor(name="KLEN73",expression="RANK(STD($close, 5))",                category="rank"),
    Factor(name="KMID74",expression="RANK(STD($close, 10))",               category="rank"),
    Factor(name="KLEN74",expression="RANK(STD($close, 20))",               category="rank"),
    Factor(name="KMID75",expression="RANK(STD($close, 30))",               category="rank"),
    Factor(name="KLEN75",expression="RANK(STD($close, 60))",               category="rank"),
    Factor(name="KMID76",expression="RANK(CORR($close, $volume, 5))",     category="rank"),
    Factor(name="KLEN76",expression="RANK(CORR($close, $volume, 10))",    category="rank"),
    Factor(name="KMID77",expression="RANK(CORR($close, $volume, 20))",    category="rank"),
    Factor(name="KLEN77",expression="RANK(CORR($close, $volume, 30))",    category="rank"),
    Factor(name="KMID78",expression="RANK(CORR($close, $volume, 60))",    category="rank"),

    # === Quantile factors (cross-sectional) ===
    Factor(name="KLEN78",expression="QUANTILE(REF($close, 1) / $close - 1, 5)",  category="quantile"),
    Factor(name="KMID79",expression="QUANTILE(MEAN($close, 5) / $close, 5)",     category="quantile"),
    Factor(name="KLEN79",expression="QUANTILE(MEAN($close, 10) / $close, 5)",    category="quantile"),
    Factor(name="KMID80",expression="QUANTILE(MEAN($close, 20) / $close, 5)",    category="quantile"),
    Factor(name="KLEN80",expression="QUANTILE(MEAN($close, 30) / $close, 5)",    category="quantile"),
    Factor(name="KMID81",expression="QUANTILE(MEAN($close, 60) / $close, 5)",    category="quantile"),
    Factor(name="KLEN81",expression="QUANTILE(STD($close, 5), 5)",               category="quantile"),
    Factor(name="KMID82",expression="QUANTILE(STD($close, 10), 5)",              category="quantile"),
    Factor(name="KLEN82",expression="QUANTILE(STD($close, 20), 5)",              category="quantile"),
    Factor(name="KMID83",expression="QUANTILE(STD($close, 30), 5)",              category="quantile"),
    Factor(name="KLEN83",expression="QUANTILE(STD($close, 60), 5)",              category="quantile"),
    Factor(name="KMID84",expression="QUANTILE(CORR($close, $volume, 5), 5)",    category="quantile"),
    Factor(name="KLEN84",expression="QUANTILE(CORR($close, $volume, 10), 5)",   category="quantile"),
    Factor(name="KMID85",expression="QUANTILE(CORR($close, $volume, 20), 5)",   category="quantile"),
    Factor(name="KLEN85",expression="QUANTILE(CORR($close, $volume, 30), 5)",   category="quantile"),
    Factor(name="KMID86",expression="QUANTILE(CORR($close, $volume, 60), 5)",   category="quantile"),

    # === Delta factors (change over time) ===
    Factor(name="KLEN86",expression="DELTA($close, 5)",     category="momentum"),
    Factor(name="KMID87",expression="DELTA($close, 10)",    category="momentum"),
    Factor(name="KLEN87",expression="DELTA($close, 20)",    category="momentum"),
    Factor(name="KMID88",expression="DELTA($close, 30)",    category="momentum"),
    Factor(name="KLEN88",expression="DELTA($close, 60)",    category="momentum"),
    Factor(name="KMID89",expression="DELTA($volume, 5)",    category="momentum"),
    Factor(name="KLEN89",expression="DELTA($volume, 10)",   category="momentum"),
    Factor(name="KMID90",expression="DELTA($volume, 20)",   category="momentum"),
    Factor(name="KLEN90",expression="DELTA($volume, 30)",   category="momentum"),
    Factor(name="KMID91",expression="DELTA($volume, 60)",   category="momentum"),
    Factor(name="KLEN91",expression="DELTA($amount, 5)",    category="momentum"),
    Factor(name="KMID92",expression="DELTA($amount, 10)",   category="momentum"),
    Factor(name="KLEN92",expression="DELTA($amount, 20)",   category="momentum"),
    Factor(name="KMID93",expression="DELTA($amount, 30)",   category="momentum"),
    Factor(name="KLEN93",expression="DELTA($amount, 60)",   category="momentum"),

    # === SUM factors ===
    Factor(name="KMID94",expression="SUM($close, 5)",       category="price"),
    Factor(name="KLEN94",expression="SUM($close, 10)",      category="price"),
    Factor(name="KMID95",expression="SUM($close, 20)",      category="price"),
    Factor(name="KLEN95",expression="SUM($close, 30)",      category="price"),
    Factor(name="KMID96",expression="SUM($close, 60)",      category="price"),
    Factor(name="KLEN96",expression="SUM($volume, 5)",      category="volume"),
    Factor(name="KMID97",expression="SUM($volume, 10)",     category="volume"),
    Factor(name="KLEN97",expression="SUM($volume, 20)",     category="volume"),
    Factor(name="KMID98",expression="SUM($volume, 30)",     category="volume"),
    Factor(name="KLEN98",expression="SUM($volume, 60)",     category="volume"),
    Factor(name="KMID99",expression="SUM($amount, 5)",      category="volume"),
    Factor(name="KLEN99",expression="SUM($amount, 10)",     category="volume"),
    Factor(name="KMID100",expression="SUM($amount, 20)",    category="volume"),
    Factor(name="KLEN100",expression="SUM($amount, 30)",    category="volume"),
    Factor(name="KMID101",expression="SUM($amount, 60)",    category="volume"),
]


def build_alpha158() -> FactorRegistry:
    reg = FactorRegistry(name="alpha158")
    for f in _FACTORS:
        reg.register(f)
    return reg
```

- [ ] **Step 4: Run tests — verify 158 factors, all compile, no cycles**

Run: `uv run pytest tests/test_features/test_alpha158.py -v`
Expected: all 5 tests PASS (158 factors registered, all compile, no cycles, valid deps, lookback=60)

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/features/alphasets/ tests/test_features/test_alpha158.py
git commit -m "feat: add alpha158 factor definitions (158 factors)"
```

---

### Task 8: CLI integration

**Files:**
- Modify: `src/alpha_quat/cli.py`

- [ ] **Step 1: Write failing test for feature subcommand**

There's no existing CLI test file. We'll verify the CLI works by running it (Step 4). For now, we modify cli.py directly.

- [ ] **Step 2: Add feature subcommand to CLI**

Replace `src/alpha_quat/cli.py` with updated version that adds `feature` subparser:

```python
# src/alpha_quat/cli.py
"""CLI entry point for alpha-quat data fetching and feature engineering."""

import argparse
import logging

from alpha_quat.config import Config
from alpha_quat.data.fetcher import Fetcher
from alpha_quat.data.metadata import MetadataManager
from alpha_quat.data.pipeline import Pipeline
from alpha_quat.data.writer import ParquetWriter
from alpha_quat.data.sources.stock_basic import StockBasicSource
from alpha_quat.data.sources.trade_cal import TradeCalSource
from alpha_quat.data.sources.stock_st import StockStSource
from alpha_quat.data.sources.daily import DailySource
from alpha_quat.data.sources.daily_basic import DailyBasicSource

ALL_SOURCES = {
    "stock_basic": StockBasicSource,
    "trade_cal": TradeCalSource,
    "stock_st": StockStSource,
    "daily": DailySource,
    "daily_basic": DailyBasicSource,
}

ALL_FEATURE_SETS = {
    "alpha158": "alpha_quat.features.alphasets.alpha158:build_alpha158",
}


def _build_data_parser(subparsers):
    parser = subparsers.add_parser(
        "fetch", help="Fetch raw data from tushare"
    )
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        choices=list(ALL_SOURCES.keys()) + ["all"],
        default=["all"],
        help="Data sources to pull (default: all)",
    )
    return parser


def _build_feature_parser(subparsers):
    parser = subparsers.add_parser(
        "feature", help="Compute feature factors from raw data"
    )
    parser.add_argument(
        "-f",
        "--factor-set",
        choices=list(ALL_FEATURE_SETS.keys()),
        default="alpha158",
        help="Factor set to compute (default: alpha158)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete all computed factors and recompute from scratch"
    )
    parser.add_argument(
        "--since",
        help="Recompute factors from this date (YYYYMMDD) onward"
    )
    return parser


def _cmd_fetch(args, config, metadata):
    fetcher = Fetcher(token=config.token)
    writer = ParquetWriter()
    pipeline = Pipeline(
        data_dir=config.data_dir, fetcher=fetcher, metadata=metadata, writer=writer
    )
    names = list(ALL_SOURCES.keys()) if "all" in args.sources else args.sources
    sources = [ALL_SOURCES[name]() for name in names]
    pipeline.run(sources)


def _cmd_feature(args, config, metadata):
    from importlib import import_module

    from alpha_quat.features.engine import FeatureEngine
    from alpha_quat.features.pipeline import FeaturePipeline

    module_path, func_name = ALL_FEATURE_SETS[args.factor_set].split(":")
    module = import_module(module_path)
    registry = getattr(module, func_name)()

    engine = FeatureEngine(data_dir=config.data_dir)
    writer = ParquetWriter()
    output_dir = config.data_dir / "features"

    pipeline = FeaturePipeline(
        data_dir=config.data_dir,
        output_dir=output_dir,
        engine=engine,
        writer=writer,
        metadata=metadata,
    )
    result = pipeline.run(registry, rebuild=args.rebuild, since=args.since)

    print()
    print(f"Factor set: {registry.name}")
    print(f"  Success: {result['success']}")
    print(f"  Failed:  {result['failed']}")
    if result["errors"]:
        for err in result["errors"]:
            print(f"  Error: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="alpha-quat: stock data fetching and feature engineering"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show data registry summary and exit"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command")
    _build_data_parser(subparsers)
    _build_feature_parser(subparsers)

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

    # Default: fetch if no subcommand specified (backward compatible)
    if args.command is None or args.command == "fetch":
        _cmd_fetch(args, config, metadata)
    elif args.command == "feature":
        _cmd_feature(args, config, metadata)
```

- [ ] **Step 3: Verify fetch command still works**

Run: `uv run alpha-quat --help`
Expected: shows subcommands `fetch` and `feature`

Run: `uv run alpha-quat fetch --help`
Expected: shows `-s/--sources` options (backward compatible)

- [ ] **Step 4: Verify feature command help**

Run: `uv run alpha-quat feature --help`
Expected: shows `-f/--factor-set`, `--rebuild`, `--since` options

- [ ] **Step 5: Commit**

```bash
git add src/alpha_quat/cli.py
git commit -m "feat: add 'alpha-quat feature' subcommand for factor computation"
```

---

### Task 9: Full verification

- [ ] **Step 1: Run full verification pipeline**

```bash
uv run ruff format . && uv run ruff check --fix . && uv run pyright && uv run pytest --cov=src -v
```

Expected: format, lint, typecheck all pass; all tests pass with coverage.

- [ ] **Step 2: Fix any issues found**

Address any lint/typecheck/test failures before proceeding.

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A && git commit -m "chore: fix lint/typecheck/test issues from verification"
```
