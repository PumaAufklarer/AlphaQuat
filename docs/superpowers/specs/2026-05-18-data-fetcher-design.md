# 数据拉取模块设计

## 概述

从 tushare 拉取日频股票数据，统一保存为 Parquet，用 duckdb 维护数据目录元信息。

## 数据源

| 优先级 | API | 拉取模式 | 分区策略 |
|--------|-----|---------|---------|
| 1 | stock_basic | 全量覆盖 | 单文件 `data/stock_basic.parquet` |
| 2 | trade_cal | 全量覆盖 | 单文件 `data/trade_cal.parquet` |
| 3 | stk_st | 增量追加 | 按日期 `data/stk_st/YYYY_MM_DD.parquet` |
| 4 | daily | 增量追加 | 按日期 `data/daily/YYYY_MM_DD.parquet` |
| 5 | daily_basic | 增量追加 | 按日期 `data/daily_basic/YYYY_MM_DD.parquet` |

- 增量型：首次运行时全量拉取历史所有交易日，之后只拉取未覆盖的最新交易日
- 全量型：每次运行覆盖写入单个文件

## 架构

```
config.yaml ──→ Config
                   │
                   ▼
              Pipeline ──→ MetadataManager (duckdb)
                   │            │
                   ▼            │
              DataSource ───────┘  (查上次拉取状态)
             (抽象基类)
                   │
                   ▼
              Fetcher (tushare 调用 + 重试)
                   │
                   ▼
              ParquetWriter (分区写入)
```

### 组件

| 组件 | 文件 | 职责 |
|------|------|------|
| Config | `config.py` | 读取 config.yaml，暴露 token、data_dir |
| DataSource | `data/source.py` | 抽象基类：api_name、分区策略、params 构造 |
| Fetcher | `data/fetcher.py` | 封装 tushare query，含重试（最多3次，间隔5秒） |
| ParquetWriter | `data/writer.py` | DataFrame → Parquet，按分区策略写出到正确路径 |
| MetadataManager | `data/metadata.py` | duckdb 中维护 data_registry 表，查询/写入拉取记录 |
| Pipeline | `data/pipeline.py` | 串联流程：断点查询 → 日期推断 → 拉取 → 写入 → 元信息更新 |

### DataSource 基类

```python
class DataSource(ABC):
    api_name: str                          # tushare API 名
    partition_by: Literal["none", "date"]  # none=单文件覆盖, date=按日分片
    fields: str                            # tushare fields 参数

    @abstractmethod
    def get_params(self, trade_date: str) -> dict:
        """给定交易日，返回 tushare query 的 kwargs"""
        ...

    def path_for(self, trade_date: str | None = None) -> Path:
        """返回 Parquet 文件路径"""
        ...
```

### 子类

| 子类 | api_name | partition_by | 特殊说明 |
|------|----------|-------------|---------|
| StockBasicSource | stock_basic | none | list_status='L' |
| TradeCalSource | trade_cal | none | exchange='SSE' |
| StkStSource | stk_st | date | - |
| DailySource | daily | date | - |
| DailyBasicSource | daily_basic | date | - |

## 目录结构

```
data/
├── stock_basic.parquet          # 每次全量覆盖
├── trade_cal.parquet            # 每次全量覆盖
├── daily/
│   ├── 2026_05_15.parquet
│   └── 2026_05_16.parquet
├── daily_basic/
│   ├── 2026_05_15.parquet
│   └── 2026_05_16.parquet
└── stk_st/
    ├── 2026_05_15.parquet
    └── 2026_05_16.parquet
```

## 数据流

```
Pipeline.run(sources=["stock_basic", "trade_cal", "daily", "daily_basic", "stk_st"])
│
├─ 全量覆盖型 (stock_basic, trade_cal):
│   Fetcher.query(api) → DataFrame → ParquetWriter.overwrite(path)
│   → MetadataManager.upsert()
│
├─ 增量型 (daily, daily_basic, stk_st):
│   1. MetadataManager.get_last_date(api_name) → "20260515" 或 None
│   2. 从 trade_cal.parquet 读取全部交易日列表
│   3. 过滤出 > last_date 的日期
│   4. for each date:
│       Fetcher.query(api, trade_date=date)
│       → ParquetWriter.write(data/{api}/{date}.parquet)
│       → MetadataManager.insert()
│   5. 汇总成功/失败报告
│
└─ 错误处理：单天失败 → 记录并跳过，继续下一天，末尾汇总失败日期
```

## duckdb 元信息

数据库文件：`data/registry.db`

```sql
CREATE TABLE IF NOT EXISTS data_registry (
    id          INTEGER PRIMARY KEY,
    api_name    VARCHAR NOT NULL,
    trade_date  DATE,
    file_path   VARCHAR NOT NULL,
    row_count   INTEGER NOT NULL,
    pull_time   TIMESTAMP DEFAULT now(),
    UNIQUE(api_name, trade_date)
);
```

查询接口：
- `get_last_date(api_name)` → `SELECT MAX(trade_date) WHERE api_name=$1`
- `insert(api_name, date, path, rows)` → `INSERT ... ON CONFLICT REPLACE`
- `summary()` → `SELECT api_name, COUNT(*), MAX(trade_date) GROUP BY api_name`

## 配置

config.yaml（gitignored）：
```yaml
tushare:
  token: "xxx"
data:
  dir: "./data"
```

## 源码目录

```
src/alpha_quat/
├── __init__.py
├── config.py
├── cli.py
├── data/
│   ├── __init__.py
│   ├── source.py
│   ├── fetcher.py
│   ├── writer.py
│   ├── metadata.py
│   ├── pipeline.py
│   └── sources/
│       ├── __init__.py
│       ├── stock_basic.py
│       ├── trade_cal.py
│       ├── stk_st.py
│       ├── daily.py
│       └── daily_basic.py
tests/
├── conftest.py
├── test_config.py
├── test_fetcher.py
├── test_writer.py
├── test_metadata.py
├── test_pipeline.py
└── test_sources/
    ├── test_stock_basic.py
    ├── test_trade_cal.py
    ├── test_stk_st.py
    ├── test_daily.py
    └── test_daily_basic.py
```

## 测试策略

- Fetcher：mock tushare API 返回值，验证重试逻辑
- ParquetWriter：用 tmp_path fixture 验证文件路径和内容
- MetadataManager：用内存 duckdb 测试 CRUD
- Pipeline：集成测试，mock 掉 Fetcher，验证日期推断、增量逻辑、错误汇总
- DataSource 子类：验证 get_params 和 path_for 返回值

## 依赖

- `tushare` — 数据源
- `duckdb` — 元信息管理（需 `uv add duckdb`）
- `pyarrow` 或 `pandas` — Parquet 读写（pandas 已有 pyarrow 后端）
