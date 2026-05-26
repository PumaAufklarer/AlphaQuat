# Transformer SR Model Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement task-by-task.

**Goal:** Build Transformer-based support/resistance probability prediction model with
Alpha360 (6 raw OHLCV fields × 60 days), enabling timing-based trading strategies.

**Architecture:** New `model/nn/` package with Transformer variant, new signal variant
for support/resistance-based trading signals, and a new backtest mode.

**Tech Stack:** Python, PyTorch 2.0+, DuckDB, numpy

---

### Prerequisites

- [ ] **Add torch dependency**

```bash
uv add torch>=2.0
```

---

### Task 1: NN package scaffold + config + base pipeline

**Files:**
- Create: `src/alpha_quat/model/nn/__init__.py`
- Create: `src/alpha_quat/model/nn/config.py` — TransformerConfig dataclass
- Create: `src/alpha_quat/model/nn/pipeline.py` — run_variant_nn() factory
- Create: `tests/test_model/test_nn/test_config.py`

```python
# config.py
@dataclass
class TransformerConfig:
    # Data
    seq_length: int = 60
    stride: int = 20
    n_features: int = 6
    n_bins: int = 100
    price_range: float = 0.20

    # Model
    d_model: int = 128
    nhead: int = 4
    n_layers: int = 4
    dim_feed: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10

    # Date ranges
    train_start: str = "20200101"
    train_end: str = "20231231"
    val_start: str = "20240101"
    val_end: str = "20240630"
```

### Task 2: Labels module — SR algorithm

**Files:**
- Create: `src/alpha_quat/model/nn/transformer/__init__.py`
- Create: `src/alpha_quat/model/nn/transformer/labels.py`
- Create: `tests/test_model/test_nn/__init__.py`
- Create: `tests/test_model/test_nn/test_labels.py`

Implement the core label algorithm. Test with synthetic price data that has
clear support/resistance levels.

### Task 3: Dataset module

**Files:**
- Create: `src/alpha_quat/model/nn/transformer/models/__init__.py`
- Create: `src/alpha_quat/model/nn/transformer/models/dataset.py`
- Create: `tests/test_model/test_nn/test_dataset.py`

Reads daily parquet files, groups by stock, creates sliding windows,
computes SR labels, normalizes features.

### Task 4: Transformer model

**Files:**
- Create: `src/alpha_quat/model/nn/transformer/models/transformer.py`
- Create: `tests/test_model/test_nn/test_model.py`

PyTorch Transformer model with 4 encoder layers, 6 output heads.

### Task 5: Training module

**Files:**
- Create: `src/alpha_quat/model/nn/transformer/train.py`
- Create: `tests/test_model/test_nn/test_train.py`

Training loop with AdamW, cosine scheduler, early stopping.

### Task 6: Evaluate module

**Files:**
- Create: `src/alpha_quat/model/nn/transformer/evaluate.py`
- Create: `tests/test_model/test_nn/test_evaluate.py`

Evaluation metrics: cross-entropy, top-3 accuracy, distribution sharpness.

### Task 7: SR variant pipeline

**Files:**
- Create: `src/alpha_quat/model/nn/transformer/variants/__init__.py`
- Create: `src/alpha_quat/model/nn/transformer/variants/base.py`
- Create: `src/alpha_quat/model/nn/transformer/variants/sr_transformer.py`
- Register in `model/nn/pipeline.py`

### Task 8: Transformer SR signal variant

**Files:**
- Create: `src/alpha_quat/strategy/signals/variants/transformer_sr_signal.py`
- Register in signal variants

### Task 9: Update CLI

- Add `model nn sr_transformer --name` subcommand
- Add `--experiment` to backtest

### Task 10: Update BacktestEngine for SR signals

- New backtest mode: SR-based entry/exit (not top-K)
- Add `sr_based` strategy option to BacktestConfig
