# Transformer Support/Resistance Model Design

## Overview

Train a Transformer model on 60-day sequences of raw OHLCV data (Alpha360) to
predict probability distributions of future support and resistance levels. Instead
of ranking stocks for top-K holding, the model enables a timing-based strategy
that can hold cash when no good opportunities exist.

## Label Algorithm

For each stock and each day n, compute support/resistance levels from the
next 60 days of price action. A "true" resistance level is a local price peak
where the stock was rejected (went up, then came back down). A "true" support
level is a local price trough where the stock bounced.

### Algorithm

```
1. For day n, get future 60-day high/low series: H[1:61], L[1:61]
2. Scan H for local peaks:
   - Day d is a peak if H[d] == max(H[d-2:d+3]) and H[d] > close[n]
   - Confirm rejection: at least one day in H[d+1:d+3] is < H[d] * 0.99
3. Classify peaks by distance from n:
   - d ≤ 5  → 5d resistance
   - d ≤ 20 → 20d resistance
   - d ≤ 60 → 60d resistance
4. Same for L → local troughs → support levels
5. Map each detected level to a probability distribution:
   - Discretize price range [close*0.8, close*1.2] into 100 bins
   - Place Gaussian kernel (σ=2 bins) at detected level's bin
   - If multiple levels, sum and renormalize
   - If no level detected, output uniform distribution
```

### Implementation

```python
# model/nn/transformer/labels.py
def compute_sr_labels(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    n: int, window=60, n_bins=100, price_range=0.20
) -> dict[str, np.ndarray]:
```

Returns 6 arrays of shape [100]: resistance_5d, resistance_20d, resistance_60d,
support_5d, support_20d, support_60d.

## Model Architecture

### Input

```
Shape: (batch, seq_len=60, n_features=6)
Features: open, high, low, close, volume (log), vwap
```

### Transformer

```
Input (B, 60, 6)
  → Linear(6 → 128) + LayerNorm
  → TransformerEncoder × 4 (d_model=128, nhead=4, dim_feed=512, dropout=0.1)
  → Global Avg Pooling (B, 128)
  → Linear(128 → 600) → reshape (B, 6, 100)
  → Softmax over dim=-1

Output:
  [0] resistance_5d   — probability over 100 price bins
  [1] resistance_20d
  [2] resistance_60d
  [3] support_5d
  [4] support_20d
  [5] support_60d
```

### Loss

Cross-entropy per head, averaged across 6 heads.

## Data Pipeline

### Daily data source

Read from `daily/YYYY_MM_DD.parquet` directly (not from `features/`). Each file
has: ts_code, open, high, low, close, vol (volume), amount.

Compute vwap on the fly: `vwap = amount / NULLIF(vol, 0)`.

### SequenceDataset

```
For each ts_code:
  1. Sort by trade_date
  2. Create sliding windows: size=60, stride=20
  3. For each window, extract 6 features → (60, 6)
  4. Compute SR labels from next 60 days → (6, 100)

Returns:
  X: (n_samples, 60, 6)   — float32
  y: (n_samples, 6, 100)  — float32 (probability distributions)
```

### Normalization

Per-feature z-score normalization computed across the training set:
- For price features (open, high, low, close, vwap): normalize by mean/std
- For volume: log transform first, then z-score
- Normalization parameters saved with the model to apply during inference

### Universe filter

Same as existing: main board (`stock_basic.market == "主板"`), exclude ST stocks.

## Training

### Config

```python
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
    lr_scheduler: str = "cosine"
    early_stopping_patience: int = 10

    # Date ranges
    train_start: str = "20200101"
    train_end: str = "20231231"
    val_start: str = "20240101"
    val_end: str = "20240630"
```

### Training loop

```
For each epoch:
  For each batch (features, labels):
    logits = model(features)    # (B, 6, 100)
    loss = cross_entropy(logits.view(-1, 100), labels.view(-1, 100))
    backward()

  Validation:
    - Cross-entropy per horizon
    - Top-3 bin accuracy: is the true level bin in top 3 predicted bins?
    - Distribution sharpness: avg entropy (lower = sharper predictions)

  Early stop on val loss
```

### Evaluation

Beyond loss/accuracy, evaluate on trading signal quality:
- For each val date, compute entry/exit signals from predictions
- Backtest a simple strategy (see below)
- Track: win rate, avg return per trade, max consecutive losses

## Signal Generation

```python
# strategy/signals/variants/transformer_sr_signal.py
class TransformerSRSignal(BaseMLSignal):
    mode = "transformer_sr"

    def generate(self, features, ctx) -> SignalResult:
        """
        1. Build 60-day sequence for each stock from raw daily data
        2. Model inference → 6 distributions per stock
        3. Compute entry/exit metrics:
           - support_strength = prob mass in [-3%, -1%] bins
           - resistance_strength = prob mass in [+1%, +3%] bins
           - rr_ratio = expected_up / expected_down
        4. If rr_ratio > 2.0 and support_strength > 0.3 → entry signal
        5. If resistance_strength > 0.5 → exit signal
        """
```

## Backtest Strategy

New backtest mode that does NOT use top-K rebalance logic:

```
For each trading day:
  For each stock in universe:
    sr_signal = model.generate(60-day sequence)

    if stock not held and sr_signal.entry:
      position_size = capital * min(rr_ratio / 5, 0.25)  # size by conviction
      buy(stock, position_size)

    if stock held:
      if sr_signal.exit:
        sell(stock)
      elif close < position.stop_loss:
        sell(stock)  # dynamic stop-loss

  Remaining capital → cash (no forced holding)
```

## File Structure

```
src/alpha_quat/model/nn/
├── __init__.py
├── config.py              # TransformerConfig
├── pipeline.py            # run_variant_nn() factory

src/alpha_quat/model/nn/transformer/
├── __init__.py
├── labels.py              # compute_sr_labels()
├── models/
│   ├── __init__.py
│   ├── dataset.py         # StockSequenceDataset
│   └── transformer.py     # StockTransformer model
├── train.py               # train() function
├── evaluate.py            # evaluation on val set
└── variants/
    ├── __init__.py
    ├── base.py            # NNBasePipeline(ABC)
    └── sr_lstm.py         # SRTransformerPipeline (using Transformer)

src/alpha_quat/strategy/signals/variants/
    └── transformer_sr_signal.py  # TransformerSRSignal
```

## CLI

```bash
# Train
uv run alpha-quat model nn sr_transformer --name exp_sr_v1

# Backtest (uses experiment config to load model + select signal)
uv run alpha-quat backtest --experiment exp_sr_v1
```

## Dependencies

```toml
"torch>=2.0",
```

## Testing

- `test_transformer_labels.py` — SR label algorithm correctness
- `test_transformer_dataset.py` — sequence generation, normalization
- `test_transformer_model.py` — forward pass shape, loss computation
- `test_transformer_pipeline.py` — end-to-end training with synthetic data
- `test_transformer_sr_signal.py` — signal generation from model output
