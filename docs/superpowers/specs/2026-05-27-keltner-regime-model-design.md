# Keltner Channel Regime Prediction Model

## Problem

Current SR Transformer predicts support/resistance price levels. This has two structural flaws:
1. **Support/resistance are historical** — they describe where price DID bounce, not where it will
2. **Price targets are weak signals** — predicting exact future prices from 100-bin distributions has high entropy/low accuracy

New approach: predict **market regime** (what the price is doing relative to a Keltner Channel), then derive trading signals from regime transitions.

## Keltner Channel Definition

- Center: EMA(close, 20)
- Range: ATR(20) × 2.0
- Upper: EMA + ATR × 2.0
- Lower: EMA − ATR × 2.0
- Position: `(close − EMA) / (ATR × 2.0)` ∈ [-∞, +∞] (theoretically; in practice ±2.0 covers 99% of data)

Features per day per stock (stored in alpha360 cache):
| Column | Formula |
|--------|---------|
| `k_pos` | `(close − EMA20) / (ATR20 × 2.0)` — channel position |
| `k_width` | `ATR20 / EMA20` — normalized volatility |
| `k_above_ema` | `close / EMA20 − 1` — quick trend bias |

## Five Regime States

Mutually exclusive, ordered by priority (higher priority wins):

| ID | Name | Condition | Priority |
|----|------|-----------|----------|
| 0 | Ranging | `k_pos ∈ (−0.5, 0.5)` | 3 |
| 1 | Support Test | `k_pos ∈ [−1.0, −0.5]` | 2 |
| 2 | Resistance Test | `k_pos ∈ [0.5, 1.0]` | 2 |
| 3 | Breakout Up | `k_pos > 1.0` | 1 |
| 4 | Breakout Down | `k_pos < −1.0` | 1 |

Priorities: a breakout state always wins over a test state (a stock can't be both "testing resistance" and "breaking out" — breakout means it's ALREADY outside the channel).

States 1-2 are **warning** states (price probing boundaries).
States 3-4 are **action** states (price exiting the channel).

## Label Generation

For each stock/day t and each horizon N ∈ {5, 20, 60}:
- Look at price data at day t+N
- Compute Keltner channel at day t+N
- Assign label = regime ID based on condition table above
- If NaN (insufficient data): skip sample

Labels are **future** states — the model sees 60 days of history and predicts what regime the stock will be in N days from now.

Labels are computed at dataset construction time (not stored in alpha360).

## Model Architecture

### Input
- Sequence: (60, 17) — existing 14 features + 3 Keltner features
- Same per-sequence normalization (price ratios, log-vol z-score, new-features z-score)

### Output
- 3 heads × 5 classes = 15 logits per sample
- Each head: (B, 5) → softmax → 5-class probability distribution
- Heads: 5d, 20d, 60d horizon regimes

### Loss
- CrossEntropyLoss per head (no label smoothing, no weighting)
- Total = arithmetic mean of 3 head losses

### Model
StockTransformer modified to output (B, 3, 5) instead of (B, 6, 100):
- `self.head = nn.Linear(d_model, 3 * n_regimes)`
- `n_regimes = 5`
- Same transformer encoder backbone

## Dataset

`KeltnerRegimeDataset` (new file): `models/keltner/dataset.py`
- Loads alpha360 cache
- Computes Keltner features per stock
- Builds sequences: X=(60, 17), y=(3,) int64, no weights needed
- Per-sequence normalization same as SR
- Skip sequences where any horizon label can't be computed (insufficient future data)

## Training

Same pipeline as SR Transformer:
- AdamW, warmup-cosine schedule
- Batch size 128
- Early stopping patience 10
- Evaluate on val set (macro F1 per horizon, accuracy per horizon)

## Inference & Trading Signals

For a stock today (60-day sequence → model → 3 horizon predictions):

| Predicted Regime (5d horizon) | Signal |
|-------------------------------|--------|
| Ranging (0) | Hold / do nothing |
| Support Test (1) | Prepare to buy — set limit orders near lower keltner |
| Resistance Test (2) | Prepare to sell — set limit orders near upper keltner |
| Breakout Up (3) | Buy — momentum following |
| Breakout Down (4) | Sell / short — momentum breakdown |

## Evaluation

- Per-horizon accuracy, macro F1
- Confusion matrix (shows which transitions the model confuses)
- Compare with naive baseline (always predict "Ranging")

## Implementation Plan

1. **sr_cache.py**: Add keltner feature computation to `_compute_derived_features()` + new columns
2. **dataset.py**: New `_compute_keltner_labels()` + update `_build_sequences` for (3,) labels
3. **transformer.py**: New `KeltnerTransformer(n_regimes=5)` or modify `StockTransformer`
4. **train_keltner.py**: New training script with CrossEntropyLoss
5. **evaluate_keltner.py**: New eval with F1/confusion matrix
6. **inference_keltner.py**: New inference + signal generator
7. **CLI**: Add `alpha-quat model nn keltner --name exp_keltner_v1`
