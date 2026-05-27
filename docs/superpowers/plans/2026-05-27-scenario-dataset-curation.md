# Scenario Dataset Curation Plan

## Overview

Replace current "future Keltner position" labeling with **current market scenario** labels.
Scenarios are defined by combining Keltner position + volume regime + EMA slope.
Dataset is curated (not all days) — each scenario class sampled equally.

## Current Data Distribution (Keltner position only, Jan 2024)

| Range | % of days | Possible Scenarios |
|-------|----------|-------------------|
| Ranging (pos ∈ [-0.3, 0.3]) | 26.2% | ranging |
| Upper half (pos ∈ (0.3, 1.0)) | 11.0% | resistance_approach, trend_preaccel |
| Lower half (pos ∈ [-1.0, -0.3)) | 39.4% | support_approach |
| Below channel low vol | 15.4% | support_bounce |
| Below channel high vol | 6.0% | false_breakdown |
| Above channel high vol | 1.7% | valid_breakout |
| Above channel low vol | 0.4% | resistance_rejection |

## Six Scenario Classes

| ID | Name | Condition | Represents | Trading Signal |
|----|------|-----------|-----------|---------------|
| 0 | Ranging | kpos ∈ [-0.3, 0.3] | No clear direction | Hold |
| 1 | Support Bounce | kpos < -1.0, vol > 1.0, close > open | Genuine bounce off channel bottom | Buy |
| 2 | False Breakdown | kpos < -1.0, vol > 1.3, close ≤ open | Breakdown w/ continuation risk | Avoid/Stop |
| 3 | Resistance Rejection | kpos > 1.0, vol ≤ 1.3 | Price above channel w/o confirmation | Sell/Nothing |
| 4 | Valid Breakout | kpos > 1.0, vol > 1.3 | Confirmed breakout above channel | Buy/Momentum |
| 5 | Trend Pre-Accel | kpos ∈ (0.3, 0.8), EMA_slope > 0, vol > 1.0 | EMAslope↑ + vol↑ → preparing for move | Position |

Note: EMA_slope ≈ (keltner_above_ema[t] - keltner_above_ema[t-5]) / 5
  - Positive = price moving away from EMA (trending)
  - Negative = price moving toward EMA (reverting)

## Labeling Logic (Priority Order)

```python
def scenario_label(kpos, vol, close, open_px, ema_slope, hl_ratio):
    green = close >= open_px

    # 4: Valid Breakout — price clearly above channel with volume
    if kpos > 1.0 and vol > 1.3:
        return 4

    # 3: Resistance Rejection — above channel but reversing
    if kpos > 1.0 and (not green or vol <= 1.3):
        return 3

    # 2: False Breakdown — below channel, heavy selling
    if kpos < -1.0 and vol > 1.3 and not green:
        return 2

    # 1: Support Bounce — below channel, buying comes in
    if kpos < -1.0 and green:
        return 1

    # 5: Trend Pre-Accel — upper half, EMA pulling away, vol picking up
    if 0.3 < kpos < 0.8 and not pd.isna(ema_slope) and ema_slope > 0.005 and vol > 1.0:
        return 5

    # 0: Ranging
    return 0
```

## Dataset Curation Strategy

Goal: ~60k total samples, ~10k per class (class 2-4-5 may have fewer, use what exists).

### Step 1: Full scan → raw label counts
Scan all stocks/dates, assign scenario labels. Count per class.

### Step 2: Sample per stock per class
- For each stock, for each class label, keep at most K samples (e.g., K=5, tighter if few stocks)
- This prevents one stock dominating a class

### Step 3: Downsample majority classes
- Determine the smallest class size → target sample count per class
- For classes 0-1-3 (which may be large), randomly sample to match target

### Step 4: Verify label quality
- For each class, check that labels are "sensible" (e.g., support_bounce → price bounced within 5d)
- Remove any samples where the labeling was ambiguous

## No Horizon Prediction

Labels describe the CURRENT state (last day of the 60-day window).
No forward-looking horizon needed.
The model's output at inference time IS the current scenario prediction.

This eliminates the 20d/60d degeneracy problem entirely.

## Feature Input

Same 14 OHLCV-derived features (NO keltner_pos/width/above_ema as features).
SR distance columns optional (can be added later if useful).

## Model

- Input: (60, 14)
- Output: (6,) — logits for 6 scenario classes
- Loss: CrossEntropyLoss with class weights (inverse frequency)
- Evaluation: macro F1, per-class precision/recall, confusion matrix

## Implementation Order

1. Write scenario labeling function + data scanner in `dataset.py`
2. Implement curated dataset builder (downsample by class)
3. Update model to (60, 14) → (6,) single-head output
4. Train and evaluate
5. Compare with random baseline and existing models
