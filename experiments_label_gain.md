# LambdaRank 标签构造实验

## 实验设计

三个变种，每个用 Optuna 30 次搜索调参，同 seed=42：

| 变种 | NTILE | label_gain | 描述 |
|------|-------|-----------|------|
| baseline | 10 | [0,1,2,...,9] 线性 | 当前基线 |
| expgain | 10 | [0,1,2,4,7,11,16,23,31,42] 指数 | top~42:1 权重 |
| ntile5 | 5 | [0,1,2,3,4] 线性 | 更少分桶 |

训练：2023-2024 / 验证：2024-2025 / 回测：2024.07~2026.05

## 结果

| 指标 | baseline | expgain | ntile5 |
|------|---------|---------|--------|
| 5d IC | 0.1189 | 0.0940 | 0.1073 |
| 20d IC | 0.1335 | 0.1152 | 0.1396 |
| 60d IC | 0.0392 | 0.0770 | 0.0250 |
| **夏普** | **1.29** | 0.07 | 0.36 |
| 累计收益 | **+42.33%** | +0.04% | +2.37% |
| 最大回撤 | -15.77% | -6.27% | -7.34% |
| 交易数 | 508 | 618 | 192 |
| 胜率 | 59.0% | 50.2% | 57.8% |

## 结论

1. **原始线性 gain + NTILE=10 是最优解**。夏普 1.29，42% 累计收益。
2. **指数增益完全失败**（0.07 夏普）。极端权重导致模型忽略 90% 样本。
3. **n_tile=5 不如 n_tile=10**。分桶太少导致排序精度不足，交易数仅 192（持有太久）。

## 复现命令

```bash
uv run alpha-quat model lightgbm lambdarank \
  --name exp_lbl_baseline \
  --train-start 20230101 --train-end 20240630 \
  --val-start 20240701 --val-end 20250430 \
  --trials 30

uv run alpha-quat backtest \
  --experiment exp_lbl_baseline \
  --start 20240701 --end 20260501 \
  --capital 50000 --monthly 8000 \
  --top-k 15 --rebalance-interval 5 \
  --weighting score_momentum
```

## 超参数（baseline 最优 trial）

| 参数 | 5d | 20d | 60d |
|------|-----|------|------|
| num_leaves | 60 | 46 | 57 |
| learning_rate | 0.0102 | 0.0289 | 0.0137 |
| n_estimators | 367 | 320 | 422 |
| feature_fraction | 0.998 | 0.546 | 0.881 |
| bagging_fraction | 0.509 | 0.555 | 0.576 |
