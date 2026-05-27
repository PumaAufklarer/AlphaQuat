# RL Trading Agent — Per-Stock Continuous Position Control

## Overview

Replace SR price prediction and regime classification with a **trading agent**:
- Per-stock, per-day decision model
- Continuous position control [-1, 1]
- REINFORCE policy gradient training
- Dataset curated via SR + Keltner Channel (only informative days)
- Reward = volatility-normalized profit

## State

```
State_t = (market_data[t-59:t], current_position, days_in_position)
```

| Component | Shape | Source |
|-----------|-------|--------|
| Market data | (60, 14) | OHLCV + 8 derived features (no Keltner, no SR) |
| Current position | (1,) | Agent's previous decision, clipped to [-1, 1] |
| Days in position | (1,) | How many days since last position change |

Total input: (60, 15) — 14 features + position_signal + days_held appended as channels.

**Position signal embedding:** Repeat current_position across all 60 timesteps as an extra feature column. This tells the model "what we've been doing."

**Days in position embedding:** Similarly repeated, allows the model to learn "I've been holding too long."

### Features (14 columns)

Same as current alpha360 cache, EXCLUDING Keltner and SR columns:
```
open, high, low, close, volume, vwap,
volume_ratio, turnover_rate, hl_ratio, ret_5d, close_ma20,
atr_ratio, vol_change, amt_change
```

Keltner features and SR distances are NOT inputs. They are used only for dataset curation and reward computation.

## Action

```
a_t ∈ ℝ          → sampled from Gaussian policy
position_t = tanh(a_t) ∈ [-1, 1]
```

| Position | Meaning |
|----------|---------|
| 1.0 | Full long |
| 0.5 | Half long |
| 0.0 | Neutral / cash |
| -0.5 | Half short |
| -1.0 | Full short |

### Policy Model

```
State (60, 15) → Transformer → mean pooling → Linear(128, 1) → μ
σ = learnable parameter (globally shared, not per-stock)
a_t ∼ N(μ, σ)
position_t = tanh(a_t)
```

- Position is NOT constrained during sampling — tanh happens after sampling for the action bound
- σ is a single positive scalar, initialized to 0.3, trained via log_σ

## Reward

Computed at day t+N (N=5) for action taken at day t:

```
ret_N = close[t+N] / close[t] - 1
trade_cost = |position_t - position_{t-1}| × commission_rate
vol_adjusted_ret = ret_N / (atr_ratio[t] + 1e-6)
reward_t = position_t × vol_adjusted_ret - trade_cost
```

| Term | Rationale |
|------|-----------|
| `position_t × ret_N` | Profit from direction and magnitude |
| `/ atr_ratio` | Vol-normalize (high vol stocks should have larger moves) |
| `trade_cost` | Discourage excessive trading |

Commission rate: 0.0005 (5 bps), same as existing backtest.

## Dataset Curation

Not all days are equally informative. Use SR levels + Keltner Channel to filter.

### Filter Rules (per stock per day)

| Condition | Action | Rationale |
|-----------|--------|-----------|
| `kpos ∈ [-0.3, 0.3]` | Sample at 5% rate (1 in 20) | Mid-channel is noise, but need some "neutral" examples |
| `kpos > 0.5 or kpos < -0.5` | KEEP ALL | Channel boundary days are decision-relevant |
| `1.0 > kpos > 0.8 or -1.0 < kpos < -0.8` | KEEP ALL | Near-boundary overshoot |
| Any SR verified within 5 days (resistance/support distance < 5) | KEEP ALL | SR levels active |
| Volume ratio < 0.5 | DROP | Dead days |
| Turnover rate > 20% | DROP | Abnormal (suspension effect) |

### Dataset Construction

1. Scan all stocks, all dates
2. Apply filter rules → candidate set
3. For each stock, up to 5000 candidate days (truncate long-lived stocks)
4. Across all stocks, target ~500K samples total
5. No class balancing (the RL reward handles class imbalance naturally)

### Validation

- Use a separate time period (e.g., 2024H1)
- No filtering needed on validation — evaluate on ALL days
- Metrics: Sharpe ratio, cumulative return, max drawdown, win rate

## RL Training Loop

### Pseudocode

```python
def train_episode(stock_df, model, optimizer):
    """
    stock_df: DataFrame for one stock, sorted by trade_date
    Returns: total_reward, trajectory_length
    """
    seq_len = 60
    N = 5  # reward horizon
    episode_len = len(stock_df) - seq_len - N
    
    traj_log_probs = []
    traj_rewards = []
    traj_values = []
    position = 0.0
    position_history = []
    
    for t in range(seq_len, len(stock_df) - N):
        x = stock_df.iloc[t-seq_len:t][FEATURE_COLS].to_numpy()  # (60, 14)
        # Build state: market + position + days_held
        pos_signal = np.full((60, 1), position)
        days = min(t - last_change, 30) / 30.0  # normalize to [0, 1]
        days_signal = np.full((60, 1), days)
        state = np.concatenate([x, pos_signal, days_signal], axis=1)  # (60, 16)
        state = normalize(state)
        
        # Policy forward
        mu = model.forward(state)  # scalar
        dist = Normal(mu, sigma)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        new_position = torch.tanh(a).item()
        
        # Store
        traj_log_probs.append(log_prob)
        position_history.append((position, new_position))
        
        # Compute reward (retroactive, at t+N we compute reward for action at t)
        ret = stock_df.iloc[t+N].close / stock_df.iloc[t].close - 1
        vol = stock_df.iloc[t].atr_ratio
        trade_cost = abs(new_position - position) * 0.0005
        reward = position * ret / (vol + 1e-6) - trade_cost
        traj_rewards.append(reward)
        
        position = new_position
    
    # Compute discounted returns
    G = 0
    returns = []
    for r in reversed(traj_rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    
    # Baseline (running mean)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Policy gradient loss
    loss = -(torch.stack(traj_log_probs) * returns).sum()
    
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return traj_rewards  # for logging
```

### Multi-Stock Batching

Each episode = one stock over time. Train on batches of stocks:

```
for batch_stocks in stock_list(batch_size=8):
    for each stock in batch:
        run episode -> accumulate loss and gradients
        # need to track separate positions per stock
    optimizer.step()
```

Alternatively, process all stocks in parallel with vectorized operations (tensor of shape (B, T, 60, features)).

### Early Stopping

- Evaluate validation Sharpe every 10 epochs
- Stop if no improvement for 30 epochs
- Save best model

## Architecture

```
class PositionAgent(nn.Module):
    def __init__(self, n_features=14, d_model=128, ...):
        super().__init__()
        self.input_proj = nn.Linear(n_features + 2, d_model)  # +2 for position + days_held
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(...)
        self.norm = nn.LayerNorm(d_model)
        self.actor_head = nn.Linear(d_model, 1)  # mu
        self.log_sigma = nn.Parameter(torch.tensor(-1.2))  # ~0.3 initial
    
    def forward(self, x):
        # x: (B, 60, 16)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        mu = self.actor_head(x)  # (B, 1)
        return mu
    
    def get_dist(self, x):
        mu = self.forward(x)
        sigma = torch.exp(self.log_sigma)
        return Normal(mu, sigma)
```

## Evaluation Metrics

| Metric | How |
|--------|-----|
| Sharpe Ratio | `mean(reward) / std(reward) * sqrt(252)` |
| Cumulative Return | `prod(1 + position_t * ret_t) - 1` |
| Max Drawdown | Peak-to-trough of cumulative returns |
| Win Rate | Fraction of days with positive reward |
| Avg Position Duration | How long before position changes |
| Trade Frequency | Number of position changes / total days |

## Implementation Order

1. **State builder**: `get_state(stock_df, t, position, days_held)` → compiled numpy state (60, 16)
2. **PositionAgent model**: Transformer + Gaussian policy head
3. **Episode runner**: `run_episode(stock_df, model)` → trajectory of (log_probs, rewards)
4. **Normalization function**: per-sequence z-score (same as before)
5. **Dataset curation**: Filter rules applied to alpha360 data
6. **Training loop**: REINFORCE with batch-of-stocks
7. **Validation**: Evaluate Sharpe on holdout period (no filtering, full days)
8. **Integration**: CLI command `alpha-quat model nn rl_agent --name exp_agent_v1`
