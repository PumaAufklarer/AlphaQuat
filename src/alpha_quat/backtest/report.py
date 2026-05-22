import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def generate_html_report(result, config, output_path):
    snapshots = result["snapshots"]
    trades = result["trades"]
    metrics = result["metrics"]

    dates = [s["date"] for s in snapshots]
    nav = [s["total_value"] for s in snapshots]
    invested_line = [metrics["total_invested"]] * len(nav)

    equity_img = _make_equity_chart(dates, nav, invested_line)
    dd_img = _make_drawdown_chart(nav, dates)
    trade_html = _build_trade_table(trades)
    config_html = _build_config_table(config)

    cum_ret = metrics["cumulative_return"] * 100
    ann_ret = metrics["annualized_return"] * 100
    max_dd = metrics["max_drawdown"] * 100
    ret_color = "green" if cum_ret >= 0 else "red"

    html = _HTML_TEMPLATE.format(
        cum_ret=cum_ret,
        ann_ret=ann_ret,
        max_dd=max_dd,
        sharpe=metrics["sharpe_ratio"],
        win_rate=metrics["win_rate"] * 100,
        total_trades=metrics["total_trades"],
        final_value=metrics["final_value"],
        ret_color=ret_color,
        equity_img=equity_img,
        dd_img=dd_img,
        trade_html=trade_html,
        config_html=config_html,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _make_equity_chart(dates, nav, invested_line):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, nav, label="Portfolio NAV", color="#2c7fb8", linewidth=1.2)
    ax.plot(
        dates,
        invested_line,
        label="Total Invested",
        color="#999999",
        linestyle="--",
        linewidth=1.0,
    )
    above = [n >= i for n, i in zip(nav, invested_line)]
    below = [n < i for n, i in zip(nav, invested_line)]
    if any(above):
        ax.fill_between(
            range(len(dates)),
            nav,
            invested_line,
            where=above,
            color="#2c7fb8",
            alpha=0.1,
        )
    if any(below):
        ax.fill_between(
            range(len(dates)),
            nav,
            invested_line,
            where=below,
            color="#d7191c",
            alpha=0.1,
        )
    ax.set_title("Equity Curve", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 10000:.1f}w")
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(
        [dates[i] for i in range(0, len(dates), step)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    fig.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


def _make_drawdown_chart(nav, dates):
    peak = nav[0]
    dd_vals = []
    for v in nav:
        if v > peak:
            peak = v
        dd_vals.append((v - peak) / peak * 100 if peak > 0 else 0)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(dates)), dd_vals, 0, color="#d7191c", alpha=0.3)
    ax.plot(dd_vals, color="#d7191c", linewidth=1)
    ax.set_title("Drawdown (%)", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(True, alpha=0.3)
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels(
        [dates[i] for i in range(0, len(dates), step)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    fig.tight_layout()
    result = _fig_to_b64(fig)
    plt.close(fig)
    return result


def _build_trade_table(trades):
    rows = []
    for t in trades:
        pnl_str = f"{t['pnl']:+.2f}"
        color = "green" if t["pnl"] > 0 else ("red" if t["pnl"] < 0 else "gray")
        rows.append(
            f"<tr><td>{t['date']}</td><td>{t['ts_code']}</td>"
            f"<td>{t['action']}</td><td>{t['price']:.2f}</td>"
            f"<td>{t['shares']}</td><td>{t['commission']:.2f}</td>"
            f'<td style="color:{color}">{pnl_str}</td></tr>'
        )
    return "".join(rows)


def _build_config_table(config):
    return (
        f"<tr><td>Period</td><td>{config.start_date} ~ {config.end_date}</td></tr>"
        f"<tr><td>Initial Capital</td><td>{config.initial_capital:,.0f}</td></tr>"
        f"<tr><td>Monthly Addition</td><td>{config.monthly_addition:,.0f}</td></tr>"
        f"<tr><td>Commission</td><td>{config.commission_rate * 10000:.1f} bp</td></tr>"
        f"<tr><td>Stop Loss</td><td>{config.stop_loss_pct * 100:.0f}%</td></tr>"
        f"<tr><td>MA Factors</td><td>{config.short_factor} / {config.long_factor}</td></tr>"
        f"<tr><td>Max Holdings</td><td>{config.top_k}</td></tr>"
    )


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f5f5f5; color: #333; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 20px; color: #1a1a2e; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;
            margin-bottom: 24px; }}
  .card {{ background: white; border-radius: 8px; padding: 16px 20px;
           min-width: 130px; text-align: center;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
  .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 16px;
                      margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                      text-align: center; }}
  .chart-container img {{ max-width: 100%; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  tr:hover {{ background: #f8f9fa; }}
  h2 {{ font-size: 16px; margin: 24px 0 12px; color: #1a1a2e; }}
  .section {{ background: white; border-radius: 8px; padding: 16px 20px;
              margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>Backtest Report</h1>

<div class="cards">
  <div class="card"><div class="label">Cumulative Return</div>
    <div class="value" style="color:{ret_color}">{cum_ret:.2f}%</div></div>
  <div class="card"><div class="label">Annualized Return</div>
    <div class="value" style="color:{ret_color}">{ann_ret:.2f}%</div></div>
  <div class="card"><div class="label">Sharpe Ratio</div>
    <div class="value">{sharpe:.2f}</div></div>
  <div class="card"><div class="label">Max Drawdown</div>
    <div class="value" style="color:red">{max_dd:.2f}%</div></div>
  <div class="card"><div class="label">Win Rate</div>
    <div class="value">{win_rate:.1f}%</div></div>
  <div class="card"><div class="label">Total Trades</div>
    <div class="value">{total_trades}</div></div>
  <div class="card"><div class="label">Final Value</div>
    <div class="value">{final_value:,.0f}</div></div>
</div>

<div class="chart-container"><h2>Equity Curve</h2>
  <img src="data:image/png;base64,{equity_img}" alt="Equity Curve"></div>

<div class="chart-container"><h2>Drawdown</h2>
  <img src="data:image/png;base64,{dd_img}" alt="Drawdown"></div>

<div class="section"><h2>Trade Log</h2><table>
  <thead><tr><th>Date</th><th>Code</th><th>Action</th><th>Price</th>
    <th>Shares</th><th>Commission</th><th>P&amp;L</th></tr></thead>
  <tbody>{trade_html}</tbody></table></div>

<div class="section"><h2>Configuration</h2><table>
  {config_html}</table></div>
</body></html>"""
