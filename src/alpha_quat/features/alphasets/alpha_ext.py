"""Extended price-pattern factors beyond core Alpha158.

Includes: channel position, skewness proxy, overnight gap, intraday patterns.
"""

from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry

_FACTORS = [
    # === Channel position — only 30d window is effective ===
    Factor(
        name="CHP30",
        expression="($close - MIN($low, 30)) / (MAX($high, 30) - MIN($low, 30))",
        category="price",
    ),
    # === SKEWP — mean position in price channel, 10/20d windows only ===
    Factor(
        name="SKEWP10",
        expression="(MEAN($close, 10) - MIN($close, 10)) / (MAX($close, 10) - MIN($close, 10))",
        category="price",
    ),
    Factor(
        name="SKEWP20",
        expression="(MEAN($close, 20) - MIN($close, 20)) / (MAX($close, 20) - MIN($close, 20))",
        category="price",
    ),
    # === GAP — overnight gap ===
    Factor(
        name="GAP",
        expression="$open / REF($close, 1) - 1",
        category="price",
    ),
    # === DRP — daily relative position in intraday range ===
    Factor(
        name="DRP",
        expression="($close - $low) / ($high - $low)",
        category="price",
    ),
    # === O2C — open-to-close intraday return ===
    Factor(
        name="O2C",
        expression="$close / $open - 1",
        category="price",
    ),
    # === HLC — selling pressure (drop from day's high) ===
    Factor(
        name="HLC",
        expression="($high - $close) / ($high - $low)",
        category="price",
    ),
    # === EMA — exponential moving average (approximated via decay-weighted sum) ===
    Factor(name="EMA12C", expression="EMA($close, 12)", category="price"),
    Factor(name="EMA26C", expression="EMA($close, 26)", category="price"),
    Factor(
        name="MACD",
        expression="EMA12C - EMA26C",
        depends_on=["EMA12C", "EMA26C"],
        category="price",
    ),
    # === RSI — relative strength index ===
    Factor(name="RSI14", expression="RSI($close, 14)", category="price"),
    # === REG_SLOPE — linear regression slope (trend strength) ===
    Factor(name="SLOPE5", expression="REG_SLOPE($close, 5)", category="price"),
    Factor(name="SLOPE20", expression="REG_SLOPE($close, 20)", category="price"),
]


def build_alpha_ext() -> FactorRegistry:
    reg = FactorRegistry(name="alpha_ext")
    for f in _FACTORS:
        reg.register(f)
    return reg
