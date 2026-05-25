"""Fundamental factors from daily_basic data.

Includes: valuation (PE/PB), profitability proxy (ROE), size (MV),
liquidity (turnover rate), and volume ratio.
"""

from alpha_quat.features.factor import Factor
from alpha_quat.features.registry import FactorRegistry

_FACTORS = [
    Factor(
        name="PE_TTM",
        expression="($pe_ttm - MIN($pe_ttm, 750)) / (MAX($pe_ttm, 750) - MIN($pe_ttm, 750))",
        category="fundamental",
    ),
    Factor(
        name="PB",
        expression="($pb - MIN($pb, 750)) / (MAX($pb, 750) - MIN($pb, 750))",
        category="fundamental",
    ),
    Factor(
        name="ROE_RAW",
        expression="$pb / NULLIF($pe_ttm, 0)",
        category="fundamental",
    ),
    Factor(
        name="ROE",
        expression="(ROE_RAW - MIN(ROE_RAW, 750)) / (MAX(ROE_RAW, 750) - MIN(ROE_RAW, 750))",
        depends_on=["ROE_RAW"],
        category="fundamental",
    ),
    Factor(
        name="MV",
        expression="($total_mv - MIN($total_mv, 750)) / (MAX($total_mv, 750) - MIN($total_mv, 750))",
        category="fundamental",
    ),
    Factor(
        name="TURN",
        expression="$turnover_rate",
        category="fundamental",
    ),
    Factor(
        name="VOLRATIO",
        expression="$volume_ratio",
        category="fundamental",
    ),
]


def build_alpha_fund() -> FactorRegistry:
    reg = FactorRegistry(name="alpha_fund")
    for f in _FACTORS:
        reg.register(f)
    return reg
