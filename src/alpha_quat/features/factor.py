"""Factor definition and expression compiler."""

import re
from dataclasses import dataclass, field


@dataclass
class Factor:
    name: str
    expression: str
    category: str = ""
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.depends_on:
            self.depends_on = self._parse_deps()

    def _parse_deps(self) -> list[str]:
        refs = set(re.findall(r"\$\w+", self.expression))
        factor_refs = set(re.findall(r"\b(f_\d{3})\b", self.expression))
        return sorted(refs | factor_refs)


def compile(expression: str) -> str:
    """Compile DSL expression to DuckDB SQL via regex substitution."""
    expr = expression

    # REF: LAG with offset
    expr = re.sub(
        r"REF\(\$?(\w+),\s*(\d+)\)",
        r"LAG(\1, \2) OVER w_time",
        expr,
    )
    # MEAN: AVG with frame
    expr = re.sub(
        r"MEAN\(\$?(\w+),\s*(\d+)\)",
        lambda m: (
            f"AVG({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    # STD: STDDEV_SAMP with frame
    expr = re.sub(
        r"STD\(\$?(\w+),\s*(\d+)\)",
        lambda m: (
            f"STDDEV_SAMP({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    # SUM: SUM with frame
    expr = re.sub(
        r"SUM\(\$?(\w+),\s*(\d+)\)",
        lambda m: (
            f"SUM({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    # MAX: MAX with frame
    expr = re.sub(
        r"MAX\(\$?(\w+),\s*(\d+)\)",
        lambda m: (
            f"MAX({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    # MIN: MIN with frame
    expr = re.sub(
        r"MIN\(\$?(\w+),\s*(\d+)\)",
        lambda m: (
            f"MIN({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    # CORR: CORR with frame (two args)
    expr = re.sub(
        r"CORR\(\$?(\w+),\s*\$?(\w+),\s*(\d+)\)",
        lambda m: (
            f"CORR({m.group(1)}, {m.group(2)}) OVER (w_time ROWS BETWEEN {int(m.group(3)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    # DELTA: feature - LAG(feature, N)
    expr = re.sub(
        r"DELTA\(\$?(\w+),\s*(\d+)\)",
        r"\1 - LAG(\1, \2) OVER w_time",
        expr,
    )
    # RANK: cross-sectional rank
    expr = re.sub(
        r"RANK\((\w+)\)",
        r"RANK() OVER (PARTITION BY trade_date ORDER BY \1)",
        expr,
    )
    # QUANTILE: cross-sectional ntile
    expr = re.sub(
        r"QUANTILE\((\w+),\s*(\d+)\)",
        r"NTILE(\2) OVER (PARTITION BY trade_date ORDER BY \1)",
        expr,
    )
    # $vwap: computed from amount/volume
    expr = expr.replace("$vwap", "amount / NULLIF(volume, 0)")
    # $raw fields
    expr = expr.replace("$open", "open")
    expr = expr.replace("$high", "high")
    expr = expr.replace("$low", "low")
    expr = expr.replace("$close", "close")
    expr = expr.replace("$volume", "volume")
    expr = expr.replace("$amount", "amount")

    return expr
