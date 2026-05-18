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

    # Step 1: Replace $raw field references first (so they're available to all operators)
    expr = expr.replace("$vwap", "amount / NULLIF(volume, 0)")
    expr = expr.replace("$open", "open")
    expr = expr.replace("$high", "high")
    expr = expr.replace("$low", "low")
    expr = expr.replace("$close", "close")
    expr = expr.replace("$volume", "volume")
    expr = expr.replace("$amount", "amount")

    # Step 2: RANK/QUANTILE — outermost operators, may contain REF/MEAN/STD/etc.
    # Must run BEFORE window function substitutions so inner expressions stay clean.
    # Greedy .+ backtracks to match the correct closing paren for nested parens.
    expr = re.sub(
        r"RANK\((.+)\)",
        r"RANK() OVER (PARTITION BY trade_date ORDER BY \1)",
        expr,
    )
    expr = re.sub(
        r"QUANTILE\((.+),\s*(\d+)\)",
        r"NTILE(\2) OVER (PARTITION BY trade_date ORDER BY \1)",
        expr,
    )

    # Step 3: Time-series operators (REF, MEAN, STD, SUM, MAX, MIN, CORR, DELTA)
    expr = re.sub(
        r"REF\((\w+),\s*(\d+)\)",
        r"LAG(\1, \2) OVER w_time",
        expr,
    )
    expr = re.sub(
        r"MEAN\((\w+),\s*(\d+)\)",
        lambda m: (
            f"AVG({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    expr = re.sub(
        r"STD\((\w+),\s*(\d+)\)",
        lambda m: (
            f"STDDEV_SAMP({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    expr = re.sub(
        r"SUM\((\w+),\s*(\d+)\)",
        lambda m: (
            f"SUM({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    expr = re.sub(
        r"MAX\((\w+),\s*(\d+)\)",
        lambda m: (
            f"MAX({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    expr = re.sub(
        r"MIN\((\w+),\s*(\d+)\)",
        lambda m: (
            f"MIN({m.group(1)}) OVER (w_time ROWS BETWEEN {int(m.group(2)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    expr = re.sub(
        r"CORR\((\w+),\s*(\w+),\s*(\d+)\)",
        lambda m: (
            f"CORR({m.group(1)}, {m.group(2)}) OVER (w_time ROWS BETWEEN {int(m.group(3)) - 1} PRECEDING AND CURRENT ROW)"
        ),
        expr,
    )
    expr = re.sub(
        r"DELTA\((\w+),\s*(\d+)\)",
        r"\1 - LAG(\1, \2) OVER w_time",
        expr,
    )

    return expr
