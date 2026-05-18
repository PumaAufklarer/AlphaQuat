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


def _find_matching_paren(text: str, open_idx: int) -> int:
    """Return index of closing paren matching the opening paren at open_idx."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _unwrap_rank(expression: str) -> str | None:
    """If expression is RANK(...), return the inner expression. Otherwise None."""
    if not expression.startswith("RANK("):
        return None
    end = _find_matching_paren(expression, 4)  # '(' is at index 4
    if end == -1:
        return None
    return expression[5:end]  # content between RANK( and )


def _unwrap_quantile(expression: str) -> tuple[str, str] | None:
    """If expression is QUANTILE(...), return (inner_expr, N). Otherwise None."""
    if not expression.startswith("QUANTILE("):
        return None
    end = _find_matching_paren(expression, 8)  # '(' is at index 8
    if end == -1:
        return None
    content = expression[9:end]  # content between QUANTILE( and )
    # Find the last comma at depth 0 (splits inner_expr, N)
    depth = 0
    for i in range(len(content) - 1, -1, -1):
        if content[i] == ")":
            depth += 1
        elif content[i] == "(":
            depth -= 1
        elif content[i] == "," and depth == 0:
            return content[:i].strip(), content[i + 1 :].strip()
    return None


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
