"""Factor registry with dependency resolution."""

import re
from collections import deque

from alpha_quat.features.factor import Factor


class FactorRegistry:
    def __init__(self, name: str) -> None:
        self.name = name
        self.factors: dict[str, Factor] = {}

    def register(self, factor: Factor):
        self.factors[factor.name] = factor

    def topological_order(self) -> list[Factor]:
        in_degree: dict[str, int] = {}
        adj: dict[str, list[str]] = {name: [] for name in self.factors}

        for name, factor in self.factors.items():
            deps = [d for d in factor.depends_on if d in self.factors]
            in_degree[name] = len(deps)
            for dep in deps:
                adj[dep].append(name)

        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        result: list[Factor] = []

        while queue:
            name = queue.popleft()
            result.append(self.factors[name])
            for neighbor in adj[name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.factors):
            raise ValueError(
                f"cycle detected in factor dependencies for registry '{self.name}'"
            )
        return result

    def min_lookback(self) -> int:
        max_n = 0
        for factor in self.factors.values():
            for n in re.findall(
                r"(?:REF|MEAN|STD|SUM|MAX|MIN|DELTA)\(\$?\w+,\s*(\d+)\)",
                factor.expression,
            ):
                max_n = max(max_n, int(n))
            for m in re.findall(
                r"CORR\(\$?\w+,\s*\$?\w+,\s*(\d+)\)", factor.expression
            ):
                max_n = max(max_n, int(m))
        return max_n
