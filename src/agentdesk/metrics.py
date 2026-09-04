"""Statistics for scoring a classifier on a finite sample.

A percentage without an interval invites a decision it cannot support. On a hundred cases, 87 %
and 91 % are the same number; treating the second as an improvement is how prompt engineering
turns into superstition.
"""

import math
from dataclasses import dataclass

# 95% two-sided.
Z = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Proportion:
    successes: int
    total: int

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def interval(self, z: float = Z) -> tuple[float, float]:
        """Wilson score interval.

        Not the textbook `p ± z·√(p(1-p)/n)`: that one is badly wrong exactly where classifier
        scores live — near 100 %, on small samples — where it happily returns bounds above 1.
        """
        n = self.total
        if n == 0:
            return (0.0, 0.0)

        p = self.value
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

        low, high = centre - margin, centre + margin

        # At the extremes the bounds are analytically exact — 1.0 for a perfect score, 0.0 for a
        # total miss — and the arithmetic above lands a float epsilon short. Pinning them keeps
        # the point estimate inside its own interval instead of a hair outside it.
        if self.successes == self.total:
            high = 1.0
        if self.successes == 0:
            low = 0.0

        return (max(0.0, low), min(1.0, high))

    def format(self) -> str:
        low, high = self.interval()
        return f"{self.value:.1%} [{low:.1%}, {high:.1%}] n={self.total}"


def is_significant(before: Proportion, after: Proportion) -> bool:
    """Whether two scores differ by more than sampling noise.

    Non-overlapping 95% intervals is a deliberately strict reading: it under-reports real
    improvements rather than over-reporting imaginary ones. On a sample this size, a change that
    cannot clear that bar is not a change worth shipping a prompt for.
    """
    return before.interval()[1] < after.interval()[0] or after.interval()[1] < before.interval()[0]


def mcnemar_p_value(only_first: int, only_second: int) -> float:
    """Exact McNemar test for two methods measured on the *same* items.

    Comparing two independent confidence intervals throws away the pairing, and on a small
    sample that costs most of the power: two engines evaluated on the same fifty questions agree
    on most of them, and only the disagreements carry information.

    `only_first` is the number of items the first method got right and the second did not;
    `only_second` the reverse. Items both got right, or both got wrong, say nothing about which
    is better and are correctly ignored.

    Returns the two-sided probability of seeing a split this lopsided if the two methods were
    equally good. Exact binomial rather than the chi-squared approximation, which is unreliable
    exactly where these counts live — below about 25 discordant pairs.
    """
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0

    extreme = min(only_first, only_second)
    tail: float = sum(math.comb(discordant, k) for k in range(extreme + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def paired_verdict(only_first: int, only_second: int, alpha: float = 0.05) -> str:
    p_value = mcnemar_p_value(only_first, only_second)
    better = "second" if only_second > only_first else "first"
    if p_value >= alpha:
        return f"not distinguishable (p={p_value:.3f}, {only_first} vs {only_second} wins)"
    return f"{better} is better (p={p_value:.3f}, {only_first} vs {only_second} wins)"
