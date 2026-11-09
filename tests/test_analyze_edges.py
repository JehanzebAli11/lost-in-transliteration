"""Rates of exactly 0 or 1 (common with n=10 MCP cells and 0-FPR classifiers) must plot.

Float rounding put the Wilson bound a hair on the wrong side of the rate
(lo=8.7e-19 at p=0, hi=0.9999999999999999 at p=1), so matplotlib refused the
negative error-bar length and fig2/fig8 were skipped on the real results.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src import analyze
from src.metrics import wilson_ci


@pytest.mark.parametrize("n", [1, 7, 10, 50, 120, 300, 395, 399, 400])
def test_wilson_bounds_bracket_the_rate(n):
    for k in range(n + 1):
        lo, hi = wilson_ci(k, n)
        p = k / n
        assert 0.0 <= lo <= p <= hi <= 1.0, (k, n, lo, hi)
    assert wilson_ci(0, n)[0] == 0.0
    assert wilson_ci(n, n)[1] == 1.0


def test_grouped_bars_tolerates_bounds_past_the_rate():
    # values as read back from an older CSV (results/mcp_summary.csv) with rounding residue
    vals = {("a", "en"): 0.0, ("a", "ur"): 1.0}
    errs = {("a", "en"): (8.673617379884035e-19, 0.2775), ("a", "ur"): (0.7225, 0.9999999999999999)}
    fig, ax = plt.subplots()
    analyze.grouped_bars(ax, ["a"], ["en", "ur"], vals, errs)
    plt.close(fig)
