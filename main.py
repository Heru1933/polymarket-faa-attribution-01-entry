"""polymarket-faa-attribution-01 entry: quantile forecasts for Yes prices.

Method, per market, from the case's own 72 hours of minute prices:

- For each horizon h (1..240 minutes) take the empirical distribution
  of h-minute price changes inside the window. Its quantiles carry
  the market's real behaviour: long flat stretches, small steps, and
  the occasional jump, none of which a fixed Gaussian spread knows.
- Blend that with the house persistence spread (sigma 0.0025 per
  square-root minute): for the outer quantiles keep whichever of the
  two is wider, so a market that slept through the whole window is
  still not forecast as certain to sleep through the next four hours.
- The median leans half-way toward the empirical median change:
  drift in these markets is real but noisy, so it is shrunk, not
  trusted whole.

Rows are clamped to [0, 1], rounded to 6 decimals and sorted, as the
scorer requires. Offline, standard library only, deterministic.
"""

import json
import math
import os
import sys
from pathlib import Path

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)
BASE_SIGMA = 0.0025
MEDIAN_SHRINK = 0.5
MIN_SAMPLES = 60


def quantile(sorted_vals, tau):
    """Linear-interpolated quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def changes_at(prices, h):
    """Sorted h-minute changes, sub-sampled so the work stays bounded
    (stride grows with h; the quantile estimates barely notice)."""
    n = len(prices)
    stride = max(1, h // 8)
    vals = [prices[i + h] - prices[i] for i in range(0, n - h, stride)]
    vals.sort()
    return vals


def price6(x):
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def forecast_market(prices, horizon):
    x = prices[-1]
    rows = []
    cache = {}
    for h in range(1, horizon + 1):
        if h not in cache:
            cache[h] = changes_at(prices, h)
        ch = cache[h]
        base = [x + z * BASE_SIGMA * math.sqrt(h) for z in Z]
        if len(ch) >= MIN_SAMPLES:
            emp = [quantile(ch, t) for t in TAUS]
            med = x + MEDIAN_SHRINK * emp[2]
            row = [
                min(x + emp[0], base[0]),
                min(x + emp[1], base[1]),
                med,
                max(x + emp[3], base[3]),
                max(x + emp[4], base[4]),
            ]
        else:
            row = base
        row = sorted(price6(v) for v in row)
        rows.append(row)
    return rows


def run(input_dir, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(input_dir).glob("*.in"))
    for f in files:
        case = json.loads(f.read_text())
        forecasts = []
        for m in case["markets"]:
            q = forecast_market(m["prices"], int(case["horizon_minutes"]))
            forecasts.append({"id": m["id"], "q": q})
        target = out / (f.stem + ".out")
        target.write_text(json.dumps({"forecasts": forecasts}) + "\n")


def main():
    run(
        os.environ.get("MAKEFUN_INPUT_DIR", "/input"),
        os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
