import math

import pandas as pd
import pytest

from fwsp.indicators import compute_features, macd, rsi, sma


def _series(values):
    return pd.Series(values, dtype="float64")


class TestSma:
    def test_basic(self):
        s = _series([1, 2, 3, 4, 5])
        out = sma(s, 5)
        assert math.isclose(out.iloc[-1], 3.0)

    def test_insufficient_periods_is_nan(self):
        s = _series([1, 2, 3])
        out = sma(s, 5)
        assert math.isnan(out.iloc[-1])


class TestRsi:
    def test_all_gains_gives_100(self):
        s = _series([float(i) for i in range(1, 30)])
        v = rsi(s).iloc[-1]
        assert v == 100.0

    def test_range_bound(self):
        import random

        random.seed(42)
        vals = [100.0]
        for _ in range(200):
            vals.append(vals[-1] + random.uniform(-3, 3))
        v = rsi(_series(vals)).iloc[-1]
        assert 0.0 <= v <= 100.0


class TestMacd:
    def test_trending_up_dif_positive(self):
        s = _series([float(i) for i in range(1, 60)])
        dif, dea, _ = macd(s)
        assert dif.iloc[-1] > 0
        assert dea.iloc[-1] > 0


def _make_df(n=140, base=10.0):
    rows = []
    price = base
    for i in range(n):
        o = price
        c = price * (1 + (0.001 if i % 7 else -0.002))
        h = max(o, c) * 1.01
        low = min(o, c) * 0.99
        rows.append((f"2026-01-{(i % 28) + 1:02d}" if i < 28
                     else f"2026-02-{((i - 28) % 28) + 1:02d}",
                     o, h, low, c, 1_000_000 + i * 1000, 1e7 + i * 1e4))
        price = c
    return pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                       "close", "volume", "amount"])


class TestComputeFeatures:
    def test_returns_features_for_sufficient_data(self):
        ft = compute_features(_make_df(140))
        assert ft is not None
        for key in ("close", "ma20", "ma60", "rsi", "dif", "dea",
                    "amount_ma20", "ret_20d"):
            assert key in ft

    def test_insufficient_data_returns_none(self):
        assert compute_features(_make_df(50)) is None
        assert compute_features(None) is None

    def test_feature_values_finite(self):
        ft = compute_features(_make_df(140))
        for key in ("ma5", "ma20", "ma60", "rsi"):
            v = ft[key]
            assert v is None or not math.isnan(v)

    def test_chg_pct_sign_matches_direction(self):
        df = _make_df(140)
        ft = compute_features(df)
        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        expected = (last_close - prev_close) / prev_close * 100
        assert math.isclose(ft["chg_pct"], expected, rel_tol=1e-9)
