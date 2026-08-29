"""P0-2 标签/成本口径统一：open→open 标签一致 + 成本常量单一来源。"""
import numpy as np
import pandas as pd

from fwsp import costs
from fwsp import factors as F
from fwsp.factor_mining import forward_returns as fm_forward_returns


def _panels(n=120, codes=20, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"{c:06d}" for c in range(codes)]
    close = pd.DataFrame(rng.standard_normal((n, codes)).cumsum(0) + 100,
                         index=dates, columns=cols)
    op = close * (1 + rng.standard_normal((n, codes)) * 0.002) + 0.5
    low = close * 0.99
    high = close * 1.01
    vol = pd.DataFrame(np.abs(rng.standard_normal((n, codes))) * 1e6 + 1e6,
                       index=dates, columns=cols)
    amt = vol * close
    out = {}
    for name, df in (("close", close), ("open", op), ("low", low),
                     ("high", high), ("volume", vol), ("amount", amt)):
        out[name] = df
    return out


def test_forward_returns_label_matches_promotion():
    panels = _panels()
    # factors.forward_returns 现在应等于 factor_mining.forward_returns（open→open）
    fwd_factors = F.forward_returns(panels)
    for h in F.HORIZONS:
        a = fwd_factors[h]
        b = fm_forward_returns(panels, h)
        assert a.shape == b.shape
        # NaN 位置一致，且数值一致
        assert a.isna().equals(b.isna())
        fill_a = a.fillna(0.0)
        fill_b = b.fillna(0.0)
        assert np.allclose(fill_a.values, fill_b.values, atol=1e-9)


def test_forward_returns_not_close_to_close():
    panels = _panels()
    open_label = F.forward_returns(panels)[10]
    close_label = panels["close"].pct_change(10).shift(-10)
    diff = (open_label.fillna(0) - close_label.fillna(0)).abs()
    assert (diff > 1e-6).to_numpy().any()


def test_costs_single_source():
    assert abs(costs.COST_BUY - (0.00025 + 0.001)) < 1e-12
    assert abs(costs.COST_SELL - (0.00025 + 0.0005 + 0.001)) < 1e-12
    from fwsp import backtest, multifactor
    assert backtest.COST_BUY is costs.COST_BUY
    assert backtest.COST_SELL is costs.COST_SELL
    assert multifactor.COST_BUY is costs.COST_BUY
    assert multifactor.COST_SELL is costs.COST_SELL


def test_walk_forward_backtest_accepts_profit_pct():
    from fwsp import multifactor as M
    panels = _panels(n=200)
    z = {f"f{i}": pd.DataFrame(np.random.default_rng(i).standard_normal(panels["close"].shape),
                              index=panels["close"].index, columns=panels["close"].columns)
          for i in range(3)}
    ics = {k: pd.Series(np.random.default_rng(1).standard_normal(len(panels["close"].index)),
                       index=panels["close"].index) for k in z}
    for pp in (None, 8.0):
        r = M.walk_forward_backtest(z, ics, panels["close"], panels["open"],
                                   panels["low"], set(), start="2024-03-01",
                                   top_n=3, horizon=10, profit_pct=pp)
        assert "total_return" in r and np.isfinite(r["total_return"])
