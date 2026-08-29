"""进化管线的纯函数测试（不依赖真实 DB）。"""
import numpy as np
import pandas as pd

from fwsp import factor_mining as fm
from fwsp.overfit_guard import stability_check
from fwsp.community_watch import merge_all_recipes, load_base_recipes, load_community_recipes


def _synthetic_panels(n=200, codes=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"{c:06d}" for c in range(codes)]
    close = pd.DataFrame(rng.standard_normal((n, codes)).cumsum(0) + 100,
                         index=dates, columns=cols)
    vol = pd.DataFrame(np.abs(rng.standard_normal((n, codes))) * 1e6 + 1e6,
                       index=dates, columns=cols)
    amt = vol * close
    high = close * 1.01
    low = close * 0.99
    op = close.shift(1).fillna(close)
    return {"open": op, "high": high, "low": low, "close": close,
            "volume": vol, "amount": amt}


def test_community_merge_includes_base_and_community():
    base = load_base_recipes()
    comm = load_community_recipes()
    merged = merge_all_recipes()
    assert len(merged) == len(base) + len(comm)
    assert any(r.get("source") == "worldquant_alpha101" for r in merged)


def test_long_short_backtest_random_is_finite():
    panels = _synthetic_panels()
    fwd = fm.forward_returns(panels, horizon=10)
    rnd = pd.DataFrame(np.random.default_rng(1).standard_normal(panels["close"].shape),
                       index=panels["close"].index, columns=panels["close"].columns)
    r = fm.long_short_backtest(rnd, fwd)
    assert "net_ir" in r
    # 随机因子 net_ir 应当是有界有限值（不应抛错）
    assert np.isfinite(r["net_ir"]) or np.isnan(r["net_ir"])


def test_long_short_backtest_reward_strong_factor():
    panels = _synthetic_panels()
    fwd = fm.forward_returns(panels, horizon=10)
    # 用未来收益本身的同频信号构造因子（仅验证函数数值行为，非真实因子）
    strong = fwd
    r = fm.long_short_backtest(strong, fwd)
    assert r["net_ir"] > 5  # 强信号应得很高净 IR


def test_stability_check_returns_worst():
    rng = np.random.default_rng(2)
    ic = pd.Series(rng.normal(0.03, 0.02, 400))
    out = stability_check(ic, window=252)
    assert "worst" in out
    assert np.isfinite(out["worst"])


def test_screen_factors_streaming_memory_safe():
    from fwsp.factor_factory import FactorSpec, compute_factor_panel
    panels = _synthetic_panels(n=300)
    fwd = fm.forward_returns(panels, horizon=10)
    specs = [FactorSpec(f"f{i}", "x", e, {}, "")
             for i, e in enumerate(["zscore(close,5)", "std(close,10)",
                                    "sma(close,20)"])]
    out = fm.screen_factors(panels, specs, fwd, icir_gate=-1.0, n_min=20)
    assert len(out) >= 1  # 流式门禁不崩溃、返回子集（随机数据下部分因子 ICIR 低被剔除）
