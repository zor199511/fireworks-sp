import numpy as np
import pandas as pd
import pytest

from fwsp.factor_mining import (forward_returns, greedy_select, ic_analysis,
                                oos_guard)


def _panel_with_fwd(T=700, C=50, seed=0, horizon=10):
    """构造含明确前向收益的面板：用随机游走生成 open，并计算 fwd。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    codes = [f"{i:06d}" for i in range(C)]
    idx = pd.Index(dates, name="date")
    cols = pd.Index(codes, name="code")
    base = rng.normal(100, 1, (T, C)).cumsum(axis=0) + 100
    op = pd.DataFrame(base, index=idx, columns=cols)
    panels = {
        "open": op, "high": op + 1, "low": op - 1, "close": op,
        "volume": pd.DataFrame(rng.normal(1e6, 1e5, (T, C)), index=idx, columns=cols),
        "amount": pd.DataFrame(rng.normal(1e8, 1e6, (T, C)), index=idx, columns=cols),
    }
    fwd = forward_returns(panels, horizon=horizon)
    return panels, fwd


class TestForwardReturns:
    def test_no_lookahead(self):
        panels, fwd = _panel_with_fwd(T=200, C=20, seed=3, horizon=10)
        op = panels["open"]
        # 扰动未来 open（k 远大于 i+horizon）不应改变 fwd[i]
        k = 199
        i = 0
        fwd0 = fwd.iloc[i].copy()
        op2 = op.copy()
        op2.iloc[k, 0] = op2.iloc[k, 0] + 1e9
        fwd2 = forward_returns({**panels, "open": op2}, horizon=10)
        assert np.allclose(fwd0.to_numpy(), fwd2.iloc[i].to_numpy(),
                           equal_nan=True), "fwd 泄漏了未来 open"

    def test_uses_tplus1_open_not_t_open(self):
        panels, fwd = _panel_with_fwd(T=200, C=20, seed=4, horizon=10)
        op = panels["open"]
        i = 50
        fwd_i = fwd.iloc[i].copy()
        # 扰动 open[i] 不应影响 fwd[i]（买入价是 open[i+1]）
        op2 = op.copy()
        op2.iloc[i, 0] = op2.iloc[i, 0] + 1e9
        fwd2 = forward_returns({**panels, "open": op2}, horizon=10)
        assert np.allclose(fwd_i.to_numpy(), fwd2.iloc[i].to_numpy(),
                           equal_nan=True), "fwd 误用了 T 日开盘价"

    def test_uses_tplus_horizon_open(self):
        panels, fwd = _panel_with_fwd(T=200, C=20, seed=5, horizon=10)
        op = panels["open"]
        i = 50
        fwd_i = fwd.iloc[i].copy()
        # 扰动 open[i+horizon] 应改变 fwd[i]（卖出价）
        op2 = op.copy()
        op2.iloc[i + 10, 0] = op2.iloc[i + 10, 0] + 1e9
        fwd2 = forward_returns({**panels, "open": op2}, horizon=10)
        assert not np.allclose(fwd_i.to_numpy(), fwd2.iloc[i].to_numpy(),
                               equal_nan=True), "fwd 未使用 T+horizon 开盘价"


def _signal_matrix(T=700, C=50, seed=0, scale=1.0):
    """构造强截面信号矩阵（每日期跨股票独立正态），用于 IC 相关测试。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    codes = [f"{i:06d}" for i in range(C)]
    idx = pd.Index(dates, name="date")
    cols = pd.Index(codes, name="code")
    return pd.DataFrame(rng.normal(0, scale, (T, C)), index=idx, columns=cols)


class TestIcAnalysis:
    def test_monotone_factor_positive_icir(self):
        fwd = _signal_matrix(T=700, C=50, seed=6)
        rng = np.random.default_rng(99)
        noise = pd.DataFrame(rng.normal(0, 0.01, fwd.shape),
                             index=fwd.index, columns=fwd.columns)
        factor = fwd + noise
        res = ic_analysis(factor, fwd)
        assert res["n"] > 50
        assert res["icir"] > 1.0, f"单调因子应得显著正 ICIR，实际 {res['icir']:.3f}"
        assert res["pos_ratio"] > 0.8

    def test_negative_monotone_factor(self):
        fwd = _signal_matrix(T=700, C=50, seed=7)
        rng = np.random.default_rng(98)
        noise = pd.DataFrame(rng.normal(0, 0.01, fwd.shape),
                             index=fwd.index, columns=fwd.columns)
        factor = -fwd + noise
        res = ic_analysis(factor, fwd)
        assert res["icir"] < 0, "反向因子应得负 ICIR"


class TestGreedySelect:
    def test_returns_lte_top_and_dedup_collinear(self):
        rng = np.random.default_rng(123)
        s1 = _signal_matrix(T=700, C=50, seed=8)
        s4 = _signal_matrix(T=700, C=50, seed=21)
        fwd = (s1 + s4) / 2.0  # 真实前向收益由两路信号构成
        nz = lambda: pd.DataFrame(rng.normal(0, 0.01, fwd.shape),
                                  index=fwd.index, columns=fwd.columns)
        f1 = s1 + nz()                       # 好因子（含 s1）
        f2 = s1 + nz() * 0.5                 # 与 f1 共线
        f3 = -s1 + nz()                      # 负 ICIR，应被 gate 淘汰
        f4 = s4 + nz()                       # 独立好因子（含 s4）

        factors = {"f1": f1, "f2": f2, "f3": f3, "f4": f4}
        selected = greedy_select(factors, fwd, icir_gate=0.5, corr_thr=0.7, top=20)
        assert len(selected) <= 20
        assert "f1" in selected, "好因子 f1 应入选"
        assert "f4" in selected, "独立好因子 f4 应入选"
        assert "f2" not in selected, "共线因子 f2 应被剔除"
        assert "f3" not in selected, "负 ICIR 因子 f3 应被 gate 淘汰"


class TestOosGuard:
    def test_promote_on_lift(self):
        ok, msg = oos_guard(0.10, 0.05, min_lift=0.02)
        assert ok and "提升" in msg

    def test_reject_without_lift(self):
        ok, msg = oos_guard(0.06, 0.05, min_lift=0.02)
        assert not ok

    def test_first_run_promotes(self):
        ok, msg = oos_guard(0.08, None)
        assert ok and "首次" in msg
