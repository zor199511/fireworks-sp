"""P0-2 收尾：两套回测引擎的 成本/止盈/止损/滑点 统一到 execution 层。

锁定：
- decide_exit 的止损/止盈/到期语义（含修正前「low 被错标为 high → 止损永不触发」的坑）。
- position_return 同时扣双边成本。
- 晋升路径 long_short_backtest 成本单一来源自 costs。
- run_backtest / walk_forward_backtest 实际跑通（回归：成本 import 未遗漏）。
"""
import numpy as np
import pandas as pd
import sqlite3

from fwsp import costs, execution
from fwsp.db import init_schema
from fwsp.factor_mining import long_short_backtest


def _seed(conn, n_days=160, codes=("000001", "000002", "000003")):
    init_schema(conn)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(1)
    for c in codes:
        for i, d in enumerate(dates):
            close = 10 + rng.standard_normal() + i * 0.01
            op = close * (1 + rng.standard_normal() * 0.002)
            hi = max(close, op) * 1.01
            lo = min(close, op) * 0.99
            vol = 1e6 + abs(rng.standard_normal()) * 1e5
            conn.execute(
                "INSERT INTO daily (code,date,open,high,low,close,volume,amount) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (c, d.strftime("%Y-%m-%d"), op, hi, lo, close, vol, vol * close))
    for d in dates[::5]:
        conn.execute("INSERT INTO index_daily (code,date,close) VALUES (?,?,?)",
                     ("sh.000300", d.strftime("%Y-%m-%d"), 3000 + rng.standard_normal()))
    conn.commit()


def test_run_backtest_runs():
    from fwsp.backtest import run_backtest
    conn = sqlite3.connect(":memory:")
    _seed(conn)
    r = run_backtest(start="2024-03-01", end="2024-06-01", top_n=3,
                     hold_days=10, strategy="reversal")
    assert "total_return" in r and np.isfinite(r["total_return"])
    assert "sharpe" in r
    conn.close()



def _pos(buy=100.0):
    return {"buy_px": buy, "peak": buy}


def test_decide_exit_stop_open_gap():
    px, reason = execution.decide_exit(_pos(), 90, 88, 101, -8, 10, None, 5, 10)
    assert reason == "stop" and px == 90


def test_decide_exit_stop_intraday():
    px, reason = execution.decide_exit(_pos(), 95, 90, 101, -8, 10, None, 5, 10)
    # 盘中触及止损价 → 以 min(开盘, 止损价) 离场
    assert reason == "stop" and px == 92


def test_decide_exit_take_profit():
    px, reason = execution.decide_exit(_pos(), 100, 99, 112, -8, 10, None, 5, 10)
    # 日内高点触及目标价 → 以目标价离场（用 high，非 close）
    assert reason == "tp"
    assert abs(px - 110) < 1e-9


def test_decide_exit_expire():
    px, reason = execution.decide_exit(_pos(), 100, 99, 105, -8, 10, None, 10, 10)
    assert reason == "expire" and px == 100


def test_decide_exit_expires_at_horizon_minus_one():
    # 与 forward_returns 的 open[t+1]→open[t+horizon] 持仓期严格对齐（消 off-by-one）
    pos = _pos()
    px, reason = execution.decide_exit(pos, 100, 99, 105, -8, 10, None, 9, 10)
    assert reason == "expire" and px == 100
    px2, reason2 = execution.decide_exit(pos, 100, 99, 105, -8, 10, None, 8, 10)
    assert reason2 is None  # 未到 horizon-1 日，无止损/止盈时不离场


def test_position_return_applies_both_costs():
    r = execution.position_return(110, 100)
    exp = (110 * (1 - costs.COST_SELL)) / (100 * (1 + costs.COST_BUY)) - 1
    assert abs(r - exp) < 1e-12


def test_long_short_backtest_cost_source():
    n, codes = 200, 10
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"{c:06d}" for c in range(codes)]
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.standard_normal((n, codes)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.standard_normal((n, codes)) * 0.01, index=idx,
                       columns=cols)
    r_low = long_short_backtest(f, fwd)
    r_high = long_short_backtest(f, fwd, cost_buy=0.01, cost_sell=0.01)
    # 成本不同 → 净 IR 应不同（成本被真实计入）
    if np.isfinite(r_low["net_ir"]) and np.isfinite(r_high["net_ir"]):
        assert r_low["net_ir"] != r_high["net_ir"]
    # 默认成本单一来源自 costs
    assert long_short_backtest.__defaults__[0] is costs.COST_BUY
    assert long_short_backtest.__defaults__[1] is costs.COST_SELL


def test_long_short_backtest_delegates_to_execution():
    # 晋升路径的回测必须落到 execution 层（单一来源）
    from fwsp import execution, factor_mining
    n, codes = 200, 10
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"{c:06d}" for c in range(codes)]
    rng = np.random.default_rng(7)
    f = pd.DataFrame(rng.standard_normal((n, codes)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.standard_normal((n, codes)) * 0.01, index=idx,
                       columns=cols)
    r_fm = factor_mining.long_short_backtest(f, fwd)
    r_ex = execution.long_short_backtest(f, fwd)
    assert abs(r_fm["net_ir"] - r_ex["net_ir"]) < 1e-12
    assert abs(r_fm["net_ret"] - r_ex["net_ret"]) < 1e-12
    # 成本常量同样来自 costs
    assert execution.long_short_backtest.__defaults__[1] is costs.COST_BUY
    assert execution.long_short_backtest.__defaults__[2] is costs.COST_SELL


def test_long_short_backtest_cost_is_two_legged():
    # 锁定「日频调仓多空双腿各一轮买卖 → 日成本 = 2×(cost_buy+cost_sell)」。
    # 回归保护：回退到 1× 或放大到 10× 都会让本测试翻红。
    n, codes = 300, 20
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    cols = [f"{c:06d}" for c in range(codes)]
    rng = np.random.default_rng(3)
    f = pd.DataFrame(rng.standard_normal((n, codes)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.standard_normal((n, codes)) * 0.01, index=idx,
                       columns=cols)
    r0 = long_short_backtest(f, fwd, cost_buy=0.0, cost_sell=0.0)
    X, Y = 0.001, 0.002
    r1 = long_short_backtest(f, fwd, cost_buy=X, cost_sell=Y)
    # 重建零成本组合序列，取其标准差（常数平移不改变 std）
    r = f.rank(axis=1, pct=True)
    long_ret = fwd.where(r >= 0.9).mean(axis=1)
    short_ret = fwd.where(r <= 0.1).mean(axis=1)
    port_gross = (long_ret - short_ret).dropna()
    std_g = port_gross.std()
    expected_drop = 2 * (X + Y) / std_g * np.sqrt(252)
    assert abs((r0["net_ir"] - r1["net_ir"]) - expected_drop) < 1e-9, \
        f"成本差应严格等于 2×(X+Y)/std×√252，实得 {r0['net_ir'] - r1['net_ir']}"
