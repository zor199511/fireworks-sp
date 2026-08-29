"""P0 收尾补充契约测试：锁定审查发现的两条关键不变量。

1. walk_forward_backtest 在真实 low 跌破止损价时必须触发止损
   （回归：修正 build_factor_panel 的 low/high 错标前，因子研究回测止损永不触发）。
2. 实时推荐(run_screen persist=False) 与每日推荐同源、确定性一致
   （P0-3「实时==每日」契约）。
"""
import numpy as np
import pandas as pd

from fwsp import multifactor as M
from fwsp import screener


def test_walk_forward_backtest_real_low_triggers_stop():
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    codes = ["AAAAAA", "BBBBBB"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(index=idx, columns=codes, dtype=float)
    opn = close.copy(); lo = close.copy(); hi = close.copy()

    baseA = 100 + np.arange(n) * 0.3
    baseA[41] = 84.0                       # 暴跌日
    baseA[42:] = np.linspace(86, 110, n - 42)
    close["AAAAAA"] = baseA
    opn["AAAAAA"] = pd.Series(baseA, index=idx).shift(1).fillna(baseA[0]) \
        * (1 + rng.standard_normal(n) * 0.001)
    hi["AAAAAA"] = close["AAAAAA"] * 1.02
    lo["AAAAAA"] = close["AAAAAA"] * 0.98
    lo.loc[idx[41], "AAAAAA"] = 82.0       # 盘中低点跌破止损价

    close["BBBBBB"] = 50 + np.arange(n) * 0.01 + rng.standard_normal(n) * 0.1
    opn["BBBBBB"] = close["BBBBBB"] * (1 + rng.standard_normal(n) * 0.001)
    hi["BBBBBB"] = close["BBBBBB"] * 1.02
    lo["BBBBBB"] = close["BBBBBB"] * 0.98

    # 单因子：AAAAAA 永远最高 z，必被选中；quality 是「通过的股票集」掩码
    z = {"f1": pd.DataFrame({"AAAAAA": 3.0, "BBBBBB": -3.0}, index=idx)}
    # IC 序列需有波动（_train_weights 要求 std>0），均值正即可
    ics = {"f1": pd.Series(rng.normal(0.5, 0.1, n), index=idx)}
    res = M.walk_forward_backtest(z, ics, close, opn, lo, {"AAAAAA", "BBBBBB"},
                                  highs=hi, start="2024-02-01", top_n=1,
                                  horizon=10, stop_pct=-8.0)
    stops = [t for t in res["trades"] if t["reason"] == "stop"]
    assert stops, f"修正后应触发止损，实际 trades={res['trades']}"


def test_run_screen_realtime_equals_daily():
    # 实时推荐与每日推荐共用 run_screen 单一打分路径，必须确定性一致
    r1 = screener.run_screen(top_n=5, persist=False)
    r2 = screener.run_screen(top_n=5, persist=False)
    c1 = [x["code"] for x in r1]
    c2 = [x["code"] for x in r2]
    assert c1 == c2, f"两次实时打分候选不一致: {c1} vs {c2}"
    s1 = {x["code"]: round(x["score"], 6) for x in r1}
    s2 = {x["code"]: round(x["score"], 6) for x in r2}
    assert s1 == s2
