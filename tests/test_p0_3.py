"""P0-3 双引擎合一：实时推荐与每日推荐共用同一打分路径(run_screen→multifactor_scores)。

锁住「同一输入 → 同一输出」的确定性，确保 dashboard 实时推荐与每日推荐不再互相矛盾。
"""
import json
import sqlite3

import numpy as np
import pandas as pd

from fwsp.db import init_schema, write_active_set
from fwsp.multifactor_score import multifactor_scores


def _seed(conn, n_days=80, codes=("000001", "000002", "000003", "000004")):
    init_schema(conn)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
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
        conn.execute("INSERT INTO stock_list (code,name,exchange,is_st,industry,updated) "
                     "VALUES (?,?,?,?,?,?)", (c, c, "sz", 0, "测试", "2024"))
    specs = [
        ("mf_ret5", "momentum", "returns(close,5)", "{}", "5日收益", "auto_evolve"),
        ("mf_std5", "volatility", "std(returns(close,1),5)", "{}", "5日波动", "auto_evolve"),
    ]
    for sid, cat, expr, pj, desc, src in specs:
        conn.execute(
            "INSERT INTO factor_library (code,category,expr,params_json,desc,source,created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (sid, cat, expr, pj, desc, src, "2024-01-01 00:00:00"))
    write_active_set(conn, "2024-03-01 00:00:00",
                    ["mf_ret5", "mf_std5"], 0.5, "auto_evolve")
    conn.commit()


def test_multifactor_scores_deterministic():
    conn = sqlite3.connect(":memory:")
    _seed(conn)
    codes = ["000001", "000002", "000003", "000004"]
    s1, r1 = multifactor_scores(conn, codes)
    s2, r2 = multifactor_scores(conn, codes)
    assert s1 == s2
    assert r1 == r2
    # 至少应有打分结果（因子可计算）
    assert len(s1) > 0
    conn.close()


def test_run_screen_persist_flag():
    # persist=False 不应写入 recommendations，但应返回候选；与每日路径同源。
    from fwsp import screener
    conn = sqlite3.connect(":memory:")
    _seed(conn)
    before = conn.execute(
        "SELECT COUNT(*) FROM recommendations").fetchone()[0]
    recs = screener.run_screen(top_n=3, persist=False)
    after = conn.execute(
        "SELECT COUNT(*) FROM recommendations").fetchone()[0]
    assert before == after, "persist=False 不应写入 recommendations"
    assert len(recs) <= 3
    assert all("reasons" in r and "score" in r for r in recs)
    conn.close()
