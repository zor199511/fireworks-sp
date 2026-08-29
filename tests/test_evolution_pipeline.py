"""进化管线的测试（纯函数测试 + auto_evolve --promote 端到端写库链路）。"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import fwsp.config as cfg
from fwsp import db
from fwsp import factor_mining as fm
from fwsp.db import init_schema, get_active_set
from fwsp.community_watch import (load_base_recipes, load_community_recipes,
                                  merge_all_recipes)
from fwsp.overfit_guard import stability_check
import auto_evolve as AE  # noqa: E402  (脚本，需 scripts 在 path)


# --------------------------------------------------------------------------
# 纯函数测试（不依赖真实 DB，回归此前行为）
# --------------------------------------------------------------------------
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
    assert np.isfinite(r["net_ir"]) or np.isnan(r["net_ir"])


def test_long_short_backtest_reward_strong_factor():
    panels = _synthetic_panels()
    fwd = fm.forward_returns(panels, horizon=10)
    # 用未来收益本身的同频信号构造因子（仅验证函数数值行为，非真实因子）
    strong = fwd
    r = fm.long_short_backtest(strong, fwd)
    assert r["net_ir"] > 5  # 强信号应得很高净 IR（扣 2× 成本后仍显著为正）


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
    assert len(out) >= 1  # 流式门禁不崩溃、返回子集


# --------------------------------------------------------------------------
# auto_evolve --promote 端到端写库链路
# --------------------------------------------------------------------------
def _seed_synthetic_db(path: Path) -> None:
    """构造强横截面结构的合成日线，使动量类因子可稳定通过 ICIR 门禁。"""
    n_stk, n_day = 60, 800
    rng = np.random.default_rng(7)
    codes = [f"{i:06d}" for i in range(n_stk)]
    dates = pd.bdate_range("2023-01-02", periods=n_day)
    alpha = rng.uniform(-0.004, 0.004, size=n_stk)  # 每只股票固定 drift

    recs = []
    for i, code in enumerate(codes):
        a = alpha[i]
        px = 10.0
        for d in dates:
            ret = a + rng.normal(0, 0.003)
            op = px * (1 + rng.normal(0, 0.001))
            cl = op * (1 + ret)
            hi = max(op, cl) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(op, cl) * (1 - abs(rng.normal(0, 0.002)))
            recs.append((code, d.strftime("%Y-%m-%d"), op, hi, lo, cl, 1e6, 1e9))
            px = cl
    df = pd.DataFrame(recs, columns=["code", "date", "open", "high", "low",
                                     "close", "volume", "amount"])
    with sqlite3.connect(path) as conn:
        init_schema(conn)
        df.to_sql("daily", conn, if_exists="append", index=False)


@pytest.fixture
def promote_db(tmp_path, monkeypatch):
    db_path = tmp_path / "fw_test.db"
    _seed_synthetic_db(db_path)
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    return db_path


def test_run_evolution_promotes_and_writes(promote_db):
    out = AE.run_evolution(dry_run=False)
    assert out["promoted"] is True, out.get("reason")
    selected = out["selected"]
    assert len(selected) > 0, "应有入选因子"

    import json
    with db.get_conn() as conn:
        n_lib = conn.execute("SELECT COUNT(*) FROM factor_library").fetchone()[0]
        n_eval = conn.execute("SELECT COUNT(*) FROM factor_eval").fetchone()[0]
        n_log = conn.execute("SELECT COUNT(*) FROM evolution_log").fetchone()[0]
        log_row = conn.execute(
            "SELECT selected_json FROM evolution_log ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
        a = get_active_set(conn, "auto_evolve")
        sel_codes = {r[0] for r in conn.execute(
            "SELECT code FROM factor_eval WHERE selected=1").fetchall()}

    assert n_lib == len(selected), "factor_library 应与入选一致"
    assert n_eval == len(selected), "factor_eval 应与入选一致"
    assert n_log == 1, "应写入一条进化日志"
    assert log_row is not None
    assert json.loads(log_row[0]) == selected, "日志应记录入选集"
    assert a is not None, "应写入 active_sets"
    # auto_demote 可能再剔除 net_ir<=0/NaN 的因子，故 active 是 selected 的子集
    assert set(a["factors"]).issubset(set(selected))
    # factor_eval.selected 镜像必须与 active_sets 完全一致
    assert sel_codes == set(a["factors"])
    # 活跃因子的净成本 IR 必须严格 > 0（晋升门禁 + 衰减降级保证）
    with db.get_conn() as conn:
        ph = ",".join("?" * len(a["factors"]))
        rows = conn.execute(
            "SELECT code, net_ir FROM factor_eval WHERE (code, run_at) IN ("
            f"SELECT code, MAX(run_at) FROM factor_eval WHERE code IN ({ph}) "
            f"GROUP BY code)", a["factors"]).fetchall()
    for code, nir in rows:
        assert nir is not None and not (isinstance(nir, float) and np.isnan(nir)) \
            and nir > 0, f"活跃因子 {code} 净IR应>0，实得 {nir}"


def test_run_evolution_dry_run_writes_nothing(tmp_path, monkeypatch):
    db_path = tmp_path / "fw_dry.db"
    _seed_synthetic_db(db_path)
    monkeypatch.setattr(cfg, "DB_PATH", db_path)

    out = AE.run_evolution(dry_run=True)
    assert out["dry_run"] is True
    assert out["promoted"] is False

    with db.get_conn() as conn:
        n_log = conn.execute("SELECT COUNT(*) FROM evolution_log").fetchone()[0]
        a = get_active_set(conn, "auto_evolve")
    assert n_log == 0, "dry_run 不应写进化日志"
    assert a is None, "dry_run 不应写 active_sets"


# --------------------------------------------------------------------------
# 更严格晋升门禁（过拟合/失稳硬剔除）
# --------------------------------------------------------------------------
class TestFactorDropReason:
    def test_healthy_factor_kept(self):
        res = {"net_ir": 1.2, "oos_is_ratio": 1.5, "stability": 0.3}
        assert AE._factor_drop_reason(res) is None

    def test_nonpositive_net_ir_dropped(self):
        assert AE._factor_drop_reason({"net_ir": 0.0, "stability": 0.3}) \
            is not None
        assert AE._factor_drop_reason({"net_ir": float("nan"), "stability": 0.3}) \
            is not None

    def test_overfit_high_oos_is_ratio_dropped(self):
        res = {"net_ir": 2.0, "oos_is_ratio": 9.0, "stability": 0.5}
        reason = AE._factor_drop_reason(res)
        assert reason is not None and "过拟合" in reason

    def test_unstable_negative_worst_icir_dropped(self):
        res = {"net_ir": 1.0, "oos_is_ratio": 1.2, "stability": -0.4}
        reason = AE._factor_drop_reason(res)
        assert reason is not None and "失稳" in reason

    def test_normal_factors_not_dropped(self):
        specs = [
            {"net_ir": 0.8, "oos_is_ratio": 2.0, "stability": 0.1},
            {"net_ir": 1.5, "oos_is_ratio": 4.9, "stability": 0.0},
        ]
        assert all(AE._factor_drop_reason(s) is None for s in specs)
