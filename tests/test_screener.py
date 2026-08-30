import numpy as np
import pandas as pd
import pytest

from fwsp import config as cfg
from fwsp import db as fwdb
from fwsp.screener import (apply_industry_cap, passes_hard_filters,
                           score_technical, run_screen)


def _seed_screenable_db(path, code="000001"):
    with fwdb.get_conn(path) as conn:
        fwdb.init_schema(conn)
        conn.execute(
            "INSERT INTO stock_list(code,name,industry,exchange,is_st) "
            "VALUES (?,?,?,?,?)", (code, "测试", "银行", "sz", 0))
        conn.execute(
            "INSERT INTO spot(code,price,pe_dyn,pb,total_mv,circ_mv,turnover) "
            "VALUES (?,?,?,?,?,?,?)", (code, 10.0, 20.0, 2.0, 100e8, 80e8, 2.0))
        conn.execute(
            "INSERT INTO fin_q(code,period,roe,debt_ratio,gross_margin,profit_yoy) "
            "VALUES (?,?,?,?,?,?)", (code, "2026Q1", 12.0, 50.0, 30.0, 10.0))
        dates = pd.date_range("2024-01-01", periods=140, freq="D")
        closes = np.linspace(8, 12, len(dates))
        rows = [(code, str(d.date()), closes[i] * 0.99, closes[i] * 1.01,
                 closes[i] * 0.98, closes[i], 1e6, 1e8)
                for i, d in enumerate(dates)]
        conn.executemany(
            "INSERT INTO daily(code,date,open,high,low,close,volume,amount) "
            "VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()


class TestFactorSystemDegradedMeta:
    def test_no_active_factors_sets_degraded(self, monkeypatch, tmp_path):
        db_path = tmp_path / "screen.db"
        _seed_screenable_db(db_path)
        monkeypatch.setattr(cfg, "DB_PATH", db_path)
        # 无 active_factors → multifactor_scores 返回空 → 回退兜底分支
        monkeypatch.setattr(
            "fwsp.screener.multifactor_scores",
            lambda c, codes: ({}, {}))
        run_screen(top_n=10, persist=True)
        with fwdb.get_conn(db_path) as conn:
            assert fwdb.get_meta(conn, "factor_system_degraded") == "1"

    def test_active_factors_clears_degraded(self, monkeypatch, tmp_path):
        db_path = tmp_path / "screen.db"
        _seed_screenable_db(db_path)
        monkeypatch.setattr(cfg, "DB_PATH", db_path)
        # 有 active_factors → 多因子合成分支 → 清除降级标记
        monkeypatch.setattr(
            "fwsp.screener.multifactor_scores",
            lambda c, codes: ({"000001": 90.0}, {"000001": ["测试因子"]}))
        run_screen(top_n=10, persist=True)
        with fwdb.get_conn(db_path) as conn:
            assert fwdb.get_meta(conn, "factor_system_degraded") == "0"

CFG = {
    "min_total_mv": 50e8,
    "max_pe": 40.0,
    "min_roe": 8.0,
    "max_debt": 65.0,
}


def _row(**kw):
    base = {"total_mv": 100e8, "pe_dyn": 20.0, "roe": 12.0, "debt_ratio": 50.0}
    base.update(kw)
    return base


class TestHardFilters:
    def test_healthy_stock_passes(self):
        assert passes_hard_filters(_row(), CFG) == []

    def test_small_market_cap_fails(self):
        fails = passes_hard_filters(_row(total_mv=30e8), CFG)
        assert any("市值" in f for f in fails)

    def test_zero_pe_fails(self):
        fails = passes_hard_filters(_row(pe_dyn=0), CFG)
        assert any("PE" in f for f in fails)

    def test_negative_pe_fails(self):
        assert passes_hard_filters(_row(pe_dyn=-46.0), CFG)

    def test_high_pe_fails(self):
        assert passes_hard_filters(_row(pe_dyn=80.0), CFG)

    def test_missing_pe_fails(self):
        assert passes_hard_filters(_row(pe_dyn=None), CFG)

    def test_low_roe_fails(self):
        fails = passes_hard_filters(_row(roe=3.0), CFG)
        assert any("ROE" in f for f in fails)

    def test_missing_roe_fails(self):
        assert passes_hard_filters(_row(roe=None), CFG)

    def test_high_debt_fails_and_null_debt_passes(self):
        # 子代理 4 critical #1: 222 高负债股绕过 debt_ratio 门.
        # 实际数据库 debt_ratio NULL 是普遍现象, 直接拒收 = 0 推荐.
        # 改为 None 通过, 由 PIT 质量面板(max_debt=85)兜底, 写 meta
        # 标记让 dashboard 知道缺失数.
        assert passes_hard_filters(_row(debt_ratio=70.0), CFG)
        # None 视为未披露, 走 PIT 兜底
        assert passes_hard_filters(_row(debt_ratio=None), CFG) == []


@pytest.fixture
def ft_base():
    return {
        "close": 10.0, "ma5": 9.9, "ma10": 10.4, "ma20": 10.2,
        "ma60": 9.8, "ma120": 9.5, "dif": -0.02, "dea": -0.05,
        "rsi": 32.0, "vol": 2_000_000.0, "vol_ma5": 1_500_000.0,
        "vol_ma20": 1_400_000.0, "high20": 11.0, "high250": 12.0,
        "amount_ma20": 1e8, "ret_20d": -18.0, "ret_60d": -25.0,
        "open_today": 10.1, "chg_pct": 2.0, "vol_prev": 1_800_000.0,
        "date": "2026-08-25",
    }


class TestReversalScoring:
    def test_deep_pullback_stabilizing_scores_high(self, ft_base):
        score, reasons = score_technical(ft_base)
        assert score >= 85
        assert any("回调" in r for r in reasons)
        assert any("企稳" in r for r in reasons)

    def test_no_signal_scores_low(self, ft_base):
        ft_base.update({"ret_20d": 5.0, "rsi": 60.0, "ma5": 10.5,
                        "chg_pct": -1.0, "dif": -0.01, "dea": 0.01})
        score, _ = score_technical(ft_base)
        assert score <= 35

    def test_score_clamped_to_100(self, ft_base):
        score, _ = score_technical(ft_base)
        assert 0 <= score <= 100

    def test_rsi_not_oversold_no_rsi_reason(self, ft_base):
        ft_base["rsi"] = 55.0
        _, reasons = score_technical(ft_base)
        assert not any("RSI" in r for r in reasons)

    def test_macd_dead_cross_no_bottom_golden(self, ft_base):
        ft_base["dif"], ft_base["dea"] = -0.03, -0.01
        _, reasons = score_technical(ft_base)
        assert not any("底部MACD金叉" in r for r in reasons)


def _stock(code, score, industry):
    return {"code": code, "score": score, "industry": industry}


class TestIndustryDiversity:
    def test_same_industry_capped(self):
        scored = [_stock(f"A{i}", 100 - i, "贵金属") for i in range(5)]
        scored += [_stock("B1", 50, "银行")]
        top = apply_industry_cap(scored, top_n=5, cap=3)
        picked = [r["code"] for r in top]
        assert picked == ["A0", "A1", "A2", "B1"]
        counts = {}
        for r in top:
            counts[r["industry"]] = counts.get(r["industry"], 0) + 1
        assert all(c <= 3 for c in counts.values())

    def test_none_industry_unlimited(self):
        scored = [_stock(f"N{i}", 100 - i, None) for i in range(6)]
        top = apply_industry_cap(scored, top_n=5, cap=3)
        assert [r["code"] for r in top] == ["N0", "N1", "N2", "N3", "N4"]

    def test_empty_string_industry_unlimited(self):
        scored = [_stock(f"E{i}", 100 - i, "") for i in range(4)]
        top = apply_industry_cap(scored, top_n=4, cap=1)
        assert len(top) == 4

    def test_insufficient_candidates_returns_actual_count(self):
        scored = [_stock("A1", 90, "贵金属"), _stock("A2", 80, "贵金属"),
                  _stock("A3", 70, "贵金属"), _stock("A4", 60, "贵金属"),
                  _stock("B1", 50, "银行")]
        top = apply_industry_cap(scored, top_n=10, cap=3)
        assert len(top) == 4
        assert [r["code"] for r in top] == ["A1", "A2", "A3", "B1"]

    def test_exact_top_n_filled(self):
        scored = [_stock(f"S{i}", 100 - i, f"行业{i}") for i in range(10)]
        top = apply_industry_cap(scored, top_n=10, cap=3)
        assert len(top) == 10
