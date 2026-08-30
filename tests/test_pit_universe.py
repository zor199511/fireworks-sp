"""PIT universe 边界 + 财报期格式兼容测试。

覆盖:
1. code_first_last_dates 从 daily 取首末日
2. _pit_boundaries 向量化排除上市前 / 退市后
3. quality_panel period 格式自动探测(20260331 / 2026-03-31)
4. quality_panel as_of 缺失时回退 period+30d
5. quality_panel PIT 边界(新 code 在 first_date 前应 False)
6. quality_panel PIT 边界(002155 退市股在 last_date 后应 False)
"""
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

from fwsp import factors as F
from fwsp.db import init_schema


def _seed_stock_list(conn, codes_with_st: dict):
    rows = [(c, f"代码{c}", "sh" if c.startswith("6") else "sz", st, "测试", "2024-01-01")
            for c, st in codes_with_st.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO stock_list (code,name,exchange,is_st,industry,updated) "
        "VALUES (?,?,?,?,?,?)", rows)


def _seed_spot(conn, codes_mv: dict):
    rows = [(c, 10.0, 0, 1e6, 1e7, 1.0, 1.0, 15.0, 1.5, mv, mv, 0, "2024-06-30")
            for c, mv in codes_mv.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO spot (code,price,pct_chg,volume,amount,turnover,vol_ratio,"
        "pe_dyn,pb,total_mv,circ_mv,chg_60d,updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)


def _seed_daily(conn, code, dates, base_price=10.0):
    rows = []
    for i, d in enumerate(dates):
        p = base_price + i * 0.1
        rows.append((code, d, p, p + 0.1, p - 0.1, p, 1e6, 1e7))
    conn.executemany(
        "INSERT OR REPLACE INTO daily (code,date,open,high,low,close,volume,amount) "
        "VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def _seed_fin(conn, code, period, roe=10.0, profit_yoy=5.0, debt_ratio=50.0,
              as_of=None):
    # period 接受 "20240331" 或 "2024-03-31"
    conn.execute(
        "INSERT OR REPLACE INTO fin_q (code,period,eps,roe,gross_margin,"
        "profit_yoy,debt_ratio,as_of,updated) VALUES (?,?,?,?,?,?,?,?,?)",
        (code, period, 1.0, roe, 30.0, profit_yoy, debt_ratio,
         as_of, "2024-06-30"))


# -------------------------------------------------------------- code_first_last_dates

class TestCodeFirstLastDates:
    def test_returns_first_and_last(self):
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        dates = ["2024-01-05", "2024-01-30", "2024-02-15",
                 "2024-02-28", "2024-03-15", "2024-03-30"]
        _seed_daily(conn, "600000", dates, 10.0)
        fl = F.code_first_last_dates(conn)
        assert len(fl) == 1
        assert fl.iloc[0]["code"] == "600000"
        assert str(fl.iloc[0]["first"].date()) == "2024-01-05"
        assert str(fl.iloc[0]["last"].date()) == "2024-03-30"
        conn.close()

    def test_empty_daily(self):
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        fl = F.code_first_last_dates(conn)
        assert fl.empty
        conn.close()


# -------------------------------------------------------------- _pit_boundaries

class TestPitBoundaries:
    def test_excludes_pre_ipo_and_post_delist(self):
        # 构造 4 只 code, 上市/退市日期不同
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        piv = pd.DataFrame(True, index=dates, columns=["A", "B", "C", "D"])
        code_fl = pd.DataFrame({
            "code": ["A", "B", "C", "D"],
            "first": pd.to_datetime(["2024-01-01", "2024-04-01",
                                     "2024-01-01", "2024-01-01"]),
            "last": pd.to_datetime(["2024-12-31", "2024-12-31",
                                    "2024-09-30", "2024-12-31"]),
        })
        out = F._pit_boundaries(piv, code_fl)
        # B: 2024-01 ~ 2024-03 应 False (pre-IPO)
        assert not bool(out.loc["2024-02-01", "B"])
        assert bool(out.loc["2024-05-01", "B"])
        # C: 2024-10 之后应 False (delisted)
        assert bool(out.loc["2024-09-15", "C"])
        assert not bool(out.loc["2024-10-15", "C"])
        # A/D: 全程 True
        assert out["A"].all()
        assert out["D"].all()
        # pre-IPO 的 B 与 B 真值不影响其他列
        assert out["A"].sum() == len(dates)

    def test_empty_inputs(self):
        empty_df = pd.DataFrame()
        code_fl = pd.DataFrame({"code": ["A"], "first": [pd.Timestamp("2024-01-01")],
                                 "last": [pd.Timestamp("2024-12-31")]})
        # piv 空 → 返原
        assert F._pit_boundaries(empty_df, code_fl).empty
        # code_fl 空 → 返原
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        piv = pd.DataFrame(True, index=dates, columns=["A"])
        out = F._pit_boundaries(piv, empty_df)
        assert out["A"].all()

    def test_vectorized_matches_loop(self):
        """PIT 边界向量化结果与逐 code 循环一致。"""
        import random
        random.seed(42)
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
        codes = [f"{i:06d}" for i in range(50)]
        piv = pd.DataFrame(True, index=dates, columns=codes)
        first_dates = [dates[random.randint(0, 50)] for _ in codes]
        last_dates = [dates[random.randint(50, len(dates) - 1)] for _ in codes]
        code_fl = pd.DataFrame({
            "code": codes,
            "first": pd.to_datetime(first_dates),
            "last": pd.to_datetime(last_dates),
        })
        # 期望:逐 code 循环
        expected = piv.copy()
        for c, f, l in zip(codes, first_dates, last_dates):
            expected[c] = (expected.index >= f) & (expected.index <= l)
        out = F._pit_boundaries(piv, code_fl)
        # 验证 columns 一致
        assert list(out.columns) == list(expected.columns)
        # 逐元素比较(转 bool)
        assert (out.values == expected.values).all()


# -------------------------------------------------------------- quality_panel period 格式

class TestQualityPanelPeriodFallback:
    def test_period_format_mixed_accepted(self):
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        _seed_stock_list(conn, {"600000": 0})
        _seed_spot(conn, {"600000": 5e9})
        # 同时支持 "20240331" 与 "2024-03-31" 两种格式
        _seed_fin(conn, "600000", "20240331", roe=12.0, as_of="2024-05-15")
        _seed_fin(conn, "600000", "2024-06-30", roe=10.0, as_of="2024-08-20")
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        piv = F.quality_panel(conn, dates)
        assert "600000" in piv.columns
        # 5-15 之前无 fin → False; 5-15 后 True (有 roe>=0)
        assert not piv.loc["2024-05-01", "600000"]
        assert piv.loc["2024-06-01", "600000"]
        conn.close()

    def test_as_of_null_falls_back_to_period_plus_30d(self):
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        _seed_stock_list(conn, {"600000": 0})
        _seed_spot(conn, {"600000": 5e9})
        # as_of NULL, period=2024-03-31 → 应回退到 2024-04-30
        _seed_fin(conn, "600000", "2024-03-31", roe=10.0, as_of=None)
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        piv = F.quality_panel(conn, dates)
        assert "600000" in piv.columns
        # 2024-04-30 之前应 False (period+30d 才是 as_of)
        assert not piv.loc["2024-04-15", "600000"]
        # 2024-04-30 当天或之后应 True
        assert piv.loc["2024-05-01", "600000"]
        conn.close()

    def test_pit_universe_excludes_pre_ipo(self):
        """新 code 上市前不进入 universe。"""
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        _seed_stock_list(conn, {"600000": 0})
        _seed_spot(conn, {"600000": 5e9})
        # 600000 daily 只从 2024-06 起
        dates_late = pd.date_range("2024-06-01", "2024-12-31", freq="D")
        _seed_daily(conn, "600000", [d.strftime("%Y-%m-%d") for d in dates_late])
        _seed_fin(conn, "600000", "2024-03-31", roe=10.0)
        _seed_fin(conn, "600000", "2024-06-30", roe=10.0)
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        piv = F.quality_panel(conn, dates)
        if "600000" in piv.columns:
            # 2024-05-31 之前应 False (daily first = 2024-06-01)
            pre_ipo = piv.loc["2024-05-31", "600000"]
            assert not pre_ipo, "PIT 边界未排除上市前"
        conn.close()

    def test_pit_universe_excludes_post_delist(self):
        """002155 退市股在 last_date 之后应 False。"""
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        _seed_stock_list(conn, {"002155": 0})
        _seed_spot(conn, {"002155": 5e9})
        # daily 只到 2024-08-19
        dates_partial = pd.date_range("2024-01-01", "2024-08-19", freq="D")
        _seed_daily(conn, "002155", [d.strftime("%Y-%m-%d") for d in dates_partial])
        _seed_fin(conn, "002155", "2024-03-31", roe=10.0, as_of="2024-05-15")
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        piv = F.quality_panel(conn, dates)
        if "002155" in piv.columns:
            # 2024-08-19 之前应 True
            assert piv.loc["2024-08-15", "002155"]
            # 2024-08-19 之后应 False
            assert not piv.loc["2024-08-25", "002155"], "PIT 边界未排除退市后"
        conn.close()
