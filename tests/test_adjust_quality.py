"""复权 + 幸存者偏差收口测试。

覆盖：
- load_panels 优先读 daily_qfq、缺失回退 daily（前复权消除除权日跳变）；
- forward_returns 在 qfq 连续价下不会因除权产生虚假收益；
- quality_mask 改为 point-in-time bool DataFrame（财报披露前 False、披露后 True），
  消除『用今天财报评判历史』的幸存者偏差；
- rank_ic_series 支持 DataFrame 时间变化质量集逐日过滤。
"""
import sqlite3

import numpy as np
import pandas as pd

from fwsp import factors as F
from fwsp.db import init_schema
from fwsp.multifactor import rank_ic_series


def _seed_daily(conn, codes_dates):
    """codes_dates: list[(code,date,open,high,low,close,volume,amount)]"""
    conn.executemany(
        "INSERT OR REPLACE INTO daily "
        "(code,date,open,high,low,close,volume,amount) VALUES (?,?,?,?,?,?,?,?)",
        codes_dates)
    conn.commit()


def _seed_qfq(conn, codes_dates):
    conn.executemany(
        "INSERT OR REPLACE INTO daily_qfq "
        "(code,date,open,high,low,close,volume,amount) VALUES (?,?,?,?,?,?,?,?)",
        codes_dates)
    conn.commit()


def _seed_fin(conn, rows):
    """rows: list[(code,period,as_of,roe,debt_ratio)]"""
    conn.executemany(
        "INSERT OR REPLACE INTO fin_q "
        "(code,period,as_of,roe,debt_ratio) VALUES (?,?,?,?,?)", rows)
    conn.commit()


def _seed_stock_list(conn, codes):
    conn.executemany(
        "INSERT OR REPLACE INTO stock_list (code,name,exchange,is_st,industry) "
        "VALUES (?,?,?,?,?)",
        [(c, c, "sh" if c[0] == "6" else "sz", 0, "测试") for c in codes])
    conn.commit()


class TestLoadPanelsAdjust:
    def test_qfq_preferred_when_present(self):
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        # daily 不复权（含除权跳变）：第 2 天价从 10 跌到 7
        raw = [("600000", d, o, o, o, o, 1e6, 1e7)
               for d, o in zip(dates, [10.0, 7.0, 10.2])]
        # daily_qfq 连续（前复权）：除权日无跳变
        qfq = [("600000", d, o, o, o, o, 1e6, 1e7)
               for d, o in zip(dates, [10.0, 10.0, 10.2])]
        _seed_daily(conn, raw)
        _seed_qfq(conn, qfq)

        p = F.load_panels(conn, adjust="qfq")
        # qfq 优先：open 序列连续，不应出现 7.0 的跳变
        assert list(p["open"]["600000"].values) == [10.0, 10.0, 10.2]
        conn.close()

    def test_fallback_to_daily_when_qfq_empty(self):
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        dates = ["2024-01-02", "2024-01-03"]
        raw = [("600000", d, o, o, o, o, 1e6, 1e7)
               for d, o in zip(dates, [10.0, 10.5])]
        _seed_daily(conn, raw)
        # 不写 daily_qfq

        p = F.load_panels(conn, adjust="qfq")  # 应回退 daily
        assert list(p["close"]["600000"].values) == [10.0, 10.5]
        conn.close()

    def test_raw_adjust_still_works(self):
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        dates = ["2024-01-02", "2024-01-03"]
        raw = [("600000", d, o, o, o, o, 1e6, 1e7)
               for d, o in zip(dates, [10.0, 7.0])]
        _seed_daily(conn, raw)
        p = F.load_panels(conn, adjust="")  # 显式不复权
        assert list(p["close"]["600000"].values) == [10.0, 7.0]
        conn.close()


class TestForwardReturnsQfq:
    def test_qfq_removes_ex_right_jump(self):
        """除权日不复权价跳变会制造虚假 forward_return；qfq 不应。"""
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        dates = [f"2024-01-0{i}" for i in range(2, 9)]  # 6 天
        # 不复权：第 4 天除权，open 从 10 跳到 7（虚假 -30%）
        raw_open = [10.0, 10.1, 10.0, 7.0, 7.1, 7.2]
        # 前复权：连续，无跳变
        qfq_open = raw_open[:3] + [10.0, 10.1, 10.2]
        raw = [("600000", d, o, o, o, o, 1e6, 1e7)
               for d, o in zip(dates, raw_open)]
        qfq = [("600000", d, o, o, o, o, 1e6, 1e7)
               for d, o in zip(dates, qfq_open)]
        _seed_daily(conn, raw)
        _seed_qfq(conn, qfq)

        raw_p = F.load_panels(conn, adjust="")
        qfq_p = F.load_panels(conn, adjust="qfq")
        fwd_raw = F.forward_returns(raw_p, horizons=(2,))[2]
        fwd_qfq = F.forward_returns(qfq_p, horizons=(2,))[2]

        # 除权日(raw)的 T+1→T+2 open 收益出现 ~-30% 虚假跳变
        raw_vals = fwd_raw["600000"].dropna().values
        # qfq 连续价下，同等窗口收益应平缓（无 -30% 级别跳变）
        qfq_vals = fwd_qfq["600000"].dropna().values
        assert abs(raw_vals).max() > 0.25
        assert abs(qfq_vals).max() < 0.05
        conn.close()


class TestQualityMaskPointInTime:
    def test_quality_true_only_after_disclosure(self):
        """quality_mask 返回 bool DataFrame：披露前 False，披露后 True。"""
        conn = sqlite3.connect(":memory:")
        init_schema(conn)
        _seed_stock_list(conn, ["600000", "600001"])
        # spot 表提供 circ_mv（quality_panel 用当前值做市值门槛）
        conn.executemany(
            "INSERT OR REPLACE INTO spot (code,circ_mv,total_mv,pe_dyn,pb) "
            "VALUES (?,?,?,?,?)",
            [("600000", 5e9, 5e9, 20.0, 2.0),
             ("600001", 5e9, 5e9, 20.0, 2.0)])
        conn.commit()
        dates = [f"2024-0{i}-01" for i in (1, 2, 3, 4)]  # 1/1,2/1,3/1,4/1
        for c in ("600000", "600001"):
            _seed_daily(conn, [(c, d, 10.0, 10.0, 10.0, 10.0, 1e6, 1e7)
                               for d in dates])
        # 600000 财报 as_of=2024-03-01 才披露（ROE 合格）；此前质量应为 False
        _seed_fin(conn, [("600000", "20240331", "2024-03-01", 12.0, 50.0)])
        # 600001 无财报 -> 始终 False（缺失不算违规，但 point-in-time 下披露前也无）

        q = F.quality_mask(conn)
        assert isinstance(q, pd.DataFrame)
        # as_of 之前（1月、2月）600000 应为 False
        assert not bool(q.loc[pd.Timestamp("2024-01-01"), "600000"])
        assert not bool(q.loc[pd.Timestamp("2024-02-01"), "600000"])
        # 披露后（3月、4月）应为 True
        assert bool(q.loc[pd.Timestamp("2024-03-01"), "600000"])
        assert bool(q.loc[pd.Timestamp("2024-04-01"), "600000"])
        conn.close()


class TestRankIcTimeVaryingQuality:
    def test_dataframe_quality_filters_by_day(self):
        """rank_ic_series 用 bool DataFrame 质量集时，逐日只保留当日合格股。

        构造 60 只股票 × 4 天：前 30 只(组X)与 fwd 强正相关，后 30 只(组Y)
        强负相关；质量集第 0/1 天仅组X合格、第 2/3 天仅组Y合格。逐日正确
        过滤时，第 0/1 天 IC 应≈正(组X)，第 2/3 天 IC 应≈负(组Y)。
        """
        rng = np.random.default_rng(0)
        n = 120
        codes_x = [f"6{i:05d}" for i in range(n // 2)]
        codes_y = [f"0{i:05d}" for i in range(n // 2)]
        codes = codes_x + codes_y
        idx = pd.date_range("2024-01-01", periods=4, freq="D")

        base_x = pd.DataFrame(rng.normal(0, 1, (4, n // 2)),
                              index=idx, columns=codes_x)
        base_y = pd.DataFrame(rng.normal(0, 1, (4, n // 2)),
                              index=idx, columns=codes_y)
        fac = pd.concat([base_x, base_y], axis=1)
        # fwd：组X 与 factor 同号（正相关），组Y 与 factor 异号（负相关）
        fwd = pd.concat([base_x, -base_y], axis=1)

        qual = pd.DataFrame(False, index=idx, columns=codes)
        qual.loc[idx[0], codes_x] = True
        qual.loc[idx[1], codes_x] = True
        qual.loc[idx[2], codes_y] = True
        qual.loc[idx[3], codes_y] = True

        s = rank_ic_series(fac, fwd, qual)
        # 4 天都 >=50 样本，应产出 4 个 IC
        assert len(s) == 4
        # 第 0/1 天仅组X(正相关) -> IC 应为正；第 2/3 天仅组Y(负相关) -> IC 应为负
        assert s.iloc[0] > 0.5 and s.iloc[1] > 0.5
        assert s.iloc[2] < -0.5 and s.iloc[3] < -0.5

    def test_set_quality_still_supported(self):
        """回归：quality 为 set 时仍按静态集合过滤（行为不变）。"""
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        fac = pd.DataFrame({"600000": [0.1, 0.2, 0.3, 0.4]}, index=idx)
        fwd = pd.DataFrame({"600000": [0.05, 0.05, 0.05, 0.05]}, index=idx)
        s = rank_ic_series(fac, fwd, {"600000"})
        assert isinstance(s, pd.Series)
