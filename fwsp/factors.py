import logging
import sqlite3

import numpy as np
import pandas as pd

from .db import get_conn

log = logging.getLogger("fwsp.factors")

HORIZONS = (5, 10, 20)


def load_panels(conn, adjust: str = "qfq"):
    """加载价格面板。adjust='qfq'(默认) 用前复权价(连续、除权日不跳变)；
    adjust='' 用不复权原始价。daily_qfq 表缺失/为空时回退 daily，保证向前兼容。
    """
    table = "daily_qfq" if adjust == "qfq" else "daily"
    try:
        rows = conn.execute(
            f"SELECT code,date,open,high,low,close,volume,amount FROM {table} "
            "ORDER BY date").fetchall()
    except sqlite3.OperationalError:
        rows = []
    if not rows and adjust == "qfq":
        rows = conn.execute(
            "SELECT code,date,open,high,low,close,volume,amount FROM daily "
            "ORDER BY date").fetchall()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low",
                                     "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    panels = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        p = df.pivot(index="date", columns="code", values=col)
        panels[col] = p
    return panels


def _safe_div(a, b):
    return a.divide(b.replace(0, np.nan))


def build_factor_panel(conn):
    """价格/成交量因子（全程历史）+ 当前基本面质量门槛（静态，非历史重放）。

    返回 dict: factors(name->DataFrame TxN), close, open, low, volume, amount,
               quality(set of kept codes, 常数)。
    """
    panels = load_panels(conn)
    c, o, h, l, v, amt = (panels["close"], panels["open"], panels["high"],
                          panels["low"], panels["volume"], panels["amount"])
    ret = c.pct_change()

    f = {}

    for n in (5, 10, 20, 60, 120):
        f[f"ret_{n}d"] = c.pct_change(n)
    f["rev_5d"] = -c.pct_change(5)
    f["rev_20d"] = -c.pct_change(20)

    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60, 120)}
    f["close_ma20"] = _safe_div(c, ma[20]) - 1
    f["close_ma60"] = _safe_div(c, ma[60]) - 1
    f["close_ma120"] = _safe_div(c, ma[120]) - 1
    f["ma5_ma20"] = _safe_div(ma[5], ma[20]) - 1
    f["ma20_ma60"] = _safe_div(ma[20], ma[60]) - 1

    f["vol_20d"] = ret.rolling(20).std()
    f["vol_60d"] = ret.rolling(60).std()
    f["downside_20d"] = ret.clip(upper=0).rolling(20).std()

    vol_ma5 = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    amt_ma20 = amt.rolling(20).mean()
    f["vol_ratio"] = _safe_div(vol_ma5, vol_ma20)
    f["amt_ratio"] = _safe_div(amt.rolling(5).mean(), amt_ma20)
    f["log_amt20"] = np.log(amt_ma20.clip(lower=1))

    high20 = h.rolling(20).max()
    high60 = h.rolling(60).max()
    high120 = h.rolling(120).max()
    f["hi20_dist"] = _safe_div(c, high20) - 1
    f["hi60_dist"] = _safe_div(c, high60) - 1
    f["hi120_dist"] = _safe_div(c, high120) - 1

    ema = lambda s, n: s.ewm(span=n, adjust=False, min_periods=n).mean()
    dif = ema(c, 12) - ema(c, 26)
    dea = dif.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (dif - dea) * 2
    f["macd_dif"] = dif
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain.divide(loss.replace(0, np.nan))
    f["rsi_14"] = 100 - 100 / (1 + rs)

    f["chg"] = ret * 100
    f["gap_up"] = _safe_div(o, c.shift(1)) - 1
    f["amihud"] = _safe_div(ret.abs(), amt.clip(lower=1))

    quality = quality_mask(conn)

    return {"factors": f, "close": c, "open": o, "high": h, "low": l,
            "volume": v, "amount": amt, "quality": quality, "ret": ret}


def quality_mask(conn, min_circ_mv=3e9, min_roe=0.0,
                 min_profit_yoy=None, max_debt_ratio=85.0,
                 exclude_st=True, dates=None):
    """Point-in-time 质量门槛，返回 DataFrame(bool, dates x codes)。

    基本面按财报 as_of 前向填充（point-in-time，消除『用今天财报评判历史』
    的幸存者偏差）；市值/ST 用当前值。dates 为空时回退全 daily 日期索引。

    返回 bool DataFrame 后，rank_ic_series / walk_forward_backtest 可逐日按
    当日质量集过滤（同 quality_panel 契约），不再恒为同一静态集合。
    """
    if dates is None:
        rows = conn.execute("SELECT DISTINCT date FROM daily ORDER BY date").fetchall()
        dates = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return quality_panel(conn, dates, min_circ_mv=min_circ_mv, min_roe=min_roe,
                          min_profit_yoy=min_profit_yoy,
                          max_debt_ratio=max_debt_ratio, exclude_st=exclude_st)


def forward_returns(panels, horizons=HORIZONS):
    """T+1 开盘买入、持有 horizon 日、T+horizon 开盘卖出之收益。

    与 factor_mining.forward_returns（auto_evolve 晋升标签）保持一致，
    消除「晋升用 open→open、实盘权重用 close→close」的标签错位。
    panels 需含 'open'（build_factor_panel 已返回）。
    """
    op = panels["open"]
    out = {}
    for hh in horizons:
        buy = op.shift(-1)        # T+1 开盘
        sell = op.shift(-hh)      # T+horizon 开盘
        out[hh] = sell / buy - 1.0
    return out


def code_first_last_dates(conn, table: str = "daily") -> pd.DataFrame:
    """每只股票在日线表的首/末交易日(用作 PIT universe 上市/退市边界)。

    上市日=该 code 在日线表首次出现的 date;退市日=该 code 在日线表最后一次
    出现的 date(若 collector 已隐式剔除退市股,通常 = 数据库最新交易日)。
    返回 DataFrame[code, first, last]。
    """
    rows = conn.execute(
        f"SELECT code, MIN(date) AS first, MAX(date) AS last "
        f"FROM {table} GROUP BY code").fetchall()
    out = pd.DataFrame(rows, columns=["code", "first", "last"])
    out["first"] = pd.to_datetime(out["first"], errors="coerce")
    out["last"] = pd.to_datetime(out["last"], errors="coerce")
    return out.dropna(subset=["first", "last"])


def _pit_boundaries(piv: pd.DataFrame, code_fl: pd.DataFrame) -> pd.DataFrame:
    """在 PIT 质量面板基础上叠加『上市前/退市后剔除』。

    code_fl: code_first_last_dates 返回的 DataFrame。date < code 首日 或
    date > code 末日 → False(不参与 IC/回测)。消除『新股未上市已纳入』与
    『已退市但 daily 仍占位(停牌股)』两类幸存者偏差。
    """
    if piv.empty or code_fl.empty:
        return piv
    cols = list(piv.columns)
    fl = code_fl.set_index("code")
    keep_codes = [c for c in cols if c in fl.index]
    if not keep_codes:
        return piv.iloc[:, :0]
    # 向量化边界:dates (884,) × keep_codes (n) 矩阵
    f_series = fl.loc[keep_codes, "first"]  # Index=keep_codes
    l_series = fl.loc[keep_codes, "last"]
    date_idx = piv.index
    # 直接用 numpy 数组做比较,避免 DataFrame 切片赋值的多义性
    f_arr = np.tile(f_series.values[None, :], (len(date_idx), 1))
    l_arr = np.tile(l_series.values[None, :], (len(date_idx), 1))
    d_arr = np.tile(date_idx.values[:, None], (1, len(keep_codes)))
    in_window = pd.DataFrame(
        (f_arr <= d_arr) & (l_arr >= d_arr),
        index=date_idx, columns=keep_codes)
    out = piv.copy()
    out.loc[:, keep_codes] = out.loc[:, keep_codes] & in_window.astype(bool)
    # 不在 keep_codes 的 code 直接 False(理论上 panels 不含)
    drop_codes = [c for c in cols if c not in keep_codes]
    if drop_codes:
        out.loc[:, drop_codes] = False
    return out


def quality_panel(conn, dates, min_circ_mv=3e9, min_roe=0.0,
                  min_profit_yoy=None, max_debt_ratio=85.0, exclude_st=True):
    """随时间变化的质量门槛，返回 DataFrame(bool, dates x codes)。

    基本面按报告 as_of 日期前向填充（报告发布前沿用上一期），市值/ST 用当前值。
    dates 应为 daily 的 DatetimeIndex；发布前的日期质量记为 False。
    """
    codes = pd.read_sql("SELECT code,is_st FROM stock_list", conn)
    spot = pd.read_sql("SELECT code,circ_mv FROM spot", conn)
    fin = pd.read_sql(
        "SELECT code,period,as_of,roe,profit_yoy,debt_ratio FROM fin_q", conn)
    st_codes = set(codes.loc[codes["is_st"].fillna(0) == 1, "code"])
    circ = spot.set_index("code")["circ_mv"].fillna(0)

    fin = fin.copy()
    # period 可能是 "20260331" 或 "2026-03-31"（旧/新库格式不同）
    fin["period"] = pd.to_datetime(
        fin["period"], format="mixed", errors="coerce")
    fin["as_of"] = pd.to_datetime(fin["as_of"], errors="coerce")
    # as_of 缺失时回退 period（报告期末+30d，财报披露日的保守近似）。
    # 旧库 fin_q.as_of 多为 NULL（backfill_fundamentals 未跑过），没有
    # 这个 fallback → 整张 PIT 面板全空 → 幸存者偏差修复形同虚设。
    missing = fin["as_of"].isna() & fin["period"].notna()
    fin.loc[missing, "as_of"] = fin.loc[missing, "period"] + pd.Timedelta(days=30)
    fin = fin.dropna(subset=["as_of"])
    qual = pd.Series(True, index=fin.index)
    qual &= ~fin["code"].isin(st_codes)
    qual &= fin["code"].map(circ).fillna(0) >= min_circ_mv
    has_roe = fin["roe"].notna()
    qual &= (~has_roe) | (fin["roe"] >= min_roe)
    if min_profit_yoy is not None:
        has_py = fin["profit_yoy"].notna()
        qual &= (~has_py) | (fin["profit_yoy"] >= min_profit_yoy)
    has_debt = fin["debt_ratio"].notna()
    qual &= (~has_debt) | (fin["debt_ratio"] <= max_debt_ratio)
    fin = fin.assign(qual=qual)

    if len(fin) == 0:
        return pd.DataFrame(False, index=dates, columns=[])
    # 同一 (as_of, code) 可能对应多个报告期，仅保留 period 最新的一行，
    # 避免 pivot 默认 mean 聚合把 0/1 质量标志平均成 0.111/0.5 等分数。
    fin = fin.loc[fin.groupby(["as_of", "code"])["period"].idxmax()]
    piv = (fin.pivot_table(index="as_of", columns="code", values="qual")
              .sort_index().ffill())
    piv = piv.reindex(dates).ffill()
    # 缺失 (as_of,code) 由下游 fillna(False) 处理；此处显式转纯 numpy bool，
    # 确保布尔契约且规避 Arrow 索引问题（无 0.111/0.5 等分数）。
    piv = piv.fillna(False).astype(bool)

    # PIT universe 边界：上市前/退市后剔除
    # 退市股通常 collector 已隐式剔除(last_date = 数据库最新日),此时此
    # 步骤为 no-op;但对于停牌股(daily 持续占位至末日)与新股(daily 从
    # 上市日起记录)能正确划定每只 code 在每日的 universe 资格,彻底消除
    # 『未上市已纳入』与『已退市但仍占位』两类幸存者偏差。
    # 子代理 5 critical #1: 原 try/except+log.warning 静默降级, PIT 失败
    # 但 IC 计算照跑 → 幸存者偏差修复形同虚设. 改为 ERROR + 写 meta 标记.
    try:
        code_fl = code_first_last_dates(conn)
        piv = _pit_boundaries(piv, code_fl)
    except Exception as e:  # noqa: BLE001
        log.error("PIT universe boundaries failed: %s", e)
        # 写 meta 让 dashboard / 告警能感知此降级
        try:
            from .db import set_meta
            set_meta(conn, "pit_universe_degraded", "1")
        except Exception:
            pass
    else:
        try:
            from .db import set_meta
            set_meta(conn, "pit_universe_degraded", "0")
        except Exception:
            pass
    return piv
