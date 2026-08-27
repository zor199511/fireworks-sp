import logging

import numpy as np
import pandas as pd

from .db import get_conn

log = logging.getLogger("fwsp.factors")

HORIZONS = (5, 10, 20)


def load_panels(conn):
    rows = conn.execute(
        "SELECT code,date,open,high,low,close,volume,amount FROM daily "
        "ORDER BY date").fetchall()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low",
                                     "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    panels = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        p = df.pivot(index="date", columns="code", values=col).sort_index()
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
    c, o, h, v, amt = (panels["close"], panels["open"], panels["high"],
                       panels["volume"], panels["amount"])
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

    return {"factors": f, "close": c, "open": o, "low": h,
            "volume": v, "amount": amt, "quality": quality, "ret": ret}


def quality_mask(conn, min_circ_mv=3e9, min_roe=0.0,
                 min_profit_yoy=None, max_debt_ratio=85.0,
                 exclude_st=True):
    """当前静态质量门槛，返回 set(keep codes)。

    注意：fin_q 仅含最新两季，无历史序列，故此门槛不可代表历史财务状态，
    仅作‘当下’质量过滤，回测中恒为同一集合（局限已在注释标明）。
    """
    codes = pd.read_sql(
        "SELECT code,name,industry,is_st FROM stock_list", conn)
    spot = pd.read_sql(
        "SELECT code,circ_mv,total_mv,pe_dyn,pb FROM spot", conn)
    fin = pd.read_sql(
        "SELECT code,roe,profit_yoy,gross_margin,debt_ratio,period "
        "FROM fin_q", conn)
    fin = fin.sort_values("period").groupby("code").tail(1)
    df = codes.merge(spot, on="code", how="left").merge(fin, on="code", how="left")
    mask = pd.Series(True, index=df.index)
    if exclude_st:
        mask &= (df["is_st"].fillna(0) == 0)
    mask &= (df["circ_mv"].fillna(0) >= min_circ_mv)
    # 基本面缺失不算违规（fin_q 仅含当前期，多数股无历史序列）
    has_roe = df["roe"].notna()
    mask &= (~has_roe) | (df["roe"] >= min_roe)
    if min_profit_yoy is not None:
        has_py = df["profit_yoy"].notna()
        mask &= (~has_py) | (df["profit_yoy"] >= min_profit_yoy)
    has_debt = df["debt_ratio"].notna()
    mask &= (~has_debt) | (df["debt_ratio"] <= max_debt_ratio)
    return set(df.loc[mask.to_numpy(), "code"])


def forward_returns(close, horizons=HORIZONS):
    out = {}
    for hh in horizons:
        out[hh] = close.pct_change(hh).shift(-hh)
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
        "SELECT code,period,as_of,roe,profit_yoy,debt_ratio FROM fin_q "
        "WHERE as_of IS NOT NULL", conn)
    st_codes = set(codes.loc[codes["is_st"].fillna(0) == 1, "code"])
    circ = spot.set_index("code")["circ_mv"].fillna(0)

    fin = fin.copy()
    fin["as_of"] = pd.to_datetime(fin["as_of"], errors="coerce")
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
    return piv.fillna(False).astype(bool)
