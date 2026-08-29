import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from .costs import COST_BUY
from .execution import (compute_shares, decide_exit, mark_equity,
                        position_return, sell_value)
from .db import get_conn

log = logging.getLogger("fwsp.backtest")


def load_panels(conn) -> dict[str, pd.DataFrame]:
    rows = conn.execute(
        "SELECT code,date,open,high,low,close,volume,amount FROM daily "
        "ORDER BY date").fetchall()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low",
                                     "close", "volume", "amount"])
    panels = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        p = df.pivot(index="date", columns="code", values=col)
        p.index = pd.to_datetime(p.index)
        panels[col] = p.sort_index()
    return panels


def _panel_indicators(panels):
    c, o = panels["close"], panels["open"]
    h, v, amt = panels["high"], panels["volume"], panels["amount"]

    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    vol_ma5 = v.rolling(5).mean()
    amt_ma20 = amt.rolling(20).mean()
    high20 = h.rolling(20).max()

    chg = c.pct_change() * 100
    ret_20d = c.pct_change(20) * 100
    ret_60d = c.pct_change(60) * 100
    ma120 = c.rolling(120).mean()
    return {"ma5": ma5, "ma20": ma20, "ma60": ma60, "ma120": ma120,
            "dif": dif, "dea": dea,
            "rsi": rsi, "vol_ma5": vol_ma5, "amt_ma20": amt_ma20,
            "high20": high20, "chg": chg,
            "ret_20d": ret_20d, "ret_60d": ret_60d}


def _scores_asof(ind, d, strategy="momentum") -> pd.Series:
    """Vectorized technical score for every stock using data up to date d."""
    base = ind["ma20"]
    idx = base.index.get_indexer([d], method="pad")[0]
    if idx < 130 or idx == -1:
        return pd.Series(dtype=float)

    def row(name):
        return ind[name].iloc[idx]

    closes = ind["closes"].iloc[idx]
    vols = ind["volumes"].iloc[idx]
    ma5, ma20, ma60 = row("ma5"), row("ma20"), row("ma60")
    ma120 = row("ma120")
    dif, dea, rsi_v = row("dif"), row("dea"), row("rsi")
    v5, a20 = row("vol_ma5"), row("amt_ma20")
    chg = ind["chg"].iloc[idx]
    ret20 = ind["ret_20d"].iloc[idx]
    ret60 = ind["ret_60d"].iloc[idx]

    score = pd.Series(0.0, index=closes.index)
    ok = closes.notna() & (a20 >= 50e6)
    if strategy == "momentum":
        m_trend = (ma5 > ma20) & (ma20 > ma60) & (closes > ma20)
        m_above60 = (~m_trend) & (closes > ma60)
        score += np.where(m_trend, 25.0, np.where(m_above60, 10.0, 0.0))
        m_macd = dif > dea
        score += np.where(m_macd, 15.0, 0.0)
        score += np.where(m_macd & (dif > 0), 5.0, 0.0)
        score += np.where(((chg > 1) & (vols > v5 * 1.2)).fillna(False),
                          15.0, 0.0)
        h20 = row("high20")
        score += np.where((closes >= h20 * 0.98).fillna(False), 15.0, 0.0)
        score += np.where(((rsi_v >= 50) & (rsi_v <= 70)).fillna(False),
                          10.0, 0.0)
        score -= np.where((rsi_v > 80).fillna(False), 10.0, 0.0)
    elif strategy == "reversal":
        # 深度回调 + 企稳信号：A股短期反转效应
        m_deep = (ret20 <= -12) & (ret20 >= -35)
        score += np.where(m_deep.fillna(False), 30.0, 0.0)
        score += np.where((rsi_v < 38).fillna(False), 20.0, 0.0)
        m_stab = (closes > ma5) & (chg > 0)     # 止跌企稳
        score += np.where(m_stab.fillna(False), 20.0, 0.0)
        m_vol_ok = closes > ma120                # 长期趋势未破坏
        score += np.where(m_vol_ok.fillna(False), 15.0, 0.0)
        m_macd_up = (dif > dea) & (dif < 0)      # 底部金叉
        score += np.where(m_macd_up.fillna(False), 10.0, 0.0)
        score += np.where(((ret60 < -20) & (ret60 > -50)).fillna(False),
                          5.0, 0.0)

    score[~ok.fillna(False)] = 0.0
    floor = 25.0 if strategy == "momentum" else 40.0
    score = score.clip(0, 100)
    return score[score > floor]


def run_backtest(start="2024-01-01", end=None, top_n=10, hold_days=5,
                 stop_pct=-8.0, profit_pct=8.0, capital=1_000_000.0,
                 strategy="reversal") -> dict:
    today = str(date.today())
    end = end or today
    with get_conn() as conn:
        panels = load_panels(conn)
        bench = conn.execute(
            "SELECT date,close FROM index_daily WHERE code='sh.000300' "
            "ORDER BY date").fetchall()

    closes = panels["close"]
    opens = panels["open"]
    lows = panels["low"]
    highs = panels["high"]
    ind = _panel_indicators(panels)
    ind["closes"] = closes
    ind["volumes"] = panels["volume"]

    cal = closes.index[(closes.index >= pd.Timestamp(start))
                       & (closes.index <= pd.Timestamp(end))]
    if len(cal) == 0:
        raise RuntimeError("no trading days in range")

    iso = cal.isocalendar()
    week_key = list(zip(iso.year, iso.week))
    rebal_idx = [i for i in range(len(cal))
                 if i == 0 or week_key[i] != week_key[i - 1]]

    cash = capital
    positions: dict[str, dict] = {}
    equity_curve = []
    trades = []

    for i in range(len(cal)):
        d = cal[i]

        # 市值标记（统一执行层）
        eq = mark_equity(positions, closes, d, cash)
        equity_curve.append((d, eq))

        # 离场决策（统一执行层）
        for code in list(positions):
            pos = positions[code]
            held = i - pos["entry_i"]
            px_open = opens.at[d, code] if code in opens.columns else None
            lo = lows.at[d, code] if code in lows.columns else None
            hi = highs.at[d, code] if code in highs.columns else None
            sell_px, reason = decide_exit(
                pos, px_open, lo, hi, stop_pct, profit_pct, None,
                held, hold_days, stuck_after=hold_days * 2)
            # 注：stuck_after 仅作数据缺口(开盘价缺失)安全网；正常行情下
            # 持有达 hold_days-1 日已由 expire 离场，见 execution.decide_exit
            if sell_px is not None:
                cash += sell_value(pos["shares"], sell_px)
                trades.append({"code": code, "entry_date": str(pos["entry_d"]),
                               "exit_date": str(d),
                               "ret": position_return(sell_px, pos["buy_px"]),
                               "reason": reason})
                del positions[code]

        # entries (rebalance days): signals from previous close, buy at open
        if i in rebal_idx and i >= 1:
            sig_day = cal[i - 1]
            sc = _scores_asof(ind, sig_day, strategy=strategy)
            picks = sc.sort_values(ascending=False).head(top_n)
            n_slots = top_n - len(positions)
            budget = eq / top_n if eq > 0 else 0
            for code, s in picks.items():
                if n_slots <= 0 or code in positions:
                    continue
                px = opens.at[d, code]
                shares = compute_shares(budget, px, cash)
                if shares <= 0:
                    continue
                cash -= shares * px * (1 + COST_BUY)
                positions[code] = {"shares": shares, "buy_px": px,
                                   "entry_i": i, "entry_d": d}
                n_slots -= 1

    # force-close remaining at last close
    last_d = cal[-1]
    for code, pos in list(positions.items()):
        px = closes.at[last_d, code]
        if pd.notna(px):
            cash += sell_value(pos["shares"], px)
            trades.append({"code": code, "entry_date": str(pos["entry_d"]),
                           "exit_date": str(last_d),
                           "ret": position_return(px, pos["buy_px"]),
                           "reason": "final"})
        del positions[code]

    eq_series = pd.Series({d: e for d, e in equity_curve}).sort_index()
    ret_total = eq_series.iloc[-1] / capital - 1
    years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    cagr = (1 + ret_total) ** (1 / years) - 1 if years > 0 else 0
    roll_max = eq_series.cummax()
    dd = ((eq_series - roll_max) / roll_max).min()
    daily_ret = eq_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0)

    bench_rows = [(d, c) for d, c in bench
                  if start <= d <= end]
    bench_ret = None
    if len(bench_rows) >= 2:
        bench_ret = bench_rows[-1][1] / bench_rows[0][1] - 1

    wins = [t for t in trades if t["ret"] > 0]
    result = {
        "start": start, "end": end,
        "total_return": ret_total, "cagr": cagr, "max_drawdown": dd,
        "sharpe": sharpe,
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "avg_trade_ret": float(np.mean([t["ret"] for t in trades]))
        if trades else 0,
        "bench_return": bench_ret,
        "equity": {str(d.date()): e for d, e in equity_curve},
        "trades": trades,
    }
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = run_backtest()
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("equity", "trades")},
                     ensure_ascii=False, indent=2))
