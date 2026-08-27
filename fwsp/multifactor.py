import logging
from datetime import date

import numpy as np
import pandas as pd

log = logging.getLogger("fwsp.multifactor")

COST_BUY = 0.00025 + 0.001
COST_SELL = 0.00025 + 0.0005 + 0.001
HORIZONS = (5, 10, 20)


def _zcross(x):
    m = x.mean(axis=1)
    s = x.std(ddof=0, axis=1).replace(0, np.nan)
    return x.sub(m, axis=0).div(s, axis=0)


def rank_ic_series(factor, fwd, quality):
    """逐日期 RankIC（spearman）序列。手动 ranking 避免依赖 scipy。"""
    common = factor.index.intersection(fwd.index)
    ics, idx = [], []
    fac = factor.reindex(common)
    fw = fwd.reindex(common)
    for d in common:
        fv = fac.loc[d]
        rv = fw.loc[d]
        if quality:
            fv = fv.drop(labels=fv.index.difference(list(quality)))
            rv = rv.reindex(fv.index)
        pair = pd.concat([fv, rv], axis=1).dropna()
        if len(pair) < 50:
            continue
        ic = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
        if pd.notna(ic):
            ics.append(ic)
            idx.append(d)
    return pd.Series(ics, index=idx)


def analyze_factors(factors, fwd, quality, horizons=HORIZONS):
    rows = []
    for name, fac in factors.items():
        for hh in horizons:
            s = rank_ic_series(fac, fwd[hh], quality)
            if len(s) < 30:
                continue
            ic = s.mean()
            icir = ic / s.std() if s.std() > 0 else 0
            t = ic / (s.std() / np.sqrt(len(s))) if s.std() > 0 else 0
            rows.append({"factor": name, "horizon": hh, "n": len(s),
                         "IC": ic, "ICIR": icir, "t": t,
                         "IC_win": float((s > 0).mean())})
    df = pd.DataFrame(rows)
    return df.sort_values(["horizon", "IC"], ascending=[True, False]).reset_index(drop=True)


def precompute(factors, fwd, quality, horizon=10):
    """预计算：每因子 z-score 面板(shift1, 无前视) + 全期 RankIC 序列。"""
    z = {k: _zcross(v).shift(1) for k, v in factors.items()}
    ic = {k: rank_ic_series(v, fwd[horizon], quality) for k, v in factors.items()}
    return z, ic


def _train_weights(ics, quality, horizon, train_dates):
    w = {}
    for name, s in ics.items():
        seg = s.reindex(train_dates).dropna()
        if len(seg) < 30:
            continue
        ic = seg.mean()
        if abs(ic) < 0.01 or seg.std() == 0:
            continue
        w[name] = np.sign(ic) * abs(ic)
    return w


def walk_forward_backtest(zscores, ics, close, opn, low, quality,
                          start="2024-06-01", end=None, top_n=10,
                          horizon=10, stop_pct=-8.0, train_days=252,
                          capital=1_000_000.0, selected=None,
                          rebal='W', trail=None):
    end = end or str(date.today())
    zsel = zscores if selected is None else {k: v for k, v in zscores.items() if k in selected}

    cal = close.index[(close.index >= pd.Timestamp(start)) &
                      (close.index <= pd.Timestamp(end))]
    if len(cal) == 0:
        raise RuntimeError("no trading days")

    cash = capital
    positions = {}
    equity_curve = []
    trades = []

    for i in range(len(cal)):
        d = cal[i]
        eq = cash
        for code, pos in positions.items():
            px = close.at[d, code] if code in close.columns else np.nan
            eq += pos["shares"] * (px if pd.notna(px) else pos["buy_px"])
        equity_curve.append((d, eq))

        for code in list(positions):
            pos = positions[code]
            held = i - pos["entry_i"]
            px_open = opn.at[d, code] if code in opn.columns else np.nan
            lo = low.at[d, code] if code in low.columns else np.nan
            stop_px = pos["buy_px"] * (1 + stop_pct / 100)
            peak_v = pos.get("peak", pos["buy_px"])
            sell_px = None
            if held >= 1 and pd.notna(px_open) and pd.notna(lo):
                if lo <= stop_px:
                    sell_px = min(px_open, stop_px)
                    reason = "stop"
                elif trail is not None and trail > 0 and \
                        lo <= peak_v * (1 - trail / 100):
                    sell_px = min(px_open, peak_v * (1 - trail / 100))
                    reason = "trail"
                elif held >= horizon:
                    sell_px = px_open
                    reason = "expire"
            if sell_px:
                cash += pos["shares"] * sell_px * (1 - COST_SELL)
                trades.append({"code": code,
                               "ret": (sell_px * (1 - COST_SELL)) /
                                      (pos["buy_px"] * (1 + COST_BUY)) - 1,
                               "reason": reason})
                del positions[code]

        for code in positions:
            cpx = close.at[d, code] if code in close.columns else np.nan
            if pd.notna(cpx):
                pos = positions[code]
                pos["peak"] = max(pos.get("peak", pos["buy_px"]), cpx)

        if rebal == 'M':
            rk = (d.year, d.month)
            prev_rk = (cal[i - 1].year, cal[i - 1].month) if i >= 1 else None
        else:
            rk = d.isocalendar()[:2]
            prev_rk = cal[i - 1].isocalendar()[:2] if i >= 1 else None
        is_rebal = (i == 0) or (rk != prev_rk)
        if is_rebal and i >= 1:
            sig_day = cal[i - 1]
            ti = close.index.get_indexer([sig_day])[0]
            train_dates = close.index[max(0, ti - train_days):ti + 1]
            qset = (set(quality.loc[train_dates[-1]].astype(bool)
                        .index[quality.loc[train_dates[-1]].astype(bool).to_numpy()])
                    if isinstance(quality, pd.DataFrame) else quality)
            w = _train_weights(ics, qset, horizon, train_dates)
            if not w:
                continue
            wtot = sum(abs(x) for x in w.values()) or 1
            w = {k: v / wtot for k, v in w.items()}
            score = None
            for name, wi in w.items():
                if name not in zsel:
                    continue
                s = zsel[name].reindex(index=[sig_day]).iloc[0]
                if isinstance(quality, pd.DataFrame):
                    keep = (quality.loc[sig_day].reindex(s.index)
                            .fillna(False).astype(bool).to_numpy())
                    s = s[keep].dropna()
                elif quality:
                    s = s.drop(labels=s.index.difference(list(quality)))
                score = s * wi if score is None else score.add(s * wi, fill_value=0)
            if score is None:
                continue
            picks = score.dropna().sort_values(ascending=False).head(top_n)
            n_slots = top_n - len(positions)
            budget = eq / top_n
            for code, _ in picks.items():
                if n_slots <= 0 or code in positions:
                    continue
                px = opn.at[d, code]
                if pd.isna(px) or px <= 0:
                    continue
                shares = int(budget / (px * (1 + COST_BUY)) / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * px * (1 + COST_BUY)
                if cost > cash:
                    continue
                cash -= cost
                positions[code] = {"shares": shares, "buy_px": px,
                                   "entry_i": i, "peak": px}
                n_slots -= 1

    last_d = cal[-1]
    for code, pos in list(positions.items()):
        px = close.at[last_d, code]
        if pd.notna(px):
            cash += pos["shares"] * px * (1 - COST_SELL)
            trades.append({"code": code,
                           "ret": (px * (1 - COST_SELL)) /
                                  (pos["buy_px"] * (1 + COST_BUY)) - 1,
                           "reason": "final"})
        del positions[code]

    eq_series = pd.Series({d: e for d, e in equity_curve}).sort_index()
    ret_total = eq_series.iloc[-1] / capital - 1
    years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    cagr = (1 + ret_total) ** (1 / years) - 1 if years > 0 else 0
    dd = ((eq_series - eq_series.cummax()) / eq_series.cummax()).min()
    dr = eq_series.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    wins = [t for t in trades if t["ret"] > 0]
    return {"total_return": ret_total, "cagr": cagr, "max_drawdown": dd,
            "sharpe": sharpe, "n_trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "equity": eq_series}


def mine_factors(factors, close, opn, low, fwd, quality, qpanel=None,
                 horizon=10, start="2024-06-01", top_n=10, max_factors=12):
    """前向贪心因子选择：每步加入使 walk-forward OOS 夏普最高的因子。

    quality: 静态质量集（用于 IC 计算）；qpanel: 时间变化质量面板（用于回测）。
    返回 (selected_list, report_df)。
    """
    z, ic = precompute(factors, fwd, quality, horizon)
    backtest_quality = qpanel if qpanel is not None else quality
    candidates = list(factors.keys())
    selected = []
    cur_best = -1e9
    report = []
    while len(selected) < max_factors:
        best_sharpe, best_name = -1e9, None
        for name in candidates:
            if name in selected:
                continue
            r = walk_forward_backtest(z, ic, close, opn, low, backtest_quality,
                                      start=start, top_n=top_n, horizon=horizon,
                                      selected=selected + [name])
            if r["sharpe"] > best_sharpe:
                best_sharpe, best_name = r["sharpe"], name
        if best_name is None or best_sharpe <= cur_best:
            break
        selected.append(best_name)
        cur_best = best_sharpe
        report.append({"n": len(selected), "added": best_name,
                       "oos_sharpe": best_sharpe})
    return selected, pd.DataFrame(report)


def save_selected(conn, factors, summary=None):
    import json
    from .db import set_meta
    set_meta(conn, "multifactor_selected", json.dumps(factors))
    if summary is not None:
        set_meta(conn, "multifactor_summary", json.dumps(summary))


def load_selected(conn):
    import json
    from .db import get_meta
    raw = get_meta(conn, "multifactor_selected")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def live_recommend(z, ic, close, opn, low, quality, horizon=10, top_n=10,
                   train_days=252, as_of=None, selected=None):
    """用 walk-forward 训练窗在最新信号日生成实时推荐。

    返回 [{code, score, reasons}]。
    """
    zsel = z if selected is None else {k: v for k, v in z.items() if k in selected}
    ti = close.index.get_indexer([pd.Timestamp(as_of)])[0] if as_of else len(close) - 1
    if ti < 0:
        ti = len(close) - 1
    sig_day = close.index[ti]
    train_dates = close.index[max(0, ti - train_days):ti + 1]
    w = _train_weights(ic, quality, horizon, train_dates)
    if not w:
        return []
    wtot = sum(abs(x) for x in w.values()) or 1
    w = {k: v / wtot for k, v in w.items()}
    score = None
    for name, wi in w.items():
        if name not in zsel:
            continue
        s = zsel[name].reindex([sig_day]).iloc[0]
        if quality is not None and len(quality):
            if isinstance(quality, pd.DataFrame):
                keep = (quality.loc[sig_day].reindex(s.index)
                        .fillna(False).astype(bool).to_numpy())
                s = s[keep].dropna()
            else:
                s = s.drop(labels=s.index.difference(list(quality)))
        score = s * wi if score is None else score.add(s * wi, fill_value=0)
    if score is None:
        return []
    picks = score.dropna().sort_values(ascending=False).head(top_n)
    out = []
    for code in picks.index:
        reasons = []
        for name, wi in w.items():
            if name not in zsel:
                continue
            zv = zsel[name].reindex([sig_day]).iloc[0].get(code)
            if pd.notna(zv):
                reasons.append(f"{name}{zv * wi:+.2f}")
        out.append({"code": code, "score": float(picks[code]), "reasons": reasons})
    return out
