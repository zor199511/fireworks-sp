"""多因子合成：用进化选出的 active_factors 驱动每日推荐打分。

与 auto_evolve 共用因子 DSL（fwsp.factor_factory）。区别：
- auto_evolve 在全宇宙上做 IC/进化；这里在「已硬过滤的候选池」上，取最新交易日
  各因子值，做截面 zscore 后用因子权重（来自 factor_eval 的 net_ir / is_icir）加权
  合成综合分。截面算子(cs_*)因此天然在候选池内标准化。
- 无 active_factors 时回退到 screener.score_technical（向后兼容）。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .db import get_conn, get_active_set
from .factor_factory import FactorSpec, compute_factor_panel

log = logging.getLogger("fwsp.multifactor_score")


def load_panel_for_codes(conn, codes: list[str], adjust: str = "qfq") -> dict[str, pd.DataFrame]:
    """加载候选池股票面板。默认前复权(qfq,连续、除权日不跳变)；daily_qfq 缺失
    回退该股票的 daily，不复权原始价。

    与 factors.load_panels/backtest.load_panels 同源口径，保证「研究回测 /
    因子挖掘 / 每日推荐」三者吃同一套价格。
    """
    if not codes:
        return {}
    ph = ",".join("?" * len(codes))
    table = "daily_qfq" if adjust == "qfq" else "daily"
    df = pd.read_sql(
        f"SELECT code,date,open,high,low,close,volume,amount FROM {table} "
        f"WHERE code IN ({ph}) ORDER BY date", conn, params=codes)
    panels = {}
    if df.empty:
        # daily_qfq 缺失 -> 回退 daily（向前兼容）
        df = pd.read_sql(
            f"SELECT code,date,open,high,low,close,volume,amount FROM daily "
            f"WHERE code IN ({ph}) ORDER BY date", conn, params=codes)
    for col in ("open", "high", "low", "close", "volume", "amount"):
        p = df.pivot(index="date", columns="code", values=col)
        p.index = pd.to_datetime(p.index)
        panels[col] = p.sort_index()
    return panels


def active_specs(conn) -> tuple[list[FactorSpec], dict[str, float]]:
    """读取 active_sets(来源 auto_evolve) 与 factor_eval 权重。

    返回 (specs, weights)。weights 用 |net_ir|（缺失时回退 |is_icir|），
    全为 NaN/空时返回等权。无 active_factors 则返回 ([], {}) 由调用方回退。
    """
    aset = get_active_set(conn, "auto_evolve")
    raw = json.dumps(aset["factors"]) if aset else None
    if not raw:
        return [], {}
    ids = json.loads(raw)
    if not ids:
        return [], {}
    rows = conn.execute(
        "SELECT code, is_icir, net_ir FROM factor_eval "
        "WHERE (code,run_at) IN ("
        "  SELECT code, MAX(run_at) FROM factor_eval "
        "  WHERE code IN (%s) GROUP BY code)"
        % ",".join("?" * len(ids)), ids).fetchall()
    eval_map = {r[0]: r for r in rows}

    # 因子表达式来自 factor_library
    lib = conn.execute(
        "SELECT code, category, expr, params_json, desc FROM factor_library "
        "WHERE code IN (%s)" % ",".join("?" * len(ids)), ids).fetchall()
    lib_map = {r[0]: r for r in lib}

    specs = []
    weights = {}
    for fid in ids:
        if fid not in lib_map:
            continue
        cat, expr, pj, desc = lib_map[fid][1], lib_map[fid][2], \
            lib_map[fid][3], lib_map[fid][4]
        specs.append(FactorSpec(fid, cat, expr,
                                json.loads(pj) if pj else {}, desc or fid))
        ev = eval_map.get(fid)
        w = abs(ev[2]) if ev and ev[2] is not None and not (isinstance(ev[2], float)
                 and np.isnan(ev[2])) else (abs(ev[1]) if ev and ev[1] is not None
                 and not (isinstance(ev[1], float) and np.isnan(ev[1])) else 1.0)
        weights[fid] = w
    return specs, weights


def multifactor_scores(conn, codes: list[str], as_of: str | None = None
                       ) -> tuple[dict[str, float], dict[str, list]]:
    """对候选 codes 计算多因子综合分。返回 (score_map, reasons_map)。

    reasons_map[code] = [(factor_id, 贡献值), ...] 取贡献最大的前 3。
    无 active_factors / specs 为空时返回 ({}, {})，调用方应回退 score_technical。
    """
    specs, weights = active_specs(conn)
    if not specs:
        return {}, {}
    if sum(weights.values()) == 0:
        weights = {k: 1.0 for k in weights}

    panels = load_panel_for_codes(conn, codes)
    if not panels:
        return {}, {}
    fpanel = compute_factor_panel(panels, specs)

    # 取每个因子最新可用交易日的值（截面）
    latest = {}
    for s in specs:
        fp = fpanel[s.id]
        if as_of is not None:
            idx = fp.index.get_indexer([pd.Timestamp(as_of)], method="pad")
            row = fp.iloc[idx[0]] if idx[0] >= 0 else fp.iloc[-1]
        else:
            row = fp.iloc[-1]
        latest[s.id] = row

    mat = pd.DataFrame({s.id: latest[s.id] for s in specs})
    # 截面 zscore（候选池内标准化）；缺失用 0
    z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan),
                                             axis=0).fillna(0.0)

    score = {}
    reasons = {}
    for code in mat.index:
        contrib = {fid: weights[fid] * z.at[code, fid] for fid in weights}
        total = float(sum(contrib.values()))
        score[code] = total
        top = sorted(contrib.items(), key=lambda kv: abs(kv[1]),
                     reverse=True)[:3]
        reasons[code] = [f"{fid}({v:+.1f})" for fid, v in top]
    return score, reasons


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with get_conn() as conn:
        sc, rs = multifactor_scores(conn, [])
    print("specs active:", len(sc))
