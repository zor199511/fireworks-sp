import json
import logging
from datetime import date

import pandas as pd

from .config import DB_PATH
from .db import get_conn, get_active_set, init_schema, set_meta, upsert_rows
from .indicators import compute_features
from .multifactor_score import multifactor_scores

log = logging.getLogger("fwsp.screener")

RECO_COLS = ["run_date", "rank", "code", "name", "industry", "score",
              "price", "reasons", "metrics", "factor_set_id"]


# ------------------------------------------------------------ hard filters

def hard_filter_rows(conn) -> pd.DataFrame:
    q = """
    SELECT p.code, s.name, s.industry,
           p.price, p.pe_dyn, p.pb, p.total_mv, p.circ_mv, p.turnover,
           f.roe, f.debt_ratio, f.gross_margin, f.profit_yoy
    FROM spot p
    JOIN stock_list s ON s.code = p.code
    LEFT JOIN fin_q f ON f.code = p.code
      AND f.period = (SELECT MAX(period) FROM fin_q WHERE code = p.code)
    WHERE s.exchange IN ('sh','sz') AND s.is_st = 0
      AND p.price IS NOT NULL AND p.price > 0
    ORDER BY p.code
    """
    df = pd.read_sql(q, conn)
    if "industry" in df.columns:
        df["industry"] = df["industry"].where(~df["industry"].isna(), None)
    return df


def passes_hard_filters(r: dict, cfg: dict) -> list[str]:
    fails = []
    mv = r.get("total_mv") or 0
    if mv < cfg["min_total_mv"]:
        fails.append(f"市值{mv/1e8:.0f}亿<{cfg['min_total_mv']/1e8:.0f}亿")
    pe = r.get("pe_dyn")
    if pe is None:
        fails.append("PE缺失")
    elif pe <= 0 or pe > cfg["max_pe"]:
        fails.append(f"PE{pe:.1f}不在(0,{cfg['max_pe']}]")
    roe = r.get("roe")
    if roe is None:
        fails.append("ROE缺失")
    elif roe < cfg["min_roe"]:
        fails.append(f"ROE{roe:.1f}<{cfg['min_roe']}%")
    debt = r.get("debt_ratio")
    if debt is not None and debt > cfg["max_debt"]:
        fails.append(f"负债率{debt:.1f}>{cfg['max_debt']}%")
    return fails


# ----------------------------------------------------------- tech scoring

def score_technical(ft: dict) -> tuple[float, list[str]]:
    """反转策略打分：深度回调的优质股出现企稳信号（回测验证优于动量）。"""
    score = 0.0
    reasons = []
    c = ft["close"]
    ma5, ma120, dif, dea = ft["ma5"], ft["ma120"], ft["dif"], ft["dea"]
    rsi_v, ret20, ret60 = ft["rsi"], ft.get("ret_20d"), ft.get("ret_60d")
    chg = ft.get("chg_pct") or 0

    if ret20 is not None and -35 <= ret20 <= -12:
        score += 30
        reasons.append(f"20日回调{ret20:.0f}%")
    if rsi_v is not None and rsi_v < 38:
        score += 20
        reasons.append(f"RSI超卖回升({rsi_v:.0f})")
    if ma5 and c > ma5 and chg > 0:
        score += 20
        reasons.append(f"止跌企稳(+{chg:.1f}%)")
    if ma120 and c > ma120:
        score += 15
        reasons.append("长期趋势未破")
    if dif is not None and dea is not None and dif > dea and dif < 0:
        score += 10
        reasons.append("底部MACD金叉")
    if ret60 is not None and -50 <= ret60 <= -20:
        score += 5

    return max(0.0, min(score, 100.0)), reasons


def apply_industry_cap(scored: list[dict], top_n: int, cap: int) -> list[dict]:
    if top_n <= 0 or not scored:
        return []
    # cap<=0 表示不限制
    if cap <= 0:
        return scored[:top_n]
    picked = []
    counts = {}
    for r in scored:
        ind = r.get("industry")
        if isinstance(ind, str) and ind.strip():
            if counts.get(ind, 0) >= cap:
                continue
            counts[ind] = counts.get(ind, 0) + 1
        picked.append(r)
        if len(picked) >= top_n:
            break
    return picked


def run_screen(top_n=10, persist: bool = True) -> list[dict]:
    from .config import FILTERS
    with get_conn() as conn:
        init_schema(conn)
        base = hard_filter_rows(conn)
        log.info("candidates after universe SQL: %d", len(base))

        survivors = []
        for r in base.to_dict("records"):
            if passes_hard_filters(r, FILTERS):
                continue
            survivors.append(r)
        log.info("after fundamental filters: %d", len(survivors))

        # 流动性预筛（与进化口径一致）：需 ≥130 日线 + 近 20 日日均额达标
        liquid = []
        for r in survivors:
            rows = conn.execute(
                "SELECT date,open,high,low,close,volume,amount FROM daily "
                "WHERE code=? ORDER BY date", (r["code"],)).fetchall()
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                             "close", "volume", "amount"])
            if len(df) < 130:
                continue
            amt20 = df["amount"].astype(float).tail(20).mean()
            if amt20 < FILTERS["min_amount_20d"]:
                continue
            liquid.append(r)

        scored = []
        mf_scores, mf_reasons = multifactor_scores(
            conn, [r["code"] for r in liquid])
        if mf_scores:
            log.info("多因子合成模式: 候选 %d", len(mf_scores))
            for r in liquid:
                code = r["code"]
                if code not in mf_scores:
                    continue
                r2 = dict(r)
                r2["score"] = mf_scores[code]
                r2["reasons"] = mf_reasons[code]
                r2["close"] = r.get("price")  # 推荐页价格取自 spot.price
                scored.append(r2)
        else:
            log.info("无 active_factors，回退硬编码反转打分")
            for r in liquid:
                rows = conn.execute(
                    "SELECT date,open,high,low,close,volume,amount FROM daily "
                    "WHERE code=? ORDER BY date", (r["code"],)).fetchall()
                df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                                 "close", "volume", "amount"])
                ft = compute_features(df)
                if not ft:
                    continue
                sc, reasons = score_technical(ft)
                r2 = dict(r)
                r2.update(ft)
                r2["score"] = sc
                r2["reasons"] = reasons
                scored.append(r2)

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = apply_industry_cap(scored, top_n, FILTERS["max_per_industry"])
        for i, r in enumerate(top, 1):
            r["rank"] = i
        log.info("final scored pool: %d; top %d selected",
                 len(scored), len(top))
        if persist:
            _persist(conn, top)
        return top


def _persist(conn, top: list[dict]):
    run_date = str(date.today())
    # 血缘：记录本次推荐所用的活跃因子集（来源 auto_evolve）
    aset = get_active_set(conn, "auto_evolve")
    factor_set_id = aset["run_at"] if aset else None
    conn.execute("DELETE FROM recommendations WHERE run_date=?", (run_date,))
    rows = []
    for i, r in enumerate(top, 1):
        metrics = {k: r.get(k) for k in
                   ("pe_dyn", "pb", "total_mv", "roe", "debt_ratio", "rsi",
                    "dif", "dea", "chg_pct", "ret_20d")}
        r["rank"] = i
        r["metrics"] = metrics
        rows.append((run_date, i, r["code"], r.get("name"),
                     r.get("industry"), r["score"], r["close"],
                     json.dumps(r["reasons"], ensure_ascii=False),
                     json.dumps(metrics, ensure_ascii=False), factor_set_id))
    upsert_rows(conn, RECO_TABLE, RECO_COLS, rows)
    set_meta(conn, "last_recommend_run", run_date)


RECO_TABLE = "recommendations"


def ensure_reco_schema():
    with get_conn() as conn:
        init_schema(conn)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            run_date TEXT, rank INTEGER, code TEXT,
            name TEXT, industry TEXT,
            score REAL, price REAL,
            reasons TEXT, metrics TEXT,
            ret_5d REAL, ret_10d REAL, ret_20d REAL, ret_60d REAL,
            PRIMARY KEY (run_date, code)
        )""")
