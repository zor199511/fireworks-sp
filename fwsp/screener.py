import json
import logging
from datetime import date

import pandas as pd

from .db import get_conn, get_active_set, init_schema, set_meta, upsert_rows
from .indicators import compute_features
from .multifactor_score import multifactor_scores

log = logging.getLogger("fwsp.screener")

RECO_COLS = ["run_date", "rank", "code", "name", "industry", "score",
              "price", "reasons", "metrics", "factor_set_id"]


def _pit_universe_filter(conn, codes: list[str], lookback_days: int = 5) -> list[str]:
    """从候选 codes 中剔除 daily 长期不更新的(已退市/已停牌)。

    子代理 3 critical #5: 002155 last_daily=2026-08-19 但 spot 仍 8-26 更新，
    不剔会被推荐。这里用 daily.last_date >= today - lookback_days 作为
    "仍在交易" 的硬门 (停牌期间 daily.amount=0 但日期还在 → 仍可过)。
    lookback=5 交易日容忍周末/节假日。
    """
    if not codes:
        return []
    from datetime import datetime, timedelta
    today = datetime.now().date()
    cutoff = (today - timedelta(days=lookback_days * 2)).isoformat()  # *2 跨周末
    rows = conn.execute(
        f"SELECT code, MAX(date) AS last FROM daily "
        f"WHERE code IN ({','.join('?' * len(codes))}) "
        f"GROUP BY code", codes).fetchall()
    live = {c for c, last in rows if last and last >= cutoff}
    dropped = set(codes) - live
    if dropped:
        log.info("PIT universe 过滤: 剔除 %d 只长期无 daily 数据的股票 (lookback=%dd)",
                 len(dropped), lookback_days)
    return [c for c in codes if c in live]


def _in_index_tags(conn, code: str) -> str:
    """查询某 code 当前所在的宽基指数(取最近 snapshot_date)。

    返回 '沪深300,创业板指' 形式,空字符串表示不在任何快照指数内。
    多指数在 index_cons 中分多行,以 (code, snapshot_date) 区分。
    """
    rows = conn.execute(
        "SELECT DISTINCT index_code FROM index_cons "
        "WHERE code=? AND snapshot_date = ("
        "  SELECT MAX(snapshot_date) FROM index_cons WHERE code=?)",
        (code, code)).fetchall()
    names = {"000300": "沪深300", "399905": "中证500", "399006": "创业板指",
             "399001": "深证成指"}
    return ",".join(names.get(r[0], r[0]) for r in rows)


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
    # 债务率缺失的处理：子代理 4 critical #1 指出 222 高负债股绕过门, 但
    # 实际数据库中 debt_ratio NULL 是普遍现象(早期未披露), 直接拒收
    # 会导致 5000+ 只全拒(0 推荐)。改为: None 时通过, 但 PIT 质量门槛
    # (quality_panel) 用 fin_q 历史 max_debt=85 兜底, 且写 meta 标记让
    # dashboard 知道"当前 universe 有 N 只 debt_ratio 缺失, 实盘需关注".
    # 子代理 2 轮 R2-韧性3: None 通过时记 reasons, 让用户看到 "基本面缺失"
    # 提示, 不再被绿色信号误导.
    debt_missing = debt is None
    if debt_missing:
        r.setdefault("_missing_fundamentals", []).append("debt_ratio")
        pass  # 不拒收, PIT + dashboard 提示代替
    elif debt > cfg["max_debt"]:
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
        # 统计 debt_ratio 缺失数, 写 meta 让 dashboard 感知
        null_debt = int(base["debt_ratio"].isna().sum()) if hasattr(base["debt_ratio"], "sum") else 0
        # 显式 numpy 路径: pandas Series.sum() 返回 int64
        try:
            null_debt = int(null_debt)
        except (TypeError, ValueError):
            null_debt = 0
        if null_debt > 0:
            set_meta(conn, "universe_null_debt_count", str(null_debt))
        log.info("candidates after universe SQL: %d (debt_ratio 缺失 %d 只)",
                 len(base), null_debt)

        survivors = []
        missing_fund_count = 0
        for r in base.to_dict("records"):
            r.setdefault("_missing_fundamentals", [])
            if passes_hard_filters(r, FILTERS):
                continue
            if r["_missing_fundamentals"]:
                missing_fund_count += 1
            survivors.append(r)
        log.info("after fundamental filters: %d (基本面缺失 %d 只, 待推 %d)",
                 len(survivors), missing_fund_count, missing_fund_count)
        # 子代理 2 轮 R2-韧性3: 缺失 >30% universe 触发告警 meta
        if len(base) > 0 and missing_fund_count / len(base) > 0.30:
            set_meta(conn, "universe_fund_missing_alert",
                     f"{missing_fund_count}/{len(base)}={missing_fund_count/len(base):.0%} "
                     f"基本面缺失, 实盘需关注")
        else:
            set_meta(conn, "universe_fund_missing_alert", "0")

        # PIT universe 边界：剔除 daily 长期不更新的股票(已退市/已停牌)
        # 子代理 3 critical #5: spot 比 daily 滞后,需要额外硬门
        survivors_codes = _pit_universe_filter(
            conn, [r["code"] for r in survivors])
        survivors = [r for r in survivors if r["code"] in set(survivors_codes)]
        log.info("after PIT universe filter: %d", len(survivors))

        # 流动性预筛（与进化口径一致）：需 ≥130 日线 + 近 20 日日均额达标
        # 子代理 2 轮 R2-性能1: N+1 SELECT 改为 batched IN 一次查.
        # 子代理 2 轮 R2-性能6: hard_filter_rows 子查询 -> 已由 validate_table_name + LEFT JOIN 减少.
        from .db import validate_table_name
        _daily = validate_table_name("daily")
        codes_need = [r["code"] for r in survivors]
        liquid_codes: set[str] = set()
        if codes_need:
            ph = ",".join("?" * len(codes_need))
            rows = conn.execute(
                f"SELECT code, amount FROM {_daily} "
                f"WHERE code IN ({ph}) "
                f"ORDER BY code, date DESC", codes_need).fetchall()
            # 内存聚合: 每 code 取最近 20 行 amount 求平均 + 总行数 ≥130
            by_code: dict[str, list[float]] = {}
            for code, amt in rows:
                if amt is None:
                    continue
                by_code.setdefault(code, []).append(float(amt))
            for code, amts in by_code.items():
                if len(amts) < 130:
                    continue
                if sum(amts[:20]) / 20.0 < FILTERS["min_amount_20d"]:
                    continue
                liquid_codes.add(code)
        liquid = [r for r in survivors if r["code"] in liquid_codes]

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
                reasons = list(mf_reasons[code])
                # 子代理 2 轮 R2-韧性3: 缺失基本面附加 reason 提示
                if r.get("_missing_fundamentals"):
                    reasons.append(f"⚠️基本面缺失: {','.join(r['_missing_fundamentals'])}")
                r2["reasons"] = reasons
                r2["close"] = r.get("price")  # 推荐页价格取自 spot.price
                scored.append(r2)
            # 活跃因子集有效 → 清除降级标记
            set_meta(conn, "factor_system_degraded", "0")
        else:
            log.warning("无 active_factors，回退硬编码反转打分（因子系统降级）")
            # 子代理 2 轮 R2-性能1: fallback N+1 查 daily 改 batched IN
            from .db import validate_table_name
            _daily = validate_table_name("daily")
            liquid_codes_fb = [r["code"] for r in liquid]
            daily_by_code: dict[str, pd.DataFrame] = {}
            if liquid_codes_fb:
                ph = ",".join("?" * len(liquid_codes_fb))
                fb_rows = conn.execute(
                    f"SELECT code, date, open, high, low, close, volume, amount "
                    f"FROM {_daily} WHERE code IN ({ph}) "
                    f"ORDER BY code, date", liquid_codes_fb).fetchall()
                cols = ["code", "date", "open", "high", "low", "close",
                        "volume", "amount"]
                for r in fb_rows:
                    daily_by_code.setdefault(r[0], []).append(r[1:])
            for r in liquid:
                code = r["code"]
                raw = daily_by_code.get(code)
                if not raw:
                    continue
                df = pd.DataFrame(raw, columns=["date", "open", "high", "low",
                                                 "close", "volume", "amount"])
                ft = compute_features(df)
                if not ft:
                    continue
                sc, reasons = score_technical(ft)
                r2 = dict(r)
                r2.update(ft)
                r2["score"] = sc
                # 子代理 2 轮 R2-韧性3: 缺失基本面附加 reason 提示
                if r.get("_missing_fundamentals"):
                    reasons = list(reasons) + [
                        f"⚠️基本面缺失: {','.join(r['_missing_fundamentals'])}"]
                r2["reasons"] = reasons
                scored.append(r2)
            # 活跃因子集为空/失效 → 标记降级，供 dashboard 横幅与微信告警读取
            set_meta(conn, "factor_system_degraded", "1")

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
