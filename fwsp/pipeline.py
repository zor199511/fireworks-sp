import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from . import collector, db, screener, tracker
from .config import FILTERS, HISTORY_START

log = logging.getLogger("fwsp.pipeline")


def update_all(conn, full=False, skip_daily=False, source="auto"):
    known = [r[0] for r in conn.execute("SELECT code FROM stock_list")]
    info = collector.fetch_spot(conn, known_codes=known)
    log.info("spot updated via %s: %d quotes", info["source"], info["quotes"])

    periods = collector.recent_quarter_ends(collector.today_cn())
    fin = collector.fetch_fin_quarterly(conn, periods)
    log.info("financial quarters stored: %s", fin or "none")

    codes = collector.universe_codes(conn)
    log.info("universe size: %d", len(codes))

    if not skip_daily:
        n_rows_total = conn.execute(
            "SELECT COUNT(*) FROM daily").fetchone()[0]
        if full or n_rows_total < len(codes) * 100:
            log.info("bootstrap full daily history (%d codes, source=%s)",
                     len(codes), source)
            if source == "baostock":
                done, fail = collector.bootstrap_baostock(
                    conn, codes, HISTORY_START)
                log.info("baostock bootstrap done: ok=%d fail=%d",
                         done, fail)
            else:
                done, fail = collector.update_all_dailies_bootstrap(
                    conn, codes, HISTORY_START)
                log.info("bootstrap done: ok=%d em_fail=%d", done, fail)
        else:
            n_new = collector.append_today_bars_from_snapshot(
                conn, info["todays_bars"])
            healed = collector.backfill_gaps(conn, codes)
            log.info("incremental bars appended=%d healed=%d",
                     n_new, healed)

    n_idx = collector.fetch_index_baostock(conn, "sh.000300", HISTORY_START)
    if n_idx:
        log.info("benchmark rows upserted: %d", n_idx)

    db.set_meta(conn, "last_update",
                datetime.now(ZoneInfo("Asia/Shanghai"))
                .isoformat(timespec="seconds"))
    return info


def refetch_qfq(conn, source="auto"):
    """一次性全量重抓前复权日线写 daily_qfq（修复不复权已知限制）。

    不复权 daily 表保持不变；因 akshare 限流约 15–20 分钟，建议在空闲时手动/
    定时触发（update_data.py --refetch-qfq）。失败静默跳过，不影响主流程。
    """
    from . import collector
    codes = collector.universe_codes(conn)
    log.info("refetch qfq for %d codes", len(codes))
    done, fail = collector.refetch_qfq_all(conn, codes, HISTORY_START)
    log.info("qfq refetch done: ok=%d fail=%d", done, fail)
    db.set_meta(conn, "last_qfq_refetch",
                datetime.now(ZoneInfo("Asia/Shanghai"))
                .isoformat(timespec="seconds"))
    return done, fail


def build_push_message(recos, stats) -> tuple[str, str]:
    d = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%m-%d")
    title = f"🎆 fireworks-sp 精选 {d}（{len(recos)}只）"
    lines = [f"## Top {len(recos)} 候选\n"]
    for i, r in enumerate(recos, 1):
        mv_yi = (r["metrics"].get("total_mv") or 0) / 1e8
        reason = "; ".join(r["reasons"][:2])
        lines.append(
            f"{i}. **{r['name']}**({r['code']}) {r['score']:.0f}分 "
            f"¥{r['price']:.2f}\n"
            f"   {reason}｜PE{r['metrics'].get('pe_dyn') or '—'} "
            f"ROE{r['metrics'].get('roe') or '—'}% {mv_yi:.0f}亿\n")
    wr5 = stats.get("win_rate_5d")
    a5 = stats.get("avg_ret_5d")
    lines.append(f"\n📊 追踪：累计推荐 {stats['total_recommendations']} 笔"
                 + (f"，5日胜率 {wr5*100:.0f}%，平均 {a5:+.2f}%"
                    if wr5 is not None else "（样本积累中）"))
    lines.append("\n> ⚠️ 候选池仅供参考，非投资建议；买入须独立判断并止损。")
    return title, "\n".join(lines)


def daily_run(push=True):
    """Full daily pipeline: update -> screen -> track -> push."""
    with db.get_conn() as conn:
        db.init_schema(conn)
        update_all(conn, full=False)

    recos = screener.run_screen(top_n=FILTERS["top_n"])
    tracker.update_tracking()
    stats = tracker.summary_stats()

    # 因子系统降级标记：活跃因子集为空/失效时，推荐为技术兜底打分
    degraded = False
    try:
        with db.get_conn() as _c:
            degraded = db.get_meta(_c, "factor_system_degraded") == "1"
    except Exception:
        degraded = False

    sent = False
    if push and recos:
        try:
            sys_path = "/home/zor/.config/opencode/scripts/lib"
            if sys_path not in __import__("sys").path:
                __import__("sys").path.insert(0, sys_path)
            from notify import send_wechat_daily
            title, desp = build_push_message(recos, stats)
            if degraded:
                title = "⚠️[因子降级] " + title
                desp = ("> ⚠️ **活跃因子集为空/失效，本轮推荐为技术兜底打分，"
                        "仅供参考，非因子系统产出。**\n\n" + desp)
            sent = send_wechat_daily("fireworks_sp", title, desp)
        except Exception as e:  # noqa: BLE001
            log.warning("push failed (ignored): %s", e)
    if degraded:
        log.warning("因子系统降级：本轮推荐为技术兜底打分")
    return recos, stats, sent
