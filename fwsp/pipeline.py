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


def refetch_qfq(conn, source: str = "baostock", resume: bool = False):
    """一次性全量重抓前复权日线写 daily_qfq（修复不复权已知限制）。

    不复权 daily 表保持不变；source='baostock'（前复权，适配 akshare EM 不通的
    受限服务器，单会话串行约 1 小时）；source='akshare' 走东方财富（本机网络通，并发）。
    resume=True 仅补 daily_qfq 缺失的股票（幂等），用于修复全量重抓遗漏。
    因限流较久，建议在空闲时手动/定时触发（update_data.py --refetch-qfq）。
    失败静默跳过，不影响主流程。
    """
    from . import collector
    if resume:
        done, fail = collector.refetch_qfq_missing(conn, "universe",
                                                   source=source)
    else:
        codes = collector.universe_codes(conn)
        log.info("refetch qfq for %d codes (source=%s)", len(codes), source)
        done, fail = collector.refetch_qfq_all(conn, codes, HISTORY_START,
                                              source=source)
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
    from .lock import file_lock, LOCK_RECOMMEND
    with file_lock(LOCK_RECOMMEND, op="daily_run") as got:
        if not got:
            log.warning("daily_run 锁被其他进程持有, 本次跳过(可能 dashboard "
                        "在跑)")
            return [], {}, False
        # 子代理 2 轮 R2-运维4: 顶层 try/except + meta 错误记录 + 告警
        try:
            return _daily_run_inner(push)
        except Exception as e:
            import datetime
            import traceback
            err = traceback.format_exc()[:500]
            try:
                with db.get_conn() as _c:
                    _c.execute(
                        "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                        ("last_run_status", f"FAIL {datetime.datetime.now().isoformat()}"))
                    _c.execute(
                        "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                        ("last_run_error", err))
                    _c.commit()
            except Exception:
                pass
            # 推送告警(不与同日 daily_run 推送去重冲突: 用独立 task)
            try:
                sys_path = "/home/zor/.config/opencode/scripts/lib"
                if sys_path not in __import__("sys").path:
                    __import__("sys").path.insert(0, sys_path)
                from notify import send_wechat_daily
                send_wechat_daily("fireworks_cron_fail",
                                  "⚠️fireworks_sp daily_run 失败",
                                  f"```\n{err}\n```")
            except Exception:
                pass
            raise


def _daily_run_inner(push):
    """daily_run 的实际实现, 外面套 file_lock + try/except."""
    with db.get_conn() as conn:
        db.init_schema(conn)
        update_all(conn, full=False)

    # 子代理 2 轮 R2-韧性2: 原版在 run_screen 之后读 degraded, 但
    # screener 内部 set_meta 写入可能因 apply_industry_cap 异常未触发.
    # 改: run_screen 后重新从 active_factors 派生 degraded(单一真相源),
    # 再读 meta 兜底; 任何异常下 try/finally 刷新 meta.
    recos = screener.run_screen(top_n=FILTERS["top_n"])
    tracker.update_tracking()
    stats = tracker.summary_stats()

    degraded = False
    try:
        with db.get_conn() as _c:
            # 派生(单一真相): 活跃因子集空 / 全 net_ir<=0 视为降级
            aset = db.get_active_set(_c, "auto_evolve")
            if not aset or not aset.get("factors"):
                degraded = True
            else:
                ids = aset["factors"]
                from .db import validate_table_name
                ph = ",".join("?" * len(ids))
                rows = _c.execute(
                    f"SELECT code, net_ir FROM {validate_table_name('factor_eval')} "
                    f"WHERE (code, run_at) IN ("
                    f"  SELECT code, MAX(run_at) FROM {validate_table_name('factor_eval')} "
                    f"  WHERE code IN ({ph}) GROUP BY code)", ids).fetchall()
                if not rows or all((r[1] is None) for r in rows):
                    degraded = True
            # 同步 meta 兜底
            _c.execute(
                "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                ("factor_system_degraded", "1" if degraded else "0"))
            _c.commit()
    except Exception as e:  # noqa: BLE001
        log.warning("factor_system_degraded derive failed: %s", e)
        degraded = True

    sent = False
    if push and recos:
        push_task = "fireworks_sp"  # 避免 except 分支下未绑定
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
            # 子代理 2 轮 R2-运维5: 因子降级用独立 task 名, 避免同日被
            # send_wechat_daily 同 task 去重吞掉
            push_task = "fireworks_degraded" if degraded else "fireworks_sp"
            sent = send_wechat_daily(push_task, title, desp)
        except Exception as e:  # noqa: BLE001
            log.warning("push failed (ignored): %s", e)
        # 子代理 2 轮 R2-运维5: 记录推送结果到 meta, dashboard 可看
        import datetime as _dt
        try:
            with db.get_conn() as _c:
                _c.execute(
                    "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                    ("last_push_ok", "1" if sent else "0"))
                _c.execute(
                    "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                    ("last_push_at", _dt.datetime.now().isoformat(timespec="seconds")))
                _c.execute(
                    "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                    ("last_push_task", push_task))
                _c.commit()
        except Exception:  # noqa: BLE001
            pass
    if degraded:
        log.warning("因子系统降级：本轮推荐为技术兜底打分")
    return recos, stats, sent
