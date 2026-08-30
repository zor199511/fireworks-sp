"""进化主程序骨架：自动挖掘因子 → walk-forward OOS 评估 → 过拟合防护 → 晋升。

默认 dry_run=True 只打印 selected 因子与 OOS_ICIR，不写库。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fwsp.backtest import load_panels
from fwsp.config import FILTERS
from fwsp.db import (get_conn, get_active_set, init_schema,
                      set_active_factors, upsert_rows)
from fwsp.factor_factory import expand_recipes
from fwsp.factor_mining import (daily_ic_series, factor_turnover,
                                forward_returns, greedy_select, ic_analysis,
                                long_short_backtest, oos_guard, screen_factors,
                                walk_forward)
from fwsp.overfit_guard import oos_is_ratio, stability_check
from fwsp.community_watch import merge_all_recipes

log = logging.getLogger("fwsp.auto_evolve")

GRID = {"windows": [5, 10, 20, 60, 120], "windows2": [20, 60, 120]}


def _liquid_universe(panels: dict, min_amt: float = 50e6):
    """ restricting to liquid stocks: 近 20 日日均成交额 >= min_amt。
    同时降低内存与噪声，与 screener 硬过滤口径一致。"""
    amt = panels["amount"]
    keep = amt.tail(20).mean()
    keep = keep[keep >= min_amt].index
    log.info("流动性筛选: %d/%d 只留入挖掘池", len(keep),
             panels["close"].shape[1])
    return {k: v[keep] for k, v in panels.items()}, keep


def auto_demote(conn, stability_thr: float = 0.0, net_ir_thr: float = 0.0) -> dict:
    """因子衰减自动降级：检查 active_factors 中每个因子的最新评估，
    稳定性(worst rolling ICIR) < thr 或 净成本 IR <= thr 则移出 active 集。
    返回 {"demoted":[(code,reason)...], "kept":[...]}。
    """
    aset = get_active_set(conn, "auto_evolve")
    if not aset:
        return {"demoted": [], "kept": [], "reason": "无 active_sets"}
    ids = aset["factors"]
    if not ids:
        return {"demoted": [], "kept": [], "reason": "active_sets 为空"}

    ph = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT code, stability, net_ir FROM factor_eval "
        f"WHERE (code,run_at) IN (SELECT code, MAX(run_at) FROM factor_eval "
        f"WHERE code IN ({ph}) GROUP BY code)", ids).fetchall()
    ev_map = {r[0]: r for r in rows}

    demoted: list[tuple] = []
    kept: list[str] = []
    for fid in ids:
        ev = ev_map.get(fid)
        if not ev:
            kept.append(fid)
            continue
        stab, nir = ev[1], ev[2]
        reasons = []
        if stab is not None and not (isinstance(stab, float) and np.isnan(stab)) \
                and stab < stability_thr:
            reasons.append(f"稳定性{stab:.2f}<{stability_thr}")
        if not (isinstance(nir, float) and not np.isnan(nir)):
            reasons.append("净IR非数值(NaN)")
        elif nir <= net_ir_thr:
            reasons.append(f"净IR{nir:.2f}<={net_ir_thr}")
        if reasons:
            demoted.append((fid, "; ".join(reasons)))
        else:
            kept.append(fid)

    if demoted:
        set_active_factors(conn, kept, run_at=aset["run_at"],
                          oos=aset["oos"], source=aset["source"])
        log.info("因子衰减降级: 移出 %s -> 保留 %s",
                 [d[0] for d in demoted], kept)
    return {"demoted": demoted, "kept": kept}


def _factor_drop_reason(res: dict) -> str | None:
    """更严格晋升门禁：返回剔除原因，None 表示可入选。

    - 净成本 IR<=0 或 NaN：不可交易 / 数值异常（坑1 修复）
    - 过拟合：OOS 信息比率 > 5× IS（overfit_guard.ratio_alert 同口径）
    - 失稳：滚动 worst ICIR < 0（近期表现转负）
    """
    nir = res.get("net_ir")
    if not (isinstance(nir, float) and not np.isnan(nir) and nir > 0):
        return "净成本IR<=0/NaN(不可交易)"
    oos = res.get("oos_is_ratio")
    if isinstance(oos, float) and not np.isnan(oos) and oos > 5.0:
        return f"过拟合 OOS/IS>{oos:.1f} (>5)"
    stab = res.get("stability")
    if isinstance(stab, float) and not np.isnan(stab) and stab < 0.0:
        return f"失稳最差滚动ICIR={stab:.2f}(<0)"
    return None


def run_evolution(dry_run: bool = False) -> dict:
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_conn() as conn:
            init_schema(conn)
            panels = load_panels(conn)

        panels, _ = _liquid_universe(panels, FILTERS["min_amount_20d"])
        fwd = forward_returns(panels, horizon=10)
        base = merge_all_recipes()  # base.yaml + community.yaml（社区因子）
        specs = expand_recipes(base, GRID)
        log.info("候选因子数: %d", len(specs))

        # 流式预筛：只保留达标因子，内存占用可控
        screened = screen_factors(panels, specs, fwd, icir_gate=0.5, n_min=50)
        log.info("ICIR 预筛通过: %d", len(screened))

        selected = greedy_select(screened, fwd, icir_gate=0.5,
                                 corr_thr=0.7, top=20)
    except Exception as e:
        log.exception("evolution failed: %s", e)
        return {"run_at": run_at, "error": str(e),
                "selected": [], "metrics": {}, "promoted": False,
                "dry_run": dry_run}

    results: dict[str, dict] = {}
    dropped_cost = []
    for fid in selected:
        f = screened[fid]
        ic = ic_analysis(f, fwd)
        wf = walk_forward(f, fwd)
        ic_series = daily_ic_series(f, fwd)
        stab = stability_check(ic_series, window=252)
        to = factor_turnover(f)
        ls = long_short_backtest(f, fwd, rebal="W")  # 实盘周频调仓口径
        results[fid] = {
            "is_icir": ic["icir"], "oos_icir": wf["oos_icir"],
            "oos_is_ratio": oos_is_ratio(ic["icir"], wf["oos_icir"]),
            "ic_mean": ic["ic_mean"],
            "stability": stab.get("worst"),
            "turnover": to["turnover"],
            "net_ir": ls["net_ir"],
        }
        # 更严格晋升门禁：净成本 IR 不可交易 / 过拟合(OOS 是 IS 的 >5 倍) / 失稳
        # （滚动 worst ICIR<0）一律剔除，不写库、不入选。
        dr = _factor_drop_reason(results[fid])
        if dr:
            dropped_cost.append(fid)

    selected = [s for s in selected if s not in dropped_cost]
    if dropped_cost:
        log.info("晋升门禁剔除(%d): %s", len(dropped_cost), dropped_cost)
    new_oos = float(np.nanmean([results[f]["oos_icir"] for f in selected])) \
        if selected else float("nan")

    with get_conn() as conn:
        last = conn.execute(
            "SELECT new_oos FROM evolution_log ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
        old_set = get_active_set(conn, "auto_evolve")
    old_oos = float(last[0]) if last and last[0] is not None else None
    old_list = old_set["factors"] if old_set else []

    promote, reason = oos_guard(new_oos, old_oos)
    if not selected:
        promote, reason = False, "无入选因子"

    out = {
        "run_at": run_at, "n_candidates": len(specs), "selected": selected,
        "metrics": results, "new_oos": new_oos, "old_oos": old_oos,
        "old_active": old_list, "promoted": (not dry_run) and promote,
        "reason": reason, "dry_run": dry_run, "demoted": [],
    }

    if not dry_run and promote:
        with get_conn() as conn:
            init_schema(conn)
            lib_rows = []
            for fid in selected:
                spec = next(s for s in specs if s.id == fid)
                lib_rows.append((spec.id, spec.category, spec.expr,
                                json.dumps(spec.params, ensure_ascii=False),
                                spec.desc, spec.source or "auto_evolve", run_at))
            upsert_rows(conn, "factor_library",
                        ["code", "category", "expr", "params_json", "desc",
                         "source", "created_at"], lib_rows)

            ev_rows = []
            for fid in selected:
                m = results[fid]
                ev_rows.append((fid, run_at, m["is_icir"], m["oos_icir"],
                                m["oos_is_ratio"],
                                float(m["stability"]) if m["stability"] is not None
                                else float("nan"),
                                float(m["turnover"]) if m["turnover"] is not None
                                else float("nan"),
                                float(m["net_ir"]) if m["net_ir"] is not None
                                else float("nan"), 1))
            upsert_rows(conn, "factor_eval",
                        ["code", "run_at", "is_icir", "oos_icir",
                         "oos_is_ratio", "stability", "turnover", "net_ir",
                         "selected"], ev_rows)

            conn.execute(
                "INSERT OR REPLACE INTO evolution_log "
                "(run_at, selected_json, old_oos, new_oos, promoted, notes) "
                "VALUES (?,?,?,?,?,?)",
                (run_at, json.dumps(selected, ensure_ascii=False), old_oos,
                 new_oos, 1, reason))
            # 唯一写入者：写 active_sets + 镜像 meta + 同步 factor_eval.selected
            set_active_factors(conn, selected, run_at=run_at, oos=new_oos,
                              source="auto_evolve")
            # 注意：晋升当轮不再顺手跑 auto_demote —— 新晋因子刚通过 net_ir>0
            # 与 oos_guard，其滚动 ICIR 的「worst」可能因某段历史为负而被误杀，
            # 导致刚写入的活跃集被当轮清空。衰减检查交由独立定时任务（--demote）负责。
        log.info("已写入 %d 个因子并晋升", len(selected))

    return out


def _wechat(task: str, title: str, desp: str) -> None:
    """微信推送结果（失败仅日志，不影响主流程）。"""
    try:
        from notify import send_wechat_daily
        send_wechat_daily(task, title, desp)
    except Exception as e:  # noqa: BLE001
        log.warning("微信推送失败(忽略): %s", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="自动进化多因子筛选")
    parser.add_argument("--promote", action="store_true",
                        help="非 dry-run：达到 OOS 门控则写入因子库并晋升 active_factors")
    parser.add_argument("--demote", action="store_true",
                        help="仅执行因子衰减自动降级（清理 active_factors 中失效因子）")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.demote:
        with get_conn() as conn:
            init_schema(conn)
            dem = auto_demote(conn)
        print(json.dumps({"demoted": dem["demoted"], "kept": dem["kept"]},
                         ensure_ascii=False, indent=2, default=str))
        dcodes = ", ".join(d[0] for d in dem["demoted"]) or "无"
        _wechat(
            "fireworks_sp_demote",
            "🧬 fireworks 因子衰减降级",
            f"移出 {len(dem['demoted'])} 个：{dcodes}\n保留 {len(dem['kept'])} 个",
        )
    else:
        result = run_evolution(dry_run=not args.promote)
        print(json.dumps({k: v for k, v in result.items()
                          if k != "metrics"}, ensure_ascii=False, indent=2,
                          default=str))
        if args.promote:
            if result.get("promoted"):
                sel = result.get("selected") or []
                desp = (f"晋升 {len(sel)} 个因子\nOOS={result.get('new_oos')}\n"
                        f"因子: {', '.join(sel)}\n门控: {result.get('reason')}")
                _wechat("fireworks_sp_evolve", "🧬 fireworks 因子进化晋升", desp)
            else:
                desp = (f"未达晋升门控，未写入。\nreason: {result.get('reason')}\n"
                        f"候选 OOS={result.get('new_oos')}")
                _wechat("fireworks_sp_evolve", "🧬 fireworks 因子进化未晋升", desp)
