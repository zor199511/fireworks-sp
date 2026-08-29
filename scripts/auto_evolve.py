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
from fwsp.db import (get_conn, get_meta, init_schema, set_meta,
                     upsert_rows)
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
        ls = long_short_backtest(f, fwd)
        results[fid] = {
            "is_icir": ic["icir"], "oos_icir": wf["oos_icir"],
            "oos_is_ratio": oos_is_ratio(ic["icir"], wf["oos_icir"]),
            "ic_mean": ic["ic_mean"],
            "stability": stab.get("worst"),
            "turnover": to["turnover"],
            "net_ir": ls["net_ir"],
        }
        # 坑1 修复：净成本信息比率<=0 的因子不可交易，剔除（不写库/不入选）
        if isinstance(ls["net_ir"], float) and not np.isnan(ls["net_ir"]) \
                and ls["net_ir"] <= 0:
            dropped_cost.append(fid)

    selected = [s for s in selected if s not in dropped_cost]
    if dropped_cost:
        log.info("因净成本 IR<=0 剔除: %s", dropped_cost)
    new_oos = float(np.nanmean([results[f]["oos_icir"] for f in selected])) \
        if selected else float("nan")

    with get_conn() as conn:
        last = conn.execute(
            "SELECT new_oos FROM evolution_log ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
        old_active = get_meta(conn, "active_factors")
    old_oos = float(last[0]) if last and last[0] is not None else None
    old_list = json.loads(old_active) if old_active else []

    promote, reason = oos_guard(new_oos, old_oos)
    if not selected:
        promote, reason = False, "无入选因子"

    out = {
        "run_at": run_at, "n_candidates": len(specs), "selected": selected,
        "metrics": results, "new_oos": new_oos, "old_oos": old_oos,
        "old_active": old_list, "promoted": (not dry_run) and promote,
        "reason": reason, "dry_run": dry_run,
    }

    if not dry_run and promote:
        with get_conn() as conn:
            init_schema(conn)
            lib_rows = []
            for fid in selected:
                spec = next(s for s in specs if s.id == fid)
                lib_rows.append((spec.id, spec.category, spec.expr,
                                json.dumps(spec.params, ensure_ascii=False),
                                spec.desc, "auto_evolve", run_at))
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
            set_meta(conn, "active_factors",
                     json.dumps(selected, ensure_ascii=False))
        log.info("已写入 %d 个因子并晋升", len(selected))

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_evolution(dry_run=True))
