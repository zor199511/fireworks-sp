import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fwsp.db import get_conn
from fwsp import factors as F
from fwsp import multifactor as M
from fwsp.backtest import run_backtest

logging.basicConfig(level=logging.WARNING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--is-start", default="2024-06-01")
    ap.add_argument("--holdout", default="2026-01-01")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-factors", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    with get_conn() as conn:
        blob = F.build_factor_panel(conn)
        fwd = F.forward_returns(blob)
        quality = blob["quality"]
        qpanel = F.quality_panel(conn, blob["close"].index)
    print(f"[build] {len(blob['factors'])} 因子, 静态质量池 {len(quality)} 只, "
          f"时间质量面板 {qpanel.shape}, {time.time()-t0:.1f}s")

    t0 = time.time()
    z, ic = M.precompute(blob["factors"], fwd, quality, args.horizon)
    print(f"[precompute] {time.time()-t0:.1f}s")

    # 1) 在 IS 窗口贪心挖掘（质量面板随时间变化）
    t0 = time.time()
    selected, report = M.mine_factors(
        blob["factors"], blob["close"], blob["open"], blob["low"], fwd,
        quality, qpanel=qpanel, highs=blob["high"], horizon=args.horizon,
        start=args.is_start, top_n=args.top_n, max_factors=args.max_factors)
    print(f"[mine] 选中 {len(selected)} 因子, {time.time()-t0:.1f}s")
    print("  逐步加入:", ", ".join(report["added"].tolist()))
    print(report.to_string(index=False))

    def show(label, r):
        print(f"  {label:24s} 总收益 {r['total_return']*100:7.1f}%  "
              f"夏普 {r['sharpe']:.2f}  回撤 {r['max_drawdown']*100:6.1f}%  "
              f"胜率 {r['win_rate']*100:4.1f}%  笔数 {int(r['n_trades'])}")

    print("\n=== 对比（IS 窗口 %s→%s）===" % (args.is_start, args.holdout))
    r_is = M.walk_forward_backtest(
        z, ic, blob["close"], blob["open"], blob["low"], qpanel, highs=blob["high"],
        start=args.is_start, end=args.holdout, top_n=args.top_n,
        horizon=args.horizon, selected=selected)
    show("多因子(选中, IS)", r_is)
    show("多因子(全因子, IS)", M.walk_forward_backtest(
        z, ic, blob["close"], blob["open"], blob["low"], qpanel, highs=blob["high"],
        start=args.is_start, end=args.holdout, top_n=args.top_n,
        horizon=args.horizon))
    show("原反转策略(IS)", run_backtest(
        start=args.is_start, end=args.holdout, top_n=args.top_n,
        hold_days=args.horizon, stop_pct=-8.0, strategy="reversal"))

    print("\n=== 冻结 holdout（%s→今，真样本外）===" % args.holdout)
    r_oos = M.walk_forward_backtest(
        z, ic, blob["close"], blob["open"], blob["low"], qpanel, highs=blob["high"],
        start=args.holdout, top_n=args.top_n, horizon=args.horizon,
        selected=selected)
    show("多因子(选中, OOS)", r_oos)
    show("多因子(全因子, OOS)", M.walk_forward_backtest(
        z, ic, blob["close"], blob["open"], blob["low"], qpanel, highs=blob["high"],
        start=args.holdout, top_n=args.top_n, horizon=args.horizon))
    show("原反转策略(OOS)", run_backtest(
        start=args.holdout, top_n=args.top_n, hold_days=args.horizon,
        stop_pct=-8.0, strategy="reversal"))

    print(f"\n选中因子: {selected}")

    # 固化到 meta，供 dashboard 实时推荐使用
    from fwsp.db import get_conn as _gc
    with _gc() as conn:
        M.save_selected(conn, selected, summary={
            "selected": selected, "is_start": args.is_start,
            "holdout": args.holdout, "horizon": args.horizon,
            "top_n": args.top_n,
            "oos_total_return": round(r_oos["total_return"], 4),
            "oos_sharpe": round(r_oos["sharpe"], 3),
            "oos_max_drawdown": round(r_oos["max_drawdown"], 4),
            "oos_win_rate": round(r_oos["win_rate"], 3),
            "is_total_return": round(r_is["total_return"], 4),
            "is_sharpe": round(r_is["sharpe"], 3),
        })
    print("[saved] 选中因子与 OOS 摘要已写入 meta")


if __name__ == "__main__":
    main()
