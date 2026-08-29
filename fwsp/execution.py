"""统一的持仓执行层：成本、止盈、止损、滑点、仓位口径的唯一实现。

`backtest.run_backtest`(技术策略) 与 `multifactor.walk_forward_backtest`(因子模型)
都调用本模块，确保两套回测对「买卖成本 / 止损 / 止盈 / 滑点」的处理完全一致，
不再各自实现导致口径漂移。
"""
import numpy as np
import pandas as pd

from .costs import COST_BUY, COST_SELL

__all__ = [
    "compute_shares", "position_return", "sell_value",
    "decide_exit", "mark_equity",
]


def compute_shares(budget: float, px: float, cash: float) -> int:
    """按预算与可用现金计算可买手数（100 股/手，含买入成本）。"""
    if budget <= 0 or pd.isna(px) or px <= 0:
        return 0
    shares = int(budget / (px * (1 + COST_BUY)) / 100) * 100
    if shares <= 0:
        return 0
    if shares * px * (1 + COST_BUY) > cash:
        return 0
    return shares


def sell_value(shares: int, px: float) -> float:
    """卖出到账金额（扣卖出成本）。"""
    return shares * px * (1 - COST_SELL)


def position_return(sell_px: float, buy_px: float) -> float:
    """单笔持仓收益率（含买卖成本）。"""
    return (sell_px * (1 - COST_SELL)) / (buy_px * (1 + COST_BUY)) - 1


def mark_equity(positions: dict, closes: pd.DataFrame, d, cash: float) -> float:
    """按当日收盘价对当前持仓做市值标记。"""
    eq = cash
    for code, pos in positions.items():
        px = closes.at[d, code] if code in closes.columns else None
        eq += pos["shares"] * (px if pd.notna(px) else pos["buy_px"])
    return eq


def decide_exit(pos: dict, px_open, lo, hi, stop_pct: float,
                profit_pct, trail, held: int, horizon: int,
                stuck_after: int | None = None):
    """统一的当日离场决策。返回 (sell_px, reason)；不触发则 (None, None)。

    - 止损：开盘破位→以开盘价止损；盘中触及→以 min(开盘, 止损价) 止损。
    - 跟踪止盈：盘中低点触及峰值回撤阈值→离场。
    - 目标止盈：盘中高点触及目标价→以目标价离场（用日内 high，更贴近实盘）。
    - 到期：持有达 horizon-1 日→开盘离场（与 forward_returns 的
      open[t+1]→open[t+horizon] 持仓期严格对齐，消除 off-by-one）。
    - 被困兜底：仅当开盘价缺失（数据缺口）且持有达 stuck_after 时，
      以买入价强平——正常行情下到期先触发，此分支为数据缺失安全网。
    """
    stop_px = pos["buy_px"] * (1 + stop_pct / 100)
    peak = pos.get("peak", pos["buy_px"])
    expire_h = horizon - 1 if horizon > 1 else horizon
    if held >= 1 and pd.notna(px_open):
        if px_open <= stop_px:
            return px_open, "stop"
        if pd.notna(lo) and lo <= stop_px:
            return min(px_open, stop_px), "stop"
        if trail and trail > 0 and pd.notna(lo) \
                and lo <= peak * (1 - trail / 100):
            return min(px_open, peak * (1 - trail / 100)), "trail"
        if profit_pct and profit_pct > 0 and pd.notna(hi):
            tp = pos["buy_px"] * (1 + profit_pct / 100)
            if hi >= tp:
                return tp, "tp"
        if held >= expire_h:
            return px_open, "expire"
    elif stuck_after and held >= stuck_after:
        return pos["buy_px"], "stuck"
    return None, None


def long_short_backtest(factor: pd.DataFrame, fwd: pd.DataFrame,
                       top_frac: float = 0.1, cost_buy: float = COST_BUY,
                       cost_sell: float = COST_SELL) -> dict:
    """成本敏感多空组合（晋升路径 net_ir 的量度）。

    与 run_backtest / walk_forward_backtest 共用同一组 COST_BUY/COST_SELL 常量，
    统一落到 execution 层。每日按因子截面排名多前 top_frac、空后 top_frac，
    次日开盘调仓。日频调仓下多空双腿各 100% 换手，故每日总交易成本按
    2×(cost_buy+cost_sell) 计（多腿 + 空腿各一轮买卖），比单边模型更严格、
    更贴近实盘——直接抬高「净成本 IR」门禁，过滤不可交易因子。
    """
    r = factor.rank(axis=1, pct=True)
    long_sig = r >= (1 - top_frac)
    short_sig = r <= top_frac
    long_ret = fwd.where(long_sig).mean(axis=1)
    short_ret = fwd.where(short_sig).mean(axis=1)
    daily_cost = 2 * (cost_buy + cost_sell)
    port = (long_ret - short_ret).dropna() - daily_cost
    port = port.dropna()
    if len(port) < 30 or port.std() == 0:
        return {"net_ir": float("nan"), "net_ret": float("nan")}
    return {"net_ir": float(port.mean() / port.std() * np.sqrt(252)),
            "net_ret": float(port.mean() * 252)}

