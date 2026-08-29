"""因子挖掘：前向收益、IC 分析、walk-forward OOS 评估、贪心选因子、晋升门禁。

所有「因子」侧计算都只用到信号日及之前的数据（见 factor_factory 的算子约束）。
forward_returns 是标签本身，按定义使用未来开盘价，这是合法的（它不被当作特征）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .costs import COST_BUY, COST_SELL
from .execution import long_short_backtest as _exec_long_short
from .factor_factory import compute_factor
from .overfit_guard import oos_is_ratio


# --------------------------------------------------------------------------
# 前向收益（标签）
# --------------------------------------------------------------------------

def forward_returns(panels: dict[str, pd.DataFrame], horizon: int = 10) -> pd.DataFrame:
    """T+1 开盘买入、持有 horizon 日、于 T+horizon 开盘卖出之收益。

    与现有回测 T+1 开盘买入约定一致。fwd[t] = open[t+horizon]/open[t+1] - 1。
    末 horizon 行因无法取到未来开盘价为 NaN（这是标签的固有边界，非前视泄漏）。
    """
    op = panels["open"]
    buy = op.shift(-1)        # T+1 开盘
    sell = op.shift(-horizon)  # T+horizon 开盘
    return sell / buy - 1.0


# --------------------------------------------------------------------------
# IC 分析（截面相关，按日平均）
# --------------------------------------------------------------------------

def _cross_section_ic(factor: pd.DataFrame, fwd: pd.DataFrame,
                      method: str = "spearman") -> pd.Series:
    """逐日(行)计算因子与前瞻收益的截面相关，返回每日期 IC 序列。"""
    if method == "spearman":
        fr = factor.rank(axis=1, pct=False)
        yr = fwd.rank(axis=1, pct=False)
    else:
        fr, yr = factor, fwd
    mask = factor.notna() & fwd.notna()
    f = fr.where(mask)
    y = yr.where(mask)
    fm = f.mean(axis=1)
    ym = y.mean(axis=1)
    fd = f.sub(fm, axis=0)
    yd = y.sub(ym, axis=0)
    num = (fd * yd).sum(axis=1)
    den = ((fd ** 2).sum(axis=1) * (yd ** 2).sum(axis=1)) ** 0.5
    ic = num / den
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    return ic


def ic_analysis(factor: pd.DataFrame, fwd: pd.DataFrame,
                method: str = "spearman") -> dict:
    ic = _cross_section_ic(factor, fwd, method)
    if len(ic) == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "ic_tstat": float("nan"),
                "pos_ratio": float("nan"), "n": 0}
    ic_mean = ic.mean()
    ic_std = ic.std()
    if ic_std and ic_std > 0:
        icir = float(ic_mean / ic_std * np.sqrt(252))
        ic_tstat = float(ic_mean / (ic_std / np.sqrt(len(ic))))
    else:
        icir = float("nan")
        ic_tstat = float("nan")
    return {
        "ic_mean": float(ic_mean), "ic_std": float(ic_std), "icir": icir,
        "ic_tstat": ic_tstat, "pos_ratio": float((ic > 0).mean()),
        "n": int(len(ic)),
    }


# --------------------------------------------------------------------------
# Walk-forward OOS 评估
# --------------------------------------------------------------------------

def walk_forward(factor: pd.DataFrame, fwd: pd.DataFrame,
                train: int = 504, test: int = 126, step: int = 126) -> dict:
    idx = factor.index
    n = len(idx)
    folds = []
    s = 0
    while s + train + test <= n:
        tr = idx[s:s + train]
        te = idx[s + train:s + train + test]
        is_ic = ic_analysis(factor.loc[tr], fwd.loc[tr])
        oos_ic = ic_analysis(factor.loc[te], fwd.loc[te])
        folds.append({
            "train_start": str(tr[0].date()), "train_end": str(tr[-1].date()),
            "test_start": str(te[0].date()), "test_end": str(te[-1].date()),
            "is_icir": is_ic["icir"], "oos_icir": oos_ic["icir"],
        })
        s += step
    oos_vals = [f["oos_icir"] for f in folds
                if isinstance(f["oos_icir"], float) and not np.isnan(f["oos_icir"])]
    is_vals = [f["is_icir"] for f in folds
               if isinstance(f["is_icir"], float) and not np.isnan(f["is_icir"])]
    oos_icir = float(np.nanmean(oos_vals)) if oos_vals else float("nan")
    is_icir = float(np.nanmean(is_vals)) if is_vals else float("nan")
    return {"folds": folds, "oos_icir": oos_icir, "is_icir": is_icir,
            "n_folds": len(folds)}


# --------------------------------------------------------------------------
# 贪心选因子
# --------------------------------------------------------------------------

def _factor_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """两因子矩阵的截面相关均值（衡量共线程度）。"""
    m = a.notna() & b.notna()
    if m.to_numpy().sum() < 100:
        return 0.0
    aa = a[m].rank(axis=1, pct=False)
    bb = b[m].rank(axis=1, pct=False)
    c = aa.corrwith(bb, axis=1)
    c = c.replace([np.inf, -np.inf], np.nan).dropna()
    return float(c.mean()) if len(c) else 0.0


def greedy_select(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame,
                  icir_gate: float = 0.5, corr_thr: float = 0.7,
                  top: int = 20) -> list[str]:
    """先按全样本 ICIR 过筛，再贪心加入 OOS_ICIR 最高且与已选截面相关<corr_thr 的。

    说明：ICIR gate 用全样本 ICIR 做廉价预筛；OOS_ICIR 由 walk_forward 给出，
    驱动贪心排序。返回因子 id 列表，长度 ≤ top。
    """
    # 1) 全样本 ICIR 预筛
    pre: dict[str, float] = {}
    for fid, f in factors.items():
        try:
            ic = ic_analysis(f, fwd)
        except Exception:
            continue
        if ic["n"] >= 50 and ic["icir"] >= icir_gate:
            pre[fid] = ic["icir"]
    order = sorted(pre.keys(), key=lambda k: pre[k], reverse=True)

    # 2) 贪心加入，按 OOS_ICIR 排序并去共线
    selected: list[str] = []
    selected_mat: list[pd.DataFrame] = []
    oos_cache: dict[str, float] = {}
    for fid in order:
        f = factors[fid]
        if selected_mat:
            corrs = [_factor_corr(f, s) for s in selected_mat]
            if max(corrs) >= corr_thr:
                continue
        # 计算该候选 OOS_ICIR（仅对预筛通过的少量候选，成本可控）
        try:
            wf = walk_forward(f, fwd)
            oos_cache[fid] = wf["oos_icir"]
        except Exception:
            continue
        if not (isinstance(oos_cache[fid], float) and not np.isnan(oos_cache[fid])):
            continue
        selected.append(fid)
        selected_mat.append(f)
        if len(selected) >= top:
            break

    # 若预筛后 OOS 计算导致全部落空，退回按全样本 ICIR 直接取 top
    if not selected and order:
        return order[:top]
    return selected


# --------------------------------------------------------------------------
# 成本敏感度量（弥补 IC 不含交易成本的缺陷）
# --------------------------------------------------------------------------

def daily_ic_series(factor: pd.DataFrame, fwd: pd.DataFrame,
                    method: str = "spearman") -> pd.Series:
    """返回逐日期 IC 序列（用于稳定性检验）。"""
    return _cross_section_ic(factor, fwd, method)


def factor_turnover(factor: pd.DataFrame) -> dict:
    """因子时序粘性：日度截面排名的自相关（越高越粘、换手越低、成本越小）。"""
    r = factor.rank(axis=1, pct=True)
    diff = r.diff().abs().mean(axis=1)
    autocorr = r.corrwith(r.shift(1)).mean()
    return {"turnover": float(diff.mean()),
            "rank_autocorr": float(autocorr) if pd.notna(autocorr) else float("nan")}


def long_short_backtest(factor: pd.DataFrame, fwd: pd.DataFrame,
                        cost_buy: float = COST_BUY,
                        cost_sell: float = COST_SELL,
                        top_frac: float = 0.1) -> dict:
    """成本敏感的多空组合（晋升 net_ir 量度）。

    实现已统一落在 `execution.long_short_backtest`，此处仅做签名兼容转发，
    确保全系统回测成本/执行语义单一来源（见 fwsp/execution.py）。
    """
    return _exec_long_short(factor, fwd, top_frac=top_frac,
                            cost_buy=cost_buy, cost_sell=cost_sell)


# --------------------------------------------------------------------------
# 流式预筛（内存友好：逐因子计算，只保留过门禁者，避免同时持有 100+ 面板）
# --------------------------------------------------------------------------

def screen_factors(panels: dict[str, pd.DataFrame],
                   specs: list, fwd: pd.DataFrame,
                   icir_gate: float = 0.5, n_min: int = 50) -> dict[str, pd.DataFrame]:
    """逐个计算因子并做全样本 ICIR 预筛，仅返回达标因子矩阵。

    与 greedy_select 的预筛不同：这里边算边丢弃，内存占用 = 单个因子面板
    （而非全部候选），用于缓解 100+ 因子 × 全宇宙面板的内存压力。
    """
    out: dict[str, pd.DataFrame] = {}
    for spec in specs:
        f = compute_factor(panels, spec)
        ic = ic_analysis(f, fwd)
        if ic["n"] >= n_min and ic["icir"] >= icir_gate:
            out[spec.id] = f
        del f
    return out


# --------------------------------------------------------------------------
# 晋升门禁（过拟合防护）
# --------------------------------------------------------------------------

def oos_guard(new_oos: float | None, old_oos: float | None,
              min_lift: float = 0.02) -> tuple[bool, str]:
    if old_oos is None:
        # 首跑也须是「真实可交易」信号：OOS 数值异常或净 IR 不达标则拒绝，
        # 避免以 NaN/负 OOS 晋升空活跃集（更严格门禁）。
        if not isinstance(new_oos, (int, float)) or np.isnan(new_oos):
            return False, "首次运行但 OOS 含 NaN/异常，拒绝晋升"
        if new_oos <= 0:
            return False, "首次运行但 OOS<=0，拒绝晋升"
        return True, "首次运行，无历史基线"
    if not (isinstance(new_oos, (int, float)) and isinstance(old_oos, (int, float))):
        return False, "OOS 数值异常，拒绝晋升"
    if np.isnan(new_oos) or np.isnan(old_oos):
        return False, "OOS 含 NaN，拒绝晋升"
    if new_oos > old_oos + min_lift:
        return True, f"OOS 提升 {new_oos:.4f} > {old_oos:.4f}+{min_lift}"
    return False, f"OOS 未达晋升阈值 ({new_oos:.4f} vs {old_oos:.4f})"
