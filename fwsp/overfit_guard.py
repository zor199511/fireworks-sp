"""过拟合防护：OOS/IS 比率、告警、稳定性、健康汇总。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def oos_is_ratio(is_icir: float | None, oos_icir: float | None) -> float:
    if not (isinstance(is_icir, (int, float)) and isinstance(oos_icir, (int, float))):
        return float("nan")
    if is_icir == 0 or (isinstance(is_icir, float) and math.isnan(is_icir)):
        return float("nan")
    return float(oos_icir / is_icir)


def ratio_alert(is_icir: float | None, oos_icir: float | None,
                hi: float = 5.0, lo: float = 0.3) -> tuple[bool, str]:
    r = oos_is_ratio(is_icir, oos_icir)
    if math.isnan(r):
        return False, "ratio 无法计算(IS_ICIR 为 0 或 NaN)"
    if r > hi:
        return True, f"OOS/IS={r:.2f} 过高(>{hi})，疑似过拟合/参数挖掘"
    if r < lo:
        return True, f"OOS/IS={r:.2f} 过低(<{lo})，样本外失效"
    return False, f"OOS/IS={r:.2f} 正常"


def stability_check(ic_series: pd.Series, window: int = 252,
                    thr: float = 0.0) -> dict:
    """滚动 ICIR 稳定性：跌破阈值标失稳。"""
    s = pd.Series(ic_series).dropna() if not isinstance(ic_series, pd.Series) \
        else ic_series.dropna()
    if len(s) < window:
        return {"rolling_icir": float("nan"), "unstable": False,
                "worst": float("nan"), "n": int(len(s))}
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
    roll_icir = (roll_mean / roll_std * np.sqrt(252)).dropna()
    worst = float(roll_icir.min())
    return {"rolling_icir": float(roll_icir.iloc[-1]), "unstable": bool(worst < thr),
            "worst": worst, "n": int(len(s))}


def factor_health(rows: list[dict]) -> list[str]:
    """汇总告警清单。rows 为 factor_eval 行(dict)。"""
    warnings: list[str] = []
    for r in rows:
        code = r.get("code", "?")
        is_icir = r.get("is_icir")
        oos_icir = r.get("oos_icir")
        alert, msg = ratio_alert(is_icir, oos_icir)
        if alert:
            warnings.append(f"{code}: {msg}")
        stab = r.get("stability")
        if stab is not None and not (isinstance(stab, float) and math.isnan(stab)) \
                and stab < 0:
            warnings.append(f"{code}: 稳定性指标为负({stab})")
    return warnings
