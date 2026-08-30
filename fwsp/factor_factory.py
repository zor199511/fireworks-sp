"""因子工厂：表达式算子 + 配方展开 + 因子计算。

设计要点（过拟合防护第一关）：
- 所有时序算子只用 rolling / shift / pct_change / ewm / rank，**绝不**使用
  lead / shift(-n) 之类的未来值算子。因子在信号日 t 只能看到 ≤ t 的数据。
- 截面算子(cs_*) 对矩阵按「行=日期」操作，即同一交易日跨股票比较，同样
  不引入未来信息。
- `returns(x, n)` = x.pct_change(n)，内部用 shift，无前视。
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"


# --------------------------------------------------------------------------
# 算子实现：每个函数接收 DataFrame(date×code) 与整数窗口，返回同形 DataFrame
# --------------------------------------------------------------------------

def _mp(n: float) -> int:
    """rolling 的最小样本数：用一半窗口，保证有足够数据做 IC 而不引入前视。"""
    return max(1, int(int(n) // 2))


def _sma(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=_mp(n)).mean()


def _ema(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.ewm(span=int(n), adjust=False, min_periods=_mp(n)).mean()


def _std(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=_mp(n)).std()


def _ts_min(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=_mp(n)).min()


def _ts_max(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=_mp(n)).max()


def _zscore(x: pd.DataFrame, n: float) -> pd.DataFrame:
    n = int(n)
    m = x.rolling(n, min_periods=_mp(n)).mean()
    s = x.rolling(n, min_periods=_mp(n)).std()
    return (x - m) / s


def _ts_rank_last(a: np.ndarray) -> float:
    # 窗口内最后一个元素的百分位排名（含自身），无前视
    return float((a <= a[-1]).mean())


def _ts_rank(x: pd.DataFrame, n: float) -> pd.DataFrame:
    # 时序排名：当前值在回看窗口中的相对位置(0~1)
    return x.rolling(int(n), min_periods=_mp(n)).apply(_ts_rank_last, raw=True)


def _cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    # 截面排名：每行(日期)跨股票排序
    return x.rank(axis=1, pct=True)


def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    # 截面 zscore：每行去均值除标准差
    m = x.mean(axis=1)
    s = x.std(axis=1)
    return x.sub(m, axis=0).div(s, axis=0)


def _ref(x: pd.DataFrame, n: float) -> pd.DataFrame:
    # 前移 n 期（用历史值，无前视）
    return x.shift(int(n))


def _sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def _min(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=_mp(n)).min()


def _max(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x.rolling(int(n), min_periods=_mp(n)).max()


def _div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a / b


def _sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a - b


def _add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a + b


def _abs(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def _log(x: pd.DataFrame) -> pd.DataFrame:
    # 保护：<=0 的 amount 取 nan（与 build_factor_panel log(amt.clip(lower=1)) 一致）
    clipped = x.clip(lower=1e-12)
    return pd.DataFrame(np.log(clipped.values), index=clipped.index, columns=clipped.columns)


def _returns(x: pd.DataFrame, n: float) -> pd.DataFrame:
    # 收益率：pct_change 内部用 shift，无前视
    return x.pct_change(int(n))


def _corr(a: pd.DataFrame, b: pd.DataFrame, n: float) -> pd.DataFrame:
    # 时序滚动相关：每只股票回看窗口内 a 与 b 的相关系数
    return a.rolling(int(n), min_periods=_mp(n)).corr(b)


OPERATORS = {
    "sma": _sma, "ts_mean": _sma, "ts_min": _ts_min, "ts_max": _ts_max,
    "ema": _ema, "std": _std, "zscore": _zscore,
    "ts_rank": _ts_rank, "cs_rank": _cs_rank, "cs_zscore": _cs_zscore,
    "ref": _ref, "sign": _sign, "corr": _corr, "min": _min, "max": _max,
    "div": _div, "sub": _sub, "add": _add, "abs": _abs, "log": _log,
    "returns": _returns,
}

BASE_FIELDS = ("open", "high", "low", "close", "volume", "amount")


# --------------------------------------------------------------------------
# 表达式解析（极简递归下降，仅支持 函数调用 / 变量 / 数字，杜绝 eval）
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<num>\d+\.?\d*)
      | (?P<id>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<op>[(),])
    )
    """, re.VERBOSE,
)


def _tokenize(s: str):
    pos = 0
    out = []
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(f"无法解析因子表达式位置 {pos}: {s[pos:pos+10]!r}")
        pos = m.end()
        if m.group("num") is not None:
            out.append(("num", float(m.group("num"))))
        elif m.group("id") is not None:
            out.append(("id", m.group("id")))
        else:
            out.append(("op", m.group("op")))
    return out


class _Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        node = self._expr()
        if self.i != len(self.t):
            raise ValueError("表达式存在多余 token")
        return node

    def _expr(self):
        tok = self.peek()
        if tok[0] == "num":
            self.next()
            return ("num", tok[1])
        if tok[0] == "id":
            self.next()
            if self.peek() == ("op", "("):
                self.next()  # (
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self._expr())
                    while self.peek() == ("op", ","):
                        self.next()
                        args.append(self._expr())
                if self.peek() != ("op", ")"):
                    raise ValueError("缺少右括号")
                self.next()  # )
                return ("call", tok[1], args)
            return ("var", tok[1])
        raise ValueError(f"意外 token: {tok}")


@functools.lru_cache(maxsize=512)
def _parse(expr: str):
    return _Parser(_tokenize(expr)).parse()


def _eval(node, env, cache=None):
    t = node[0]
    if t == "num":
        return float(node[1])
    if t == "var":
        name = node[1]
        if name not in env:
            raise KeyError(f"未知字段: {name}（可用: {BASE_FIELDS}）")
        return env[name]
    if t == "call":
        if cache is not None:
            hit = cache.get(id(node))
            if hit is not None:
                return hit
        fname = node[1]
        if fname not in OPERATORS:
            raise KeyError(f"未知算子: {fname}")
        args = [_eval(a, env, cache) for a in node[2]]
        res = OPERATORS[fname](*args)
        if cache is not None:
            cache[id(node)] = res
        return res
    raise ValueError(f"坏节点: {node}")


# --------------------------------------------------------------------------
# 因子规格与计算
# --------------------------------------------------------------------------

@dataclass
class FactorSpec:
    id: str
    category: str
    expr: str
    params: dict = field(default_factory=dict)
    desc: str = ""
    source: str = ""


def compute_factor(panels: dict[str, pd.DataFrame], spec: FactorSpec) -> pd.DataFrame:
    """计算单个因子，返回 date×code 的 DataFrame。全程无未来值。"""
    env = {k: panels[k] for k in BASE_FIELDS if k in panels}
    ast = _parse(spec.expr)
    out = _eval(ast, env, cache={})
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out)
    return out


def compute_factor_panel(panels: dict[str, pd.DataFrame],
                        specs: list[FactorSpec]) -> dict[str, pd.DataFrame]:
    env = {k: panels[k] for k in BASE_FIELDS if k in panels}
    cache: dict[int, pd.DataFrame] = {}
    out = {}
    for s in specs:
        res = _eval(_parse(s.expr), env, cache=cache)
        out[s.id] = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
    return out


# --------------------------------------------------------------------------
# 配方加载与展开
# --------------------------------------------------------------------------

def load_base_recipes(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else RECIPES_DIR / "base.yaml"
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    recipes = data.get("recipes", data) if isinstance(data, dict) else data
    if not isinstance(recipes, list):
        raise ValueError("base.yaml 顶层应为 recipes 列表")
    required = ("id", "category", "expr")
    for r in recipes:
        for k in required:
            if k not in r:
                raise ValueError(f"配方缺少字段 {k}: {r}")
        r.setdefault("desc", r["id"])
        r.setdefault("scan", {})
    return recipes


def expand_recipes(base: list[dict], grid: dict[str, list]) -> list[FactorSpec]:
    """按 grid 扫描占位符 {w}/{m} 等，生成 100+ 候选因子。"""
    specs: list[FactorSpec] = []
    for r in base:
        scan = r.get("scan", {}) or {}
        src = r.get("source", "") or ""
        if not scan:
            specs.append(FactorSpec(r["id"], r["category"], r["expr"],
                                    {}, r.get("desc", ""), source=src))
            continue
        # 笛卡尔积：每个占位符对应 grid 中的一个列表
        keys = list(scan.keys())
        lists = [grid[scan[k]] for k in keys]
        import itertools
        for combo in itertools.product(*lists):
            subs = {k: v for k, v in zip(keys, combo)}
            expr = r["expr"].format(**subs)
            suffix = "_".join(f"{k}{v}" for k, v in subs.items())
            fid = f"{r['id']}__{suffix}"
            desc = f"{r.get('desc','')} [{', '.join(f'{k}={v}' for k,v in subs.items())}]"
            specs.append(FactorSpec(fid, r["category"], expr, dict(subs), desc,
                                   source=src))
    return specs
