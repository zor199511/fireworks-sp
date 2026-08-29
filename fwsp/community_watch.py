"""社区因子监控：把公开量化研究里的因子思路搬进本地挖掘池。

两层来源：
1. 本地 recipes/community.yaml —— 已翻译好的 WorldQuant Alpha101 / QLib Alpha158
   表达式，直接并入挖掘池（离线可用）。
2. 联网刷新 refresh_from_github() —— 调用 GitHub Search API 找「近期活跃的因子/
   多因子」仓库，列出 trending 源供人工或自动提炼；并尽力解析 WorldQuant Alpha101
   原始公式文本，把可翻译的简单条目追加进 community.yaml（解析失败则跳过，不阻塞）。

所有联网动作都带超时与异常兜底：断网/限流时静默回退到本地缓存，保证 auto_evolve
离线也能跑。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from .factor_factory import RECIPES_DIR, load_base_recipes

log = logging.getLogger("fwsp.community_watch")

COMMUNITY_YAML = RECIPES_DIR / "community.yaml"

# GitHub 上可借鉴的因子源（精选，避免动态搜索的不可靠）
CURATED_SOURCES = [
    ("worldquant_alpha101", "Harvey-Sun/World_Quant_Alphas",
     "WorldQuant Alpha101 公式化因子圣经(364星)"),
    ("qlib_alpha158", "microsoft/qlib",
     "Microsoft QLib，Alpha158/360 因子表达式库"),
    ("alpha_skills", "VernonOY/alpha-skills",
     "25+ 因子 + ICIR 门槛 + 经济直觉评分方法论"),
    ("dynamic_factor_rotation", "yuhua-crypto/dynamic-factor-rotation",
     "A 股多因子框架(akshare, 18 个真实 A 股因子)"),
]


def load_community_recipes(path=None) -> list:
    p = Path(path) if path else COMMUNITY_YAML
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    recipes = data.get("recipes", []) if isinstance(data, dict) else data
    for r in recipes:
        r.setdefault("desc", r["id"])
        r.setdefault("scan", {})
    return recipes


def merge_all_recipes() -> list:
    """base.yaml + community.yaml 合并，供 auto_evolve 挖掘。"""
    return load_base_recipes() + load_community_recipes()


def list_curated_sources() -> list:
    return CURATED_SOURCES


def refresh_from_github(timeout: int = 15) -> dict:
    """联网刷新：返回近期活跃的因子仓库列表（trending 渠道）。

    实现：用 GitHub Search API（匿名，有限流）搜 'alpha factor' / 'worldquant' /
    'A股 因子' 仓库，按 star 排序取前若干。断网/限流时返回空并记 warning。
    """
    out = {"sources": [], "parsed": 0, "note": ""}
    try:
        import requests
    except ImportError:
        out["note"] = "requests 未安装，跳过联网刷新"
        return out
    queries = [
        "alpha+factor+language:python",
        "worldquant+alpha101",
        "A股+多因子+选股",
    ]
    seen = set()
    try:
        for q in queries:
            url = ("https://api.github.com/search/repositories"
                   f"?q={q}&sort=stars&order=desc&per_page=8")
            resp = requests.get(url, timeout=timeout,
                                headers={"Accept": "application/vnd.github+json"})
            if resp.status_code != 200:
                out["note"] = f"github API {resp.status_code}（可能限流）"
                continue
            for it in resp.json().get("items", []):
                full = it["full_name"]
                if full in seen:
                    continue
                seen.add(full)
                out["sources"].append({
                    "repo": full, "stars": it.get("stargazers_count", 0),
                    "desc": (it.get("description") or "")[:80],
                    "updated": it.get("pushed_at", ""),
                })
    except Exception as e:  # 断网等
        out["note"] = f"联网刷新失败(离线回退): {e}"
        return out
    out["parsed"] = _try_parse_alpha101(timeout=timeout)
    return out


def _try_parse_alpha101(timeout: int = 15, max_add: int = 6) -> int:
    """尽力解析 WorldQuant Alpha101 原文公式，把可翻译的简单条目追加 community.yaml。

    只登记原始公式到 community.yaml（expr 用占位，需人工翻译，避免错误表达式）。
    任何失败静默返回已追加数。
    """
    raw = ("https://raw.githubusercontent.com/Harvey-Sun/World_Quant_Alphas/"
           "master/alpha101_formula.txt")
    try:
        import requests
        resp = requests.get(raw, timeout=timeout)
        if resp.status_code != 200:
            return 0
        text = resp.text
    except Exception:
        return 0

    existing = {r["id"] for r in load_community_recipes()}
    added = 0
    for m in re.finditer(r"alpha#(\d+)\s*[:=]\s*(.+)", text, re.IGNORECASE):
        num = m.group(1)
        fid = f"wq_raw_{num}"
        if fid in existing:
            continue
        formula = m.group(2).strip().replace("\n", " ")
        if added >= max_add:
            break
        _append_community({
            "id": fid, "category": "community",
            "source": "worldquant_alpha101_raw",
            "desc": f"Alpha101 原文# {num}（待翻译）: {formula[:120]}",
            "expr": "returns(close,1)",
            "scan": {},
        })
        existing.add(fid)
        added += 1
    return added


def _append_community(recipe: dict):
    recipes = load_community_recipes()
    recipes.append(recipe)
    with open(COMMUNITY_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump({"recipes": recipes}, f, allow_unicode=True,
                       sort_keys=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("curated sources:")
    for s in list_curated_sources():
        print(" ", s)
    res = refresh_from_github()
    print("refresh:", json.dumps(res, ensure_ascii=False, indent=2))
