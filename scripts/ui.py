"""UI component library — banking + warehouse dashboard blend.

Provides: inject_css, sidebar_nav, section_title, metric_row, card_container,
badge, alert_banner, sparkline, and ECharts chart wrappers.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_echarts import st_echarts as _st_echarts  # type: ignore[import-untyped]
except ImportError:  # graceful degradation if echarts not installed
    _st_echarts = None  # type: ignore[assignment]

_CSS_PATH = Path(__file__).resolve().parent / "static" / "app.css"


# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject custom CSS once per page load."""
    if _CSS_PATH.exists():
        st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar_nav(pages: list[str]) -> str:
    """Radio-based navigation in the sidebar. Returns selected page."""
    return st.radio("页面", pages, label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Typography / layout helpers
# ---------------------------------------------------------------------------

def section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


@contextmanager
def card_container(title: str | None = None):
    """Yields a styled card block. Usage: with card_container(): ..."""
    tag = "div"
    st.markdown(f'<{tag} class="card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    yield
    st.markdown(f'</{tag}>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI / badges / alerts
# ---------------------------------------------------------------------------

def metric_row(items: list[dict], *, key_prefix: str = "") -> None:
    """Render a row of KPI cards.

    Each *item* dict: {"label": str, "value": str, "delta": str (opt),
                        "delta_type": "positive"|"negative"|"neutral" (opt)}
    """
    cards_html = []
    for i, item in enumerate(items):
        cls = item.get("delta_type", "neutral")
        delta_html = (f'<div class="kpi-delta {cls}">{item["delta"]}</div>'
                      if item.get("delta") else "")
        cards_html.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{item["label"]}</div>'
            f'<div class="kpi-value">{item["value"]}</div>'
            f'{delta_html}</div>')
    html = f'<div class="kpi-row">{"".join(cards_html)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def badge(text: str, variant: str = "info") -> str:
    """Return an HTML badge span. variant: success|warning|danger|info|muted."""
    return f'<span class="badge badge-{variant}">{text}</span>'


import re as _re

_BADGE_SIGN_RE = _re.compile(r"([+-]\d+\.?\d*)\s*\)?\s*$")


def reason_badge(text: str) -> str:
    """推荐理由 badge：带符号数值(如 fid(+1.2) / name-1.23)按正负着色，
    正=利好(success) 负=利空(danger)，其余中性(info)。"""
    m = _BADGE_SIGN_RE.search(text)
    if m:
        variant = "success" if m.group(1).startswith("+") else "danger"
        return badge(text, variant)
    return badge(text, "info")


def alert_banner(text: str, variant: str = "info") -> None:
    """Render a colored alert banner."""
    st.markdown(f'<div class="alert-banner {variant}">{text}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ECharts wrapper (thin, guarded)
# ---------------------------------------------------------------------------

def _echarts(options: dict, height: int = 400) -> None:
    """Render an ECharts chart. Falls back to a warning if package missing."""
    if _st_echarts is None:
        st.warning("streamlit-echarts 未安装，图表无法渲染")
        return
    _st_echarts(options, height=height)


def sparkline(series: list[float] | pd.Series, color: str = "#f59e0b",
              height: int = 40) -> None:
    """Inline sparkline chart for KPI cards."""
    data = [round(float(v), 3) for v in series]
    opts = {
        "backgroundColor": "transparent",
        "grid": {"left": 2, "right": 2, "top": 2, "bottom": 2},
        "xAxis": {"type": "category", "show": False, "data": list(range(len(data)))},
        "yAxis": {"show": False, "scale": True},
        "series": [{
            "type": "line", "data": data, "smooth": True, "showSymbol": False,
            "lineStyle": {"color": color, "width": 2},
            "areaStyle": {"color": color, "opacity": 0.15},
        }],
    }
    _echarts(opts, height=height)


# ---------------------------------------------------------------------------
# Chart builders — all return echarts options dict
# ---------------------------------------------------------------------------

_DARK = "transparent"
_AXIS = {"axisLabel": {"color": "#94a3b8"}}
_GRID = {"left": 55, "right": 18, "top": 28, "bottom": 12}


def candle_chart(df: pd.DataFrame) -> dict:
    """K-line with MA5/20/60 + volume sub-chart + dataZoom."""
    dates = (df["date"].dt.strftime("%Y-%m-%d")
             if hasattr(df["date"], "dt") else df["date"]).tolist()
    o, h, l, c = (df["open"].tolist(), df["high"].tolist(),
                   df["low"].tolist(), df["close"].tolist())
    vol = df["volume"].tolist()
    kdata = [[o[i], c[i], l[i], h[i]] for i in range(len(df))]
    vdata = [{"value": v, "itemStyle": {"color": "#ef4444" if c[i] >= o[i] else "#10b981"}}
             for i, v in enumerate(vol)]
    ma5  = [None if pd.isna(v) else round(v, 2) for v in df["ma5"]]
    ma20 = [None if pd.isna(v) else round(v, 2) for v in df["ma20"]]
    ma60 = [None if pd.isna(v) else round(v, 2) for v in df["ma60"]]
    return {
        "backgroundColor": _DARK,
        "animation": False,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "grid": [
            {**_GRID, "height": "62%"},
            {**_GRID, "top": "78%", "height": "14%"},
        ],
        "xAxis": [
            {"type": "category", "data": dates, "gridIndex": 0,
             "axisLabel": {**_AXIS.get("axisLabel", {}), "show": False},
             "axisTick": {"show": False}},
            {"type": "category", "data": dates, "gridIndex": 1, **_AXIS},
        ],
        "yAxis": [
            {"scale": True, "gridIndex": 0, **_AXIS},
            {"scale": True, "gridIndex": 1, **_AXIS, "splitNumber": 2},
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1]},
            {"type": "slider", "xAxisIndex": [0, 1], "bottom": 0, "height": 12},
        ],
        "series": [
            {"type": "candlestick", "data": kdata, "xAxisIndex": 0,
             "yAxisIndex": 0, "itemStyle": {
                 "color": "#ef4444", "color0": "#10b981",
                 "borderColor": "#ef4444", "borderColor0": "#10b981"}},
            *[{"type": "line", "data": d, "name": n, "xAxisIndex": 0,
               "yAxisIndex": 0, "showSymbol": False, "smooth": True,
               "lineStyle": {"color": clr, "width": 1}}
              for n, d, clr in [("MA5", ma5, "#f59e0b"),
                                ("MA20", ma20, "#3b82f6"),
                                ("MA60", ma60, "#8b5cf6")]],
            {"type": "bar", "data": vdata, "xAxisIndex": 1, "yAxisIndex": 1},
        ],
    }


def ratio_bar(df: pd.DataFrame, x_col: str, y_col: str,
              thresholds: list[tuple[float, str]] | None = None) -> dict:
    """Bar chart with optional horizontal threshold lines."""
    x = df[x_col].tolist()
    data = [round(float(v), 4) if pd.notna(v) else 0 for v in df[y_col]]
    marklines = [{"yAxis": v, "label": {"show": True, "formatter": str(v)},
                  "lineStyle": {"color": c, "type": "dashed"}}
                 for v, c in (thresholds or [])]
    return {
        "backgroundColor": _DARK,
        "tooltip": {"trigger": "axis"},
        "grid": {**_GRID, "bottom": 70},
        "xAxis": {"type": "category", "data": x, "axisLabel": {**_AXIS.get("axisLabel", {}), "rotate": 40}},
        "yAxis": {"type": "value", **_AXIS},
        "series": [{
            "type": "bar", "data": data,
            "itemStyle": {"color": "#3b82f6"},
            "markLine": {"silent": True, "data": marklines},
        }],
    }


def pareto_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                   color_col: str) -> dict:
    """Scatter x_col vs y_col, colored by color_col, with diagonal markLine."""
    data = []
    for _, r in df.iterrows():
        data.append([
            round(float(r[x_col]), 3), round(float(r[y_col]), 3),
            round(float(r[color_col]), 3), str(r.get("code", "")),
        ])
    vmin, vmax = (0, 1)
    if data:
        colors = [d[2] for d in data]
        vmin, vmax = min(colors), max(colors)
    # diagonal
    lo = min(d[0] for d in data) if data else 0
    hi = max(d[0] for d in data) if data else 1
    return {
        "backgroundColor": _DARK,
        "tooltip": {"trigger": "item",
                    "formatter": lambda: None,  # placeholder; set via JS if needed
                    },
        "grid": {**_GRID, "right": 90},
        "xAxis": {"type": "value", "name": "IS ICIR", **_AXIS},
        "yAxis": {"type": "value", "name": "OOS ICIR", **_AXIS},
        "visualMap": {
            "min": vmin, "max": vmax, "dimension": 2,
            "inRange": {"color": ["#ef4444", "#f59e0b", "#10b981"]},
            "text": ["净IR高", "低"], "right": 0, "top": 0,
            "textStyle": {"color": "#94a3b8"},
        },
        "series": [{
            "type": "scatter", "data": data, "symbolSize": 11,
            "markLine": {"silent": True, "data": [
                {"xAxis": lo, "yAxis": lo, "lineStyle": {"color": "#475569", "type": "dotted"}},
                {"xAxis": hi, "yAxis": hi, "lineStyle": {"color": "#475569", "type": "dotted"}},
            ]},
        }],
    }


def timeline(ev: pd.DataFrame) -> dict:
    """OOS ICIR evolution over runs. Promotion points highlighted."""
    x = ev["run_at"].astype(str).tolist()
    y = ev["new_oos"].astype(float).round(4).tolist()
    promoted_idx = [i for i, v in enumerate(ev["promoted"]) if v]
    mark_data = [{"type": "max", "name": "最高"}] + (
        [{"coord": [x[i], y[i]], "value": "晋升",
          "itemStyle": {"color": "#10b981"}} for i in promoted_idx] if promoted_idx else [])
    return {
        "backgroundColor": _DARK,
        "tooltip": {"trigger": "axis"},
        "grid": {**_GRID},
        "xAxis": {"type": "category", "data": x, **_AXIS},
        "yAxis": {"type": "value", **_AXIS},
        "series": [{
            "type": "line", "data": y, "smooth": True,
            "lineStyle": {"color": "#f59e0b"},
            "itemStyle": {"color": "#f59e0b"},
            "markPoint": {"data": mark_data},
        }],
    }


def equity_curve(eq: pd.Series, bench: pd.Series | None = None) -> dict:
    """Strategy equity line vs optional benchmark."""
    eq = eq.sort_index()
    x = [str(d.date()) for d in eq.index]
    y = eq.round(4).tolist()
    series: list[dict] = [{
        "type": "line", "data": y, "name": "策略", "smooth": True,
        "showSymbol": False,
        "lineStyle": {"color": "#f59e0b"}, "itemStyle": {"color": "#f59e0b"},
    }]
    if bench is not None and len(bench) > 1:
        bench = bench.sort_index()
        bx = [str(d.date()) for d in bench.index]
        series.append({
            "type": "line", "data": bench.round(4).tolist(),
            "name": "沪深300", "smooth": True, "showSymbol": False,
            "lineStyle": {"color": "#64748b", "width": 1},
            "itemStyle": {"color": "#64748b"},
        })
        x = bx  # align to bench dates
    return {
        "backgroundColor": _DARK,
        "tooltip": {"trigger": "axis"},
        "legend": {"textStyle": {"color": "#94a3b8"}},
        "grid": {**_GRID},
        "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 12, "bottom": 0}],
        "xAxis": {"type": "category", "data": x, **_AXIS},
        "yAxis": {"type": "value", **_AXIS},
        "series": series,
    }


def factor_bar(df: pd.DataFrame, x_col: str, y_col: str,
               color_col: str | None = None) -> dict:
    """Bar chart of factor metric, optionally colored by another column."""
    x = df[x_col].tolist()
    raw = df[y_col].fillna(0).tolist()
    if color_col is not None:
        cv = df[color_col].fillna(0).tolist()
        data = [{"value": round(v, 4), "itemStyle": {
            "color": "#ef4444" if cv[i] <= 0 else "#3b82f6"}}
            for i, v in enumerate(raw)]
    else:
        data = [round(v, 4) for v in raw]
    return {
        "backgroundColor": _DARK,
        "tooltip": {"trigger": "axis"},
        "grid": {**_GRID, "bottom": 60},
        "xAxis": {"type": "category", "data": x,
                  "axisLabel": {**_AXIS.get("axisLabel", {}), "rotate": 40}},
        "yAxis": {"type": "value", **_AXIS},
        "series": [{"type": "bar", "data": data}],
    }


def freq_bar(fc: pd.DataFrame) -> dict:
    """Factor selection frequency bar."""
    return factor_bar(fc, "factor", "times_selected")
