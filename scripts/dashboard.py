import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="fireworks-sp", page_icon="🎆",
                   layout="wide")

from ui import (inject_css, sidebar_nav, section_title, metric_row,  # noqa: E402
                card_container, badge, alert_banner, candle_chart,
                ratio_bar, pareto_scatter, timeline, equity_curve,
                factor_bar, freq_bar)
inject_css()

from fwsp.backtest import run_backtest  # noqa: E402
from fwsp.config import FILTERS  # noqa: E402
from fwsp.db import get_conn, init_schema  # noqa: E402
from fwsp.tracker import summary_stats, update_tracking  # noqa: E402
from fwsp import factors as F, multifactor as M  # noqa: E402


@st.cache_data(ttl=120)
def load_meta():
    with get_conn() as conn:
        init_schema(conn)
        row = conn.execute(
            "SELECT (SELECT COUNT(*) FROM stock_list),"
            "(SELECT COUNT(DISTINCT code) FROM daily),"
            "(SELECT MAX(date) FROM daily),"
            "(SELECT value FROM meta WHERE key='last_update')").fetchone()
    return {"stocks": row[0], "codes": row[1], "last_bar": row[2],
            "last_update": row[3]}


@st.cache_data(ttl=300)
def latest_recommendations():
    with get_conn() as conn:
        d = conn.execute("SELECT MAX(run_date) FROM recommendations"
                         ).fetchone()[0]
        if not d:
            return None, []
        rows = conn.execute(
            "SELECT rank,code,name,industry,score,price,reasons,metrics "
            "FROM recommendations WHERE run_date=? ORDER BY rank",
            (d,)).fetchall()
    recos = []
    for r in rows:
        recos.append({
            "rank": r[0], "code": r[1], "name": r[2], "industry": r[3],
            "score": r[4], "price": r[5],
            "reasons": json.loads(r[6] or "[]"),
            "metrics": json.loads(r[7] or "{}"),
        })
    return d, recos


@st.cache_data(ttl=600)
def load_kline(code: str):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT date,open,high,low,close,volume FROM daily "
            "WHERE code=? ORDER BY date DESC LIMIT 250", (code,)).fetchall()
    df = pd.DataFrame(rows[::-1], columns=["date", "open", "high", "low",
                                           "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for m, n in (("ma5", 5), ("ma20", 20), ("ma60", 60)):
        df[m] = df["close"].rolling(n).mean()
    return df


def candle_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"],
        close=df["close"], name="K线",
        increasing_line_color="#ef4444", decreasing_line_color="#10b981"))
    for m, col in (("ma5", "#f59e0b"), ("ma20", "#3b82f6"),
                   ("ma60", "#8b5cf6")):
        fig.add_trace(go.Scatter(x=df["date"], y=df[m], name=m,
                                 line=dict(color=col, width=1)))
    fig.update_layout(height=460, xaxis_rangeslider_visible=False,
                       margin=dict(l=8, r=8, t=24, b=8),
                       legend=dict(orientation="h", y=1.02))
    return fig


@st.cache_data(ttl=3600)
def build_multifactor():
    with get_conn() as conn:
        blob = F.build_factor_panel(conn)
        fwd = F.forward_returns(blob["close"])
        quality = blob["quality"]
    z, ic = M.precompute(blob["factors"], fwd, quality, 10)
    return z, ic, blob["close"], blob["open"], blob["low"], quality


@st.cache_data(ttl=600)
def load_multifactor_meta():
    with get_conn() as conn:
        return M.load_selected(conn)


# --------------------------------------------------------------- sidebar

meta = load_meta()
with st.sidebar:
    st.title("🎆 fireworks-sp")
    st.caption(f"数据更新: {meta['last_update'] or '—'}")
    st.caption(f"股票池 {meta['stocks']} | K线覆盖 {meta['codes']} | "
               f"最新交易日 {meta['last_bar'] or '—'}")
    page = sidebar_nav(["今日推荐", "个股查询", "策略回测", "多因子策略",
                        "推荐追踪", "因子进化", "因子库", "进化历史"])
    st.divider()
    st.caption("候选池生成器，非投资建议。\n买入需独立判断并严格止损。")

# ---------------------------------------------------------------- pages

if page == "今日推荐":
    section_title(f"今日 Top {FILTERS['top_n']} 候选")
    run_date, recos = latest_recommendations()
    if not recos:
        st.info("还没有推荐记录。先运行 scripts/recommend.py")
        st.stop()

    avg_score = sum(r["score"] for r in recos) / len(recos)
    metric_row([
        {"label": "平均评分", "value": f"{avg_score:.1f}"},
        {"label": "入选数", "value": str(len(recos))},
        {"label": "信号日期", "value": run_date},
    ])

    for r in recos:
        mv_yi = (r["metrics"].get("total_mv") or 0) / 1e8
        head = (f"#{r['rank']} {r['name']} ({r['code']})  ·  "
                f"{r['industry'] or '—'}  ·  评分 {r['score']:.0f}  ·  "
                f"¥{r['price']:.2f}")
        with card_container(head):
            # reasons as badges
            reasons_html = " ".join(badge(txt, "info") for txt in r["reasons"])
            if reasons_html:
                st.markdown(reasons_html, unsafe_allow_html=True)
            tail = (f"PE {r['metrics'].get('pe_dyn') or '—'} | "
                    f"PB {r['metrics'].get('pb') or '—'} | "
                    f"ROE {r['metrics'].get('roe') or '—'}% | "
                    f"市值 {mv_yi:.0f}亿")
            st.caption(tail)
            kdf = load_kline(r["code"])
            if len(kdf):
                _echarts(candle_chart(kdf), height=380)

elif page == "个股查询":
    st.header("个股K线与指标")
    with get_conn() as conn:
        pairs = conn.execute(
            "SELECT s.code, s.name FROM stock_list s JOIN spot p "
            "ON p.code=s.code ORDER BY s.code").fetchall()
    code_map = {f"{n} ({c})": c for c, n in pairs}
    pick = st.selectbox("选择股票", list(code_map.keys()),
                        index=list(code_map.keys()).index(
                            "贵州茅台 (600519)")
                        if "贵州茅台 (600519)" in code_map else 0)
    if pick:
        kdf = load_kline(code_map[pick])
        _echarts(candle_chart(kdf), height=460)
        with card_container("最近行情"):
            st.dataframe(kdf.tail(15).iloc[:, :6].sort_values(
                "date", ascending=False).set_index("date"),
                use_container_width=True, hide_index=True)

elif page == "策略回测":
    st.header("策略回测")
    st.caption("周度调仓 · Top10 等权 · T+1开盘买入 · 持有10日 · -8%止损 · "
               "含佣金/印花税/滑点。反转策略：超跌优质股企稳信号。"
               "注：基本面门槛未纳入历史重放，结果偏乐观。")
    c1, c2, c3, c4 = st.columns(4)
    start = c1.text_input("开始日期", "2024-06-01")
    hold = c2.number_input("持有天数", 1, 40, 5)
    stop = c3.number_input("止损%", -30, -1, -8)
    profit = c4.number_input("止盈%", 1, 30, 8)
    if st.button("运行回测", type="primary"):
        with st.spinner("回测中…"):
            try:
                res = run_backtest(start=start, hold_days=int(hold),
                                   stop_pct=float(stop),
                                   profit_pct=float(profit))
                st.session_state.bt = res
            except Exception as e:  # noqa: BLE001
                st.error(f"回测失败: {e}")

    res = st.session_state.get("bt")
    if res:
        bench_pct = f"{res['bench_return']*100:+.1f}%" if res["bench_return"] is not None else "—"
        metric_row([
            {"label": "总收益", "value": f"{res['total_return']*100:+.1f}%",
             "delta": f"基准 {bench_pct}",
             "delta_type": "positive" if (res["bench_return"] or 0) > 0 else "negative"},
            {"label": "年化", "value": f"{res['cagr']*100:+.1f}%"},
            {"label": "最大回撤", "value": f"{res['max_drawdown']*100:.1f}%"},
            {"label": "胜率", "value": f"{res['win_rate']*100:.0f}%",
             "delta": f"{res['n_trades']}笔"},
            {"label": "夏普", "value": f"{res['sharpe']:.2f}"},
        ])

        eq = pd.Series(res["equity"])
        eq.index = pd.to_datetime(eq.index)
        eq_n = eq / eq.iloc[0]

        with get_conn() as conn:
            b = conn.execute(
                "SELECT date,close FROM index_daily WHERE code='sh.000300' "
                "AND date BETWEEN ? AND ? ORDER BY date",
                (res["start"], res["end"])).fetchall()
        bdf = pd.DataFrame(b, columns=["date", "close"])
        bench_s = None
        if len(bdf) > 1:
            bench_s = bdf.set_index(pd.to_datetime(bdf["date"]))["close"]
            bench_s = bench_s / bench_s.iloc[0]
        with card_container("净值曲线"):
            _echarts(equity_curve(eq_n, bench_s), height=400)

elif page == "多因子策略":
    st.header("多因子策略（walk-forward 自动挖掘）")
    selected = load_multifactor_meta()
    if not selected:
        st.warning("尚未挖掘因子。点下方「重新挖掘因子」生成（约 5 分钟）。")
    else:
        with get_conn() as conn:
            raw = conn.execute(
                "SELECT value FROM meta WHERE key='multifactor_summary'"
            ).fetchone()
        import json
        summ = json.loads(raw[0]) if raw else None
        st.caption("选中因子: " + ", ".join(selected))
        if summ:
            oos_ret = summ['oos_total_return']*100
            oos_sh = summ['oos_sharpe']
            oos_ret_cls = "positive" if oos_ret > 0 else "negative"
            oos_sh_cls  = "positive" if oos_sh > 0 else "negative"
            metric_row([
                {"label": "OOS 总收益", "value": f"{oos_ret:+.1f}%",
                 "delta_type": oos_ret_cls},
                {"label": "OOS 夏普", "value": f"{oos_sh:.2f}",
                 "delta_type": oos_sh_cls},
                {"label": "OOS 回撤", "value": f"{summ['oos_max_drawdown']*100:.1f}%"},
                {"label": "OOS 胜率", "value": f"{summ['oos_win_rate']*100:.0f}%"},
            ])
            st.caption(f"样本外窗口 {summ['holdout']}→今；"
                       f"对比 IS 总收益 {summ['is_total_return']*100:+.1f}% / "
                       f"夏普 {summ['is_sharpe']:.2f}。非投资建议。")

    c1, c2 = st.columns(2)
    if c1.button("生成实时推荐", type="primary"):
        with st.spinner("构建因子面板(首次约2分钟，之后缓存1小时)…"):
            z, ic, close, opn, low, quality = build_multifactor()
            recs = M.live_recommend(z, ic, close, opn, low, quality,
                                    selected=selected)
        if not recs:
            st.error("无可用推荐（因子未挖掘或市场数据不足）")
        else:
            st.success(f"实时 Top {len(recs)} 候选（信号日 "
                       f"{str(close.index[-1].date())}）")
            rows = [{"排名": i + 1, "代码": r["code"],
                     "综合分": round(r["score"], 2),
                     "因子贡献": ", ".join(r["reasons"])}
                    for i, r in enumerate(recs)]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption("因子贡献 = 各因子 z 分 × 训练窗 IC 权重，"
                       "仅供研究。")

    if c2.button("重新挖掘因子(约5分钟)"):
        with st.spinner("贪心挖掘中，请勿关闭…"):
            import subprocess, sys
            subprocess.run([sys.executable, "scripts/factor_mine.py"],
                           cwd=str(Path(__file__).resolve().parent.parent),
                           check=True)
            st.cache_data.clear()
        st.success("挖掘完成，已更新 meta。刷新页面查看。")

    st.divider()
    st.subheader("参数化多因子回测")
    st.caption("用已挖掘因子 + 时间变化质量面板跑 walk-forward（不做 Greedy 重选）。")
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    freq = rc1.selectbox("调仓频率", ["周", "月"], index=0, key="mf_freq")
    hzn = rc2.number_input("持有天数", 5, 40, 10, key="mf_hold")
    stp = rc3.number_input("止损%", -30, -1, -8, key="mf_stop")
    trl = rc4.number_input("跟踪止损%", 0, 30, 0, key="mf_trail")
    tpn = rc5.number_input("Top N", 3, 30, 10, key="mf_topn")
    if st.button("运行多因子回测", key="mf_run", type="primary"):
        if not selected:
            st.warning("请先挖掘因子（上方按钮）。")
        else:
            with st.spinner("构建因子面板(首次约2分钟) + 回测…"):
                z, ic, close, opn, low, quality = build_multifactor()
                with get_conn() as conn:
                    qp = F.quality_panel(conn, close.index)
                res = M.walk_forward_backtest(
                    z, ic, close, opn, low, qp, start="2024-06-01",
                    top_n=int(tpn), horizon=int(hzn), stop_pct=float(stp),
                    rebal=('M' if freq == '月' else 'W'),
                    trail=(float(trl) if trl > 0 else None),
                    selected=selected)
            metric_row([
                {"label": "总收益", "value": f"{res['total_return']*100:+.1f}%"},
                {"label": "年化",   "value": f"{res['cagr']*100:+.1f}%"},
                {"label": "最大回撤", "value": f"{res['max_drawdown']*100:.1f}%"},
                {"label": "胜率",   "value": f"{res['win_rate']*100:.0f}%",
                 "delta": f"{int(res['n_trades'])}笔"},
                {"label": "夏普",   "value": f"{res['sharpe']:.2f}"},
            ])
            eq = res["equity"]
            eq_n = eq / eq.iloc[0]
            with get_conn() as conn:
                b = conn.execute(
                    "SELECT date,close FROM index_daily WHERE code='sh.000300' "
                    "AND date BETWEEN ? AND ? ORDER BY date",
                    (str(eq.index[0].date()), str(eq.index[-1].date()))
                ).fetchall()
            bdf = pd.DataFrame(b, columns=["date", "close"])
            bench_s = None
            if len(bdf) > 1:
                bench_s = bdf.set_index(pd.to_datetime(bdf["date"]))["close"]
                bench_s = bench_s / bench_s.iloc[0]
            with card_container("净值曲线"):
                _echarts(equity_curve(eq_n, bench_s), height=400)

elif page == "推荐追踪":
    st.header("历史推荐表现追踪")
    if st.button("刷新追踪数据"):
        update_tracking()
        st.cache_data.clear()
    s = summary_stats()
    wr5 = s["win_rate_5d"]
    a5  = s["avg_ret_5d"]
    a20 = s["avg_ret_20d"]
    metric_row([
        {"label": "累计推荐", "value": str(s["total_recommendations"]),
         "delta": f"样本{int(s['tracked_5d'])}" if s["tracked_5d"] else None},
        {"label": "5日胜率",  "value": f"{wr5*100:.0f}%" if wr5 is not None else "—",
         "delta_type": "positive" if wr5 and wr5 > 0.5 else "negative" if wr5 and wr5 < 0.45 else "neutral"},
        {"label": "5日均收益", "value": f"{a5:+.2f}%" if a5 is not None else "—",
         "delta_type": "positive" if a5 and a5 > 0 else "negative" if a5 and a5 < 0 else "neutral"},
        {"label": "20日均收益", "value": f"{a20:+.2f}%" if a20 is not None else "—",
         "delta_type": "positive" if a20 and a20 > 0 else "negative" if a20 and a20 < 0 else "neutral"},
    ])

    with get_conn() as conn:
        hist = pd.read_sql(
            "SELECT run_date AS 日期, COUNT(*) AS 只数, "
            "AVG(score) AS 平均分, AVG(ret_5d) AS 五日收益, "
            "AVG(ret_10d) AS 十日收益, AVG(ret_20d) AS 廿日收益 "
            "FROM recommendations GROUP BY run_date ORDER BY run_date DESC",
            conn)
    st.dataframe(hist, use_container_width=True, hide_index=True)

elif page == "因子进化":
    st.header("因子进化 · 过拟合防护看板")
    st.caption("自动挖掘因子的样本内(IS)/样本外(OOS)表现、净成本 IR、稳定性。"
               "OOS/IS 比率过高(>5)疑似过拟合，过低(<0.3)样本外失效，标红告警。"
               "净成本 IR(扣除交易费后的多空信息比率)<=0 的因子不可交易。")
    with get_conn() as conn:
        init_schema(conn)
        rows = conn.execute(
            "SELECT code, run_at, is_icir, oos_icir, oos_is_ratio, stability, "
            "turnover, net_ir, selected FROM factor_eval "
            "ORDER BY run_at DESC, oos_icir DESC"
        ).fetchall()
    if not rows:
        st.info("还没有因子评估结果。先运行 scripts/auto_evolve.py 写入评估。")
        st.stop()

    df = pd.DataFrame(rows, columns=["code", "run_at", "is_icir", "oos_icir",
                                     "oos_is_ratio", "stability", "turnover",
                                     "net_ir", "selected"])
    from fwsp.overfit_guard import ratio_alert
    alerts = df.apply(
        lambda r: ratio_alert(r["is_icir"], r["oos_icir"])[0], axis=1)
    df["alert"] = alerts

    with card_container("OOS/IS 比率（红=告警）"):
        thresholds = [(5.0, "#ef4444"), (0.3, "#f59e0b")]
        _echarts(ratio_bar(df, "code", "oos_is_ratio", thresholds), height=420)
    with card_container("Pareto: IS vs OOS（绿=净成本 IR 高）"):
        _echarts(pareto_scatter(df, "is_icir", "oos_icir", "net_ir"), height=380)

    # guard status banner
    n_alert = int(df["alert"].sum())
    n_bad   = int((df["net_ir"] <= 0).sum())
    if n_alert:
        alert_banner(f"⚠ {n_alert} 个因子 OOS/IS 比率告警（疑似过拟合）", "warning")
    if n_bad:
        alert_banner(f"⛔ {n_bad} 个因子净成本 IR≤0（不可交易）", "danger")

    metric_row([
        {"label": "评估因子数", "value": str(len(df))},
        {"label": "入选因子数", "value": str(int(df["selected"].sum()))},
        {"label": "告警因子数", "value": str(n_alert),
         "delta_type": "warning" if n_alert else "success"},
        {"label": "不可交易",   "value": str(n_bad),
         "delta_type": "danger" if n_bad else "success"},
    ])

    st.subheader("因子明细")
    show = df.copy()
    for c in ("is_icir", "oos_icir", "oos_is_ratio", "stability",
             "turnover", "net_ir"):
        show[c] = show[c].round(3)
    show = show[["code", "run_at", "is_icir", "oos_icir", "oos_is_ratio",
                 "net_ir", "turnover", "stability", "selected", "alert"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

elif page == "因子库":
    st.header("因子库 · 候选因子总览")
    st.caption("base.yaml（自建）+ community.yaml（WorldQuant/QLib 翻译）合并池。"
               "颜色区分来源，条形为最新 OOS ICIR，红=净成本 IR≤0 不可交易。")
    with get_conn() as conn:
        init_schema(conn)
        rows = conn.execute(
            "SELECT l.code, l.category, l.source, l.desc, "
            "e.is_icir, e.oos_icir, e.net_ir, e.stability, e.selected "
            "FROM factor_library l "
            "LEFT JOIN factor_eval e ON e.code=l.code AND e.run_at=("
            "  SELECT MAX(run_at) FROM factor_eval WHERE code=l.code)"
        ).fetchall()
    if not rows:
        st.info("因子库为空。运行 scripts/auto_evolve.py 生成因子。")
        st.stop()
    lib = pd.DataFrame(rows, columns=["code", "category", "source", "desc",
                                      "is_icir", "oos_icir", "net_ir",
                                      "stability", "selected"])

    src = st.multiselect("来源筛选", ["auto_evolve", "worldquant_alpha101",
                                       "qlib_alpha158", ""],
                         default=[])
    if src:
        lib = lib[lib["source"].isin(src)]

    with card_container("OOS ICIR（红=不可交易）"):
        _echarts(factor_bar(lib, "code", "oos_icir", color_col="net_ir"), height=420)
    metric_row([
        {"label": "因子总数", "value": str(len(lib))},
        {"label": "社区来源", "value": str(int((lib["source"] != "auto_evolve").sum()))},
        {"label": "可交易",   "value": str(int((lib["net_ir"] > 0).sum())),
         "delta_type": "success"},
    ])
    # source badges in table
    lib_display = lib.copy()
    lib_display["source_badge"] = lib_display["source"].apply(
        lambda s: badge("auto", "info") if s == "auto_evolve"
        else badge(s.split("_")[0] if s else "—", "muted"))
    show_cols = ["code", "category", "source_badge", "desc",
                 "is_icir", "oos_icir", "net_ir", "stability", "selected"]
    with card_container("因子明细"):
        st.dataframe(lib_display[show_cols].round(3), use_container_width=True,
                     hide_index=True)

elif page == "进化历史":
    st.header("进化历史 · 自动进化时间线")
    st.caption("每次 auto_evolve 的晋升记录：新 OOS 走势、入选因子数、因子选择频率。")
    with get_conn() as conn:
        init_schema(conn)
        rows = conn.execute(
            "SELECT run_at, selected_json, old_oos, new_oos, promoted, notes "
            "FROM evolution_log ORDER BY run_at"
        ).fetchall()
    if not rows:
        st.info("还没有进化记录。运行 scripts/auto_evolve.py（非 dry_run）产生。")
        st.stop()

    ev = pd.DataFrame(rows, columns=["run_at", "selected_json", "old_oos",
                                     "new_oos", "promoted", "notes"])
    ev["n_selected"] = ev["selected_json"].apply(
        lambda s: len(json.loads(s)) if s else 0)
    ev["new_oos"] = ev["new_oos"].astype(float)

    with card_container("OOS 走势（绿点=晋升）"):
        _echarts(timeline(ev), height=340)

    # 因子入选频率（跨运行）
    from collections import Counter
    cnt = Counter()
    for s in ev["selected_json"]:
        for f in json.loads(s or "[]"):
            cnt[f] += 1
    if cnt:
        fc = pd.DataFrame(cnt.items(), columns=["factor", "times_selected"])
        fc = fc.sort_values("times_selected", ascending=False)
        with card_container("因子入选频率（跨进化轮次）"):
            _echarts(freq_bar(fc), height=320)

    st.subheader("历次进化记录")
    for _, r in ev.iterrows():
        tag_html = badge("✅ 晋升", "success") if r["promoted"] else badge("⛔ 未晋升", "danger")
        with card_container(f"{r['run_at']}  OOS={r['new_oos']:.3f}  入选{int(r['n_selected'])}只  {tag_html}"):
            st.write("入选因子:", ", ".join(json.loads(r["selected_json"] or "[]")))
            if r["notes"]:
                st.caption(r["notes"])
