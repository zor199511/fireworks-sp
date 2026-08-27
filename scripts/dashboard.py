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
    page = st.radio("页面", ["今日推荐", "个股查询", "策略回测",
                             "多因子策略", "推荐追踪"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("候选池生成器，非投资建议。\n买入需独立判断并严格止损。")

# ---------------------------------------------------------------- pages

if page == "今日推荐":
    st.header(f"今日 Top {FILTERS['top_n']} 候选")
    run_date, recos = latest_recommendations()
    if not recos:
        st.info("还没有推荐记录。先运行 scripts/recommend.py")
        st.stop()

    cols = st.columns(FILTERS["top_n"] if FILTERS['top_n'] <= 5 else 5)
    avg_score = sum(r["score"] for r in recos) / len(recos)
    kpis = [("平均评分", f"{avg_score:.1f}"),
            ("入选数", str(len(recos))),
            ("信号日期", run_date)]
    for i, (label, val) in enumerate(kpis):
        with cols[i % len(cols)]:
            st.metric(label, val)

    for r in recos:
        mv_yi = (r["metrics"].get("total_mv") or 0) / 1e8
        head = (f"#{r['rank']} {r['name']} ({r['code']})  ·  "
                f"{r['industry'] or '—'}  ·  评分 {r['score']:.0f}  ·  "
                f"¥{r['price']:.2f}")
        body = (" · ".join(r["reasons"]))
        tail = (f"PE {r['metrics'].get('pe_dyn') or '—'} | "
                f"PB {r['metrics'].get('pb') or '—'} | "
                f"ROE {r['metrics'].get('roe') or '—'}% | "
                f"市值 {mv_yi:.0f}亿")
        with st.expander(head):
            st.write(body)
            st.caption(tail)
            df = load_kline(r["code"])
            if len(df):
                st.plotly_chart(candle_chart(df), use_container_width=True)

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
        df = load_kline(code_map[pick])
        st.plotly_chart(candle_chart(df), use_container_width=True)
        with st.expander("最近行情"):
            st.dataframe(df.tail(15).iloc[:, :6].sort_values(
                "date", ascending=False).set_index("date"))

elif page == "策略回测":
    st.header("策略回测")
    st.caption("周度调仓 · Top10 等权 · T+1开盘买入 · 持有10日 · -8%止损 · "
               "含佣金/印花税/滑点。反转策略：超跌优质股企稳信号。"
               "注：基本面门槛未纳入历史重放，结果偏乐观。")
    c1, c2, c3 = st.columns(3)
    start = c1.text_input("开始日期", "2024-06-01")
    hold = c2.number_input("持有天数", 5, 40, 10)
    stop = c3.number_input("止损%", -30, -1, -8)
    if st.button("运行回测", type="primary"):
        with st.spinner("回测中…"):
            try:
                res = run_backtest(start=start, hold_days=int(hold),
                                   stop_pct=float(stop))
                st.session_state.bt = res
            except Exception as e:  # noqa: BLE001
                st.error(f"回测失败: {e}")

    res = st.session_state.get("bt")
    if res:
        k1, k2, k3, k4, k5 = st.columns(5)
        bench = f"(基准 {res['bench_return']*100:+.1f}%)" \
            if res["bench_return"] is not None else ""
        k1.metric("总收益", f"{res['total_return']*100:+.1f}%", bench)
        k2.metric("年化", f"{res['cagr']*100:+.1f}%")
        k3.metric("最大回撤", f"{res['max_drawdown']*100:.1f}%")
        k4.metric("胜率", f"{res['win_rate']*100:.0f}%",
                  f"{res['n_trades']}笔")
        k5.metric("夏普", f"{res['sharpe']:.2f}")

        eq = pd.Series(res["equity"])
        eq.index = pd.to_datetime(eq.index)
        eq_n = eq / eq.iloc[0]

        with get_conn() as conn:
            b = conn.execute(
                "SELECT date,close FROM index_daily WHERE code='sh.000300' "
                "AND date BETWEEN ? AND ? ORDER BY date",
                (res["start"], res["end"])).fetchall()
        bdf = pd.DataFrame(b, columns=["date", "close"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_n.index, y=eq_n.values,
                                 name="策略", line=dict(color="#f59e0b")))
        if len(bdf) > 1:
            bn = bdf["close"] / bdf["close"].iloc[0]
            fig.add_trace(go.Scatter(x=pd.to_datetime(bdf["date"]), y=bn,
                                     name="沪深300",
                                     line=dict(color="#64748b", width=1)))
        fig.update_layout(height=380, margin=dict(l=8, r=8, t=16, b=8),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OOS 总收益", f"{summ['oos_total_return']*100:+.1f}%")
            c2.metric("OOS 夏普", f"{summ['oos_sharpe']:.2f}")
            c3.metric("OOS 回撤", f"{summ['oos_max_drawdown']*100:.1f}%")
            c4.metric("OOS 胜率", f"{summ['oos_win_rate']*100:.0f}%")
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
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("总收益", f"{res['total_return']*100:+.1f}%")
            c2.metric("年化", f"{res['cagr']*100:+.1f}%")
            c3.metric("最大回撤", f"{res['max_drawdown']*100:.1f}%")
            c4.metric("胜率", f"{res['win_rate']*100:.0f}%",
                      f"{int(res['n_trades'])}笔")
            c5.metric("夏普", f"{res['sharpe']:.2f}")
            eq = res["equity"]
            eq_n = eq / eq.iloc[0]
            with get_conn() as conn:
                b = conn.execute(
                    "SELECT date,close FROM index_daily WHERE code='sh.000300' "
                    "AND date BETWEEN ? AND ? ORDER BY date",
                    (str(eq.index[0].date()), str(eq.index[-1].date()))
                ).fetchall()
            bdf = pd.DataFrame(b, columns=["date", "close"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq_n.index, y=eq_n.values,
                                     name="多因子", line=dict(color="#f59e0b")))
            if len(bdf) > 1:
                bn = bdf["close"] / bdf["close"].iloc[0]
                fig.add_trace(go.Scatter(x=pd.to_datetime(bdf["date"]), y=bn,
                                         name="沪深300",
                                         line=dict(color="#64748b", width=1)))
            fig.update_layout(height=380, margin=dict(l=8, r=8, t=16, b=8),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

elif page == "推荐追踪":
    st.header("历史推荐表现追踪")
    if st.button("刷新追踪数据"):
        update_tracking()
        st.cache_data.clear()
    s = summary_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计推荐", s["total_recommendations"])
    wr5 = s["win_rate_5d"]
    c2.metric("5日胜率", f"{wr5*100:.0f}%" if wr5 is not None else "—",
              f"样本{int(s['tracked_5d'])}" if s["tracked_5d"] else None)
    a5 = s["avg_ret_5d"]
    c3.metric("5日均收益", f"{a5:+.2f}%" if a5 is not None else "—")
    a20 = s["avg_ret_20d"]
    c4.metric("20日均收益", f"{a20:+.2f}%" if a20 is not None else "—")

    with get_conn() as conn:
        hist = pd.read_sql(
            "SELECT run_date AS 日期, COUNT(*) AS 只数, "
            "AVG(score) AS 平均分, AVG(ret_5d) AS 五日%, "
            "AVG(ret_10d) AS 十日%, AVG(ret_20d) AS 廿日% "
            "FROM recommendations GROUP BY run_date ORDER BY run_date DESC",
            conn)
    st.dataframe(hist, use_container_width=True, hide_index=True)
