# 🎆 fireworks-sp

每日 A 股候选池生成器：可解释评分 + 回测验证 + 推荐追踪闭环。

## 架构

```
数据层   akshare(东财) / 腾讯行情(兜底) / baostock(K线+指数) → SQLite
策略层   基本面硬过滤(市值/PE/ROE/负债率) + 技术面打分(均线/MACD/量价/RSI)
回测层   面板向量化周度回测: T+1开盘买、持有10日、-8%止损、含费用
展示层   Streamlit 看板 (今日推荐/个股查询/策略回测/推荐追踪)
推送层   cron 每交易日17:30 → Server酱微信
```

## 快速开始

```bash
uv sync                                        # 安装依赖
uv run python scripts/update_data.py           # 更新数据（首次自动全量拉取）
uv run python scripts/recommend.py             # 生成今日推荐
uv run python scripts/daily_pipeline.py        # 全流程（含微信推送）
uv run streamlit run scripts/dashboard.py      # 看板 http://<host>:8501
```

## 数据源容灾

| 数据 | 主源 | 兜底 |
|---|---|---|
| 行情快照 | 东财 push2 | 腾讯 qt.gtimg.cn |
| 历史K线 | 东财 kline | baostock 单会话批量 |
| 财务季度 | 东财 datacenter(yjbb/zcfz) | —（缺失字段自动跳过） |
| 指数 | baostock | — |

东财 push2 集群对本机有限流（触发后约几十分钟冷却），系统已按"少调用、可降级"设计：
每日增量不逐股请求，直接从行情快照切当日 OHLCV 追加。

## 已知限制（v1）

- K线为**不复权**价格：分红除权日指标会有小幅跳变；回测未复权计算
- 回测仅重放技术面打分（基本面门槛无历史序列），且用当前成分股池 → 存在幸存者偏差，结果**偏乐观**
- PE 分位过滤需要 spot 历史积累，v1 用静态阈值 PE∈(0,40]

## 定时任务

```
30 17 * * 1-5  ~/fireworks-sp/scripts/run_daily.sh   # cron
```

推送经 `~/.config/opencode/scripts/lib/notify.py` → Server酱，同日去重。

## 免责声明

输出为量化筛选候选池，非投资建议。股市有风险，入市需谨慎。
