"""Server酱微信推送共享模块（所有定时任务 runner 统一调用）。

用法：
    from notify import send_wechat_daily
    send_wechat_daily("workbuddy", "WorkBuddy 签到", "+100 积分，连签 13 天")

约定：
    - SendKey/AppKey 存于 notify.json（600），任何日志不得输出 key
    - 推送失败只打日志，绝不抛异常影响主流程
    - send_wechat_daily 按任务名+日期去重，同日重复触发只发第一条（省免费额度）
"""

import datetime
import json
import os
import sys
import urllib.parse
import urllib.request

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(LIB_DIR, "notify.json")
API = "https://sctapi.ftqq.com/{key}.send"


def _load_key():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return (json.load(f).get("serverchan_sendkey") or "").strip()
    except Exception as e:
        print("[notify] 配置读取失败: %s" % e, file=sys.stderr)
        return ""


def send_wechat(title, desp="", key=None):
    """直接推送一条消息。返回 True/False，永不抛异常。"""
    key = key or _load_key()
    if not key:
        print("[notify] 无 key，跳过推送", file=sys.stderr)
        return False
    try:
        data = urllib.parse.urlencode(
            {"title": title[:100], "desp": desp or "-"}
        ).encode()
        req = urllib.request.Request(
            API.format(key=key), data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            ok = body.get("code") == 0
            print("[notify] %s (code=%s)" % ("OK" if ok else "FAIL", body.get("code")))
            return ok
    except Exception as e:
        print("[notify] ERROR: %s" % e, file=sys.stderr)
        return False


def send_wechat_daily(task, title, desp=""):
    """同任务同日只推一条；返回是否实际发送。"""
    marker = os.path.join(
        LIB_DIR, ".sent-%s-%s" % (task, datetime.date.today().strftime("%Y%m%d"))
    )
    if os.path.exists(marker):
        print("[notify] %s 今日已推送过，跳过" % task)
        return False
    if send_wechat(title, desp):
        with open(marker, "w") as f:
            f.write(title)
        return True
    return False


if __name__ == "__main__":
    ok = send_wechat(sys.argv[1] if len(sys.argv) > 1 else "测试",
                     sys.argv[2] if len(sys.argv) > 2 else "notify.py 测试消息")
    sys.exit(0 if ok else 1)
